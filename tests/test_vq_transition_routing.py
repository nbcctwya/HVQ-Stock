"""Tests for experiment 024: VQ transition-aware routing.

Covers four groups:
  * history / leakage: deterministic (instrument, datetime)-keyed historical
    code context with strict causality and loud failures;
  * code provenance: codes come from the frozen Stage 1, the current code
    comes from the live quantizer, and no learnable code embedding exists;
  * transition semantics: consecutive prototype differences only, padding
    excluded, zero state for empty history;
  * baseline equivalence: zero-initialized projection degrades the model
    bitwise to the base experiment.
"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.dataset import init_data_loader
from dataset.schema import GROUP_SLICES, TOTAL_DIM
from module.code_history import (
    _check_split_chronology,
    _index_frame,
    build_code_history,
    build_history_tables,
    encode_split_codes,
)
from module.layers.moe import FactorGatedMoE
from module.quantise import VectorQuantiser
import module.transition as transition_module
from module.transition import VQTransitionEncoder
from trainer.train_ypred import GenerateReturn
from utils import run_inference


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def make_frame(dates, instruments):
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), list(instruments)],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({
        "datetime": index.get_level_values("datetime"),
        "instrument": index.get_level_values("instrument").astype(str),
    })


def default_frames_codes():
    """2 instruments x 5 train days + 2 valid days + 2 test days."""
    instruments = ["AAA", "BBB"]
    frames, codes = {}, {}
    blocks = {
        "train": (pd.date_range("2020-01-06", periods=5), [1, 2, 3, 4, 5]),
        "valid": (pd.date_range("2021-01-04", periods=2), [6, 7]),
        "test": (pd.date_range("2022-01-03", periods=2), [8, 9]),
    }
    for split, (dates, codes_a) in blocks.items():
        frames[split] = make_frame(dates, instruments)
        codes_b = [c + 10 for c in codes_a]
        codes[split] = np.asarray(
            [code for pair in zip(codes_a, codes_b) for code in pair],
            dtype=np.int64,
        )
    return frames, codes


class FakeSampler:
    """Minimal canonical split sampler: positional rows of [T, 244]."""

    def __init__(self, seed, dates, instruments):
        generator = np.random.default_rng(seed)
        self.index = pd.MultiIndex.from_product(
            [pd.to_datetime(dates), list(instruments)],
            names=["datetime", "instrument"],
        )
        self.values = generator.standard_normal(
            (len(self.index), 20, TOTAL_DIM)
        ).astype(np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        return self.values[index].copy()

    def get_index(self):
        return self.index


class _RevINStub(nn.Module):
    def forward(self, x, mode="norm"):
        return x * 0.5 + 0.1


class _EncoderStub(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(158, dim)

    def forward(self, x):
        return torch.tanh(self.proj(x[:, -1, :]))


class StubStage1Model(nn.Module):
    """Stand-in exposing the frozen Stage 1 interface used by the builder."""

    def __init__(self, num_embed=16, dim=8):
        super().__init__()
        self.revin = _RevINStub()
        self.encoder = _EncoderStub(dim)
        self.quantizer = VectorQuantiser(
            num_embed=num_embed, embed_dim=dim, beta=0.25, distance="l2",
            anchor="probrandom", first_batch=False, contras_loss=False,
        )
        self.eval()
        for param in self.parameters():
            param.requires_grad = False


def make_samplers():
    instruments = ["AAA", "BBB", "CCC"]
    return {
        "train": FakeSampler(1, pd.date_range("2020-01-06", periods=6), instruments),
        "valid": FakeSampler(2, pd.date_range("2021-01-04", periods=3), instruments),
        "test": FakeSampler(3, pd.date_range("2022-01-03", periods=3), instruments),
    }


def tiny_config(transition=True, num_features=8, seq_len=5, num_prior=3):
    config = {
        "vqvae": {
            "num_features": num_features,
            "seq_len": seq_len,
            "hidden_size": 8,
            "num_prior_factors": num_prior,
            "vq_embed_dim": 8,
            "num_embed": 16,
            "encoder": {"num_heads": 2, "num_layers": 1},
            "quantizer": {
                "decay": 0.95, "commit_weight": 0.25, "distance": "l2",
                "anchor": "probrandom", "first_batch": False,
                "contras_loss": True,
            },
            "decoder": {"initial_T": 2, "hidden_channels": 8},
        },
        "predictor": {
            "saved_model": "unused.ckpt",
            "num_features": num_features,
            "individual": False,
            "aux_weight": 0.01,
            "aux_imp": 3,
            "kernel_size": 3,
            "n_expert": 2,
            "k": 1,
            "pred_len": 4,
            "moe_hidden": 8,
            "dropout": 0.1,
            "rank": 0,
            "target_day": 2,
            "use_prior": True,
            "transition_aware_routing": transition,
            "transition_history_len": 4,
            "transition_gru_hidden": 64,
            "transformer": {
                "num_heads": 2, "num_layers": 1, "d_model": 8,
                "dim_feedforward": 16, "dropout": 0.1, "batch_first": True,
            },
        },
        "train": {"learning_rate": 0.0001},
    }
    return config


def build_model(transition=True, num_features=8, seq_len=5, num_prior=3):
    # Only reseed the streams consumed by module construction; do not touch
    # global deterministic-algorithm flags inside the shared test process.
    torch.manual_seed(0)
    np.random.seed(0)
    with mock.patch.object(
        GenerateReturn, "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(
            tiny_config(transition, num_features, seq_len, num_prior), T_max=10
        )


# --------------------------------------------------------------------------
# History / leakage
# --------------------------------------------------------------------------

class HistoryTableTest(unittest.TestCase):
    def setUp(self):
        self.frames, self.codes = default_frames_codes()
        self.tables = build_history_tables(self.frames, self.codes, 4)

    def history_datetimes(self, split, row):
        """Recompute the datetimes whose codes were selected for a sample."""
        frame = self.frames[split]
        instrument = frame["instrument"].iloc[row]
        current_dt = frame["datetime"].iloc[row]
        visible = {"train": ["train"], "valid": ["train", "valid"],
                   "test": ["train", "valid", "test"]}[split]
        pool = []
        for s in visible:
            other = self.frames[s]
            for dt, inst, code in zip(other["datetime"], other["instrument"],
                                      self.codes[s]):
                if inst == instrument and dt < current_dt:
                    pool.append((dt, code))
        pool.sort()
        return pool

    def test_identity_mapping_one_to_one(self):
        for split in ("train", "valid", "test"):
            for row in range(len(self.frames[split])):
                pool = self.history_datetimes(split, row)
                expected = [code for _, code in pool[-4:]]
                length = self.tables[split]["hist_len"][row]
                self.assertEqual(length, len(expected))
                self.assertEqual(
                    list(self.tables[split]["hist_codes"][row][:length]),
                    expected,
                )

    def test_history_strictly_earlier(self):
        for split in ("train", "valid", "test"):
            for row in range(len(self.frames[split])):
                pool = self.history_datetimes(split, row)
                current_dt = self.frames[split]["datetime"].iloc[row]
                for dt, _ in pool:
                    self.assertLess(dt, current_dt)

    def test_no_cross_instrument_leakage(self):
        # Instrument BBB codes are all >= 11; AAA history must never see them.
        for split in ("train", "valid", "test"):
            frame = self.frames[split]
            for row in range(len(frame)):
                length = self.tables[split]["hist_len"][row]
                codes = self.tables[split]["hist_codes"][row][:length]
                if frame["instrument"].iloc[row] == "AAA":
                    self.assertTrue(all(c < 11 for c in codes))
                else:
                    self.assertTrue(all(c >= 11 for c in codes))

    def test_valid_reads_earlier_train(self):
        # AAA@valid d1 (row 0): history = last 4 train codes of AAA.
        self.assertEqual(self.tables["valid"]["hist_len"][0], 4)
        self.assertEqual(
            list(self.tables["valid"]["hist_codes"][0]), [2, 3, 4, 5]
        )
        # AAA@valid d2 (row 2): last 4 of train+valid-d1 codes.
        self.assertEqual(
            list(self.tables["valid"]["hist_codes"][2]), [3, 4, 5, 6]
        )

    def test_test_reads_earlier_train_and_valid(self):
        # AAA@test d1 (row 0): last 4 of train(5) + valid(2) codes of AAA.
        self.assertEqual(
            list(self.tables["test"]["hist_codes"][0]), [4, 5, 6, 7]
        )

    def test_train_never_reads_valid_or_test(self):
        # AAA@train d5 (row 8): only earlier train codes.
        self.assertEqual(
            list(self.tables["train"]["hist_codes"][8]), [1, 2, 3, 4]
        )

    def test_short_history_keeps_samples(self):
        # First train day has no history; the sample must still exist.
        self.assertEqual(self.tables["train"]["hist_len"][0], 0)
        self.assertEqual(self.tables["train"]["hist_len"][1], 0)
        self.assertEqual(
            len(self.tables["train"]["hist_len"]), len(self.frames["train"])
        )
        self.assertEqual(self.tables["train"]["hist_len"][2], 1)

    def test_duplicate_keys_fail(self):
        index = pd.MultiIndex.from_tuples(
            [("2020-01-06", "AAA"), ("2020-01-06", "AAA")],
            names=["datetime", "instrument"],
        )
        with self.assertRaises(ValueError):
            _index_frame(index, "train")

    def test_malformed_index_fails(self):
        with self.assertRaises(ValueError):
            _index_frame(pd.Index(["a", "b"]), "train")
        bad = pd.MultiIndex.from_product(
            [[1, 2], ["AAA"]], names=["date", "asset"]
        )
        with self.assertRaises(ValueError):
            _index_frame(bad, "train")

    def test_impossible_chronology_fails(self):
        frames, _ = default_frames_codes()
        frames["valid"] = make_frame(pd.date_range("2019-01-01", periods=2),
                                     ["AAA", "BBB"])
        with self.assertRaises(ValueError):
            _check_split_chronology(frames)


class HistoryBuilderEndToEndTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = StubStage1Model()
        self.samplers = make_samplers()

    def test_codes_match_frozen_stage1_forward(self):
        codes = encode_split_codes(
            self.model, self.samplers["train"], "train", batch_size=5
        )
        sampler = self.samplers["train"]
        for row in (0, 3, len(sampler) - 1):
            batch = torch.as_tensor(sampler[[row]], dtype=torch.float32)
            feature = batch[:, :, GROUP_SLICES["feature"]]
            with torch.no_grad():
                hidden = self.model.encoder(self.model.revin(feature, mode="norm"))
                _, _, (_, _, vq_idx) = self.model.quantizer(hidden)
            self.assertEqual(int(codes[row]), int(vq_idx[0]))

    def test_frozen_stage1_unmodified_by_build(self):
        before = {k: v.detach().clone()
                  for k, v in self.model.state_dict().items()}
        training_flag = self.model.training
        build_code_history(self.model, self.samplers, history_len=4,
                           batch_size=5)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), msg=key)
        self.assertEqual(self.model.training, training_flag)
        self.assertFalse(self.model.encoder.training)

    def test_labels_are_never_read(self):
        poisoned = make_samplers()
        for sampler in poisoned.values():
            sampler.values[:, :, GROUP_SLICES["label"]] = np.nan
        clean = build_code_history(self.model, self.samplers, history_len=4,
                                   batch_size=5)
        dirty = build_code_history(self.model, poisoned, history_len=4,
                                   batch_size=5)
        for split in ("train", "valid", "test"):
            self.assertTrue(np.array_equal(clean[split]["codes"],
                                           dirty[split]["codes"]))
            self.assertTrue(np.array_equal(clean[split]["hist_codes"],
                                           dirty[split]["hist_codes"]))

    def test_no_future_codes_anywhere(self):
        tables = build_code_history(self.model, self.samplers, history_len=4,
                                    batch_size=5)
        frames = {split: _index_frame(self.samplers[split].get_index(), split)
                  for split in ("train", "valid", "test")}
        for split in ("train", "valid", "test"):
            visible = {"train": ["train"], "valid": ["train", "valid"],
                       "test": ["train", "valid", "test"]}[split]
            frame = frames[split]
            for row in range(len(frame)):
                instrument = frame["instrument"].iloc[row]
                current_dt = frame["datetime"].iloc[row]
                # Positional re-derivation: codes of the same instrument from
                # allowed splits with datetime strictly before the sample.
                pool = []
                for s in visible:
                    other = frames[s]
                    for dt, inst, code in zip(other["datetime"],
                                              other["instrument"],
                                              tables[s]["codes"].tolist()):
                        if inst == instrument and dt < current_dt:
                            pool.append((dt, code))
                pool.sort()
                expected = [code for _, code in pool[-4:]]
                length = int(tables[split]["hist_len"][row])
                self.assertEqual(length, len(expected))
                self.assertEqual(
                    tables[split]["hist_codes"][row][:length].tolist(),
                    expected,
                )

    def test_shuffle_invariant_history_binding(self):
        tables = build_code_history(self.model, self.samplers, history_len=4,
                                    batch_size=5)

        def collect(seed):
            np.random.seed(seed)
            loader, _ = init_data_loader(
                self.samplers["train"], shuffle=True, num_workers=0,
                history=tables["train"],
            )
            mapping = {}
            for hist_codes, hist_len, data in loader:
                self.assertEqual(hist_codes.dtype, torch.long)
                self.assertEqual(hist_len.dtype, torch.long)
                self.assertEqual(data.dtype, torch.float32)
                for i in range(data.shape[0]):
                    mapping[data[i].numpy().tobytes()] = (
                        hist_codes[i].clone(), int(hist_len[i])
                    )
            return mapping

        first = collect(123)
        second = collect(999)
        self.assertEqual(set(first), set(second))
        for key in first:
            self.assertTrue(torch.equal(first[key][0], second[key][0]))
            self.assertEqual(first[key][1], second[key][1])

    def test_cache_roundtrip_and_stale_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "history.pt"
            built = build_code_history(self.model, self.samplers,
                                       history_len=4, batch_size=5,
                                       cache_path=cache, provenance_key="ckpt-v1")
            cached = build_code_history(self.model, self.samplers,
                                        history_len=4, batch_size=5,
                                        cache_path=cache, provenance_key="ckpt-v1")
            for split in ("train", "valid", "test"):
                self.assertTrue(np.array_equal(built[split]["hist_codes"],
                                               cached[split]["hist_codes"]))
            with self.assertRaises(ValueError):
                build_code_history(self.model, self.samplers, history_len=4,
                                   batch_size=5, cache_path=cache,
                                   provenance_key="ckpt-v2")


# --------------------------------------------------------------------------
# Transition semantics
# --------------------------------------------------------------------------

class TransitionEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.codebook = nn.Parameter(torch.randn(16, 8))
        self.encoder = VQTransitionEncoder(
            self.codebook, num_experts=2, history_len=4, gru_hidden=64
        )

    def test_projection_zero_initialized(self):
        self.assertEqual(int(torch.count_nonzero(self.encoder.proj.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(self.encoder.proj.bias)), 0)

    def test_zero_history_zero_state_and_bias(self):
        hist_codes = torch.zeros(3, 4, dtype=torch.long)
        hist_len = torch.zeros(3, dtype=torch.long)
        current = torch.tensor([1, 2, 3])
        state = self.encoder.transition_state(hist_codes, hist_len, current)
        self.assertTrue(torch.equal(state, torch.zeros_like(state)))
        bias = self.encoder(hist_codes, hist_len, current)
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))

    def test_zero_init_bias_for_any_history(self):
        hist_codes = torch.randint(0, 16, (5, 4))
        hist_len = torch.tensor([0, 1, 2, 3, 4])
        current = torch.randint(0, 16, (5,))
        bias = self.encoder(hist_codes, hist_len, current)
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))

    def _captured_deltas(self, hist_codes, hist_len, current):
        captured = {}
        original_pack = transition_module.pack_padded_sequence

        def spy(deltas, lengths, **kwargs):
            captured["deltas"] = deltas.detach().clone()
            captured["lengths"] = lengths.clone()
            return original_pack(deltas, lengths, **kwargs)

        with mock.patch.object(transition_module, "pack_padded_sequence", spy):
            self.encoder.transition_state(hist_codes, hist_len, current)
        return captured

    def test_transitions_are_consecutive_prototype_differences(self):
        # Trajectory: history [3, 9], current 5 -> deltas e9-e3, e5-e9.
        hist_codes = torch.tensor([[3, 9, 0, 0]])
        hist_len = torch.tensor([2])
        current = torch.tensor([5])
        captured = self._captured_deltas(hist_codes, hist_len, current)
        book = self.codebook.detach()
        expected = torch.stack([book[9] - book[3], book[5] - book[9]])
        self.assertTrue(torch.equal(captured["deltas"][0, :2], expected))
        self.assertTrue(torch.equal(
            captured["deltas"][0, 2:], torch.zeros_like(captured["deltas"][0, 2:])
        ))
        self.assertEqual(list(captured["lengths"]), [2])

    def test_same_code_repetition_gives_zero_transition(self):
        hist_codes = torch.tensor([[7, 7, 7, 7]])
        hist_len = torch.tensor([4])
        current = torch.tensor([7])
        captured = self._captured_deltas(hist_codes, hist_len, current)
        self.assertTrue(torch.equal(captured["deltas"],
                                    torch.zeros_like(captured["deltas"])))
        self.assertEqual(list(captured["lengths"]), [4])

    def test_padding_codes_are_not_encoded(self):
        current = torch.tensor([5, 5])
        hist_len = torch.tensor([1, 1])
        clean = torch.tensor([[9, 0, 0, 0], [9, 0, 0, 0]])
        dirty = torch.tensor([[9, 15, 14, 13], [9, 1, 2, 3]])
        state_clean = self.encoder.transition_state(clean, hist_len, current)
        state_dirty = self.encoder.transition_state(dirty, hist_len, current)
        self.assertTrue(torch.allclose(state_clean, state_dirty, atol=1e-6))

    def test_no_learnable_code_embedding(self):
        for module in self.encoder.modules():
            self.assertNotIsInstance(module, nn.Embedding)
        # The frozen codebook must not become a registered/serialized
        # parameter of this branch.
        self.assertNotIn("codebook_weight", self.encoder.state_dict())
        param_ids = {id(p) for p in self.encoder.parameters()}
        self.assertNotIn(id(self.codebook), param_ids)

    def test_lookup_is_detached(self):
        hist_codes = torch.tensor([[1, 2, 3, 4]])
        hist_len = torch.tensor([4])
        current = torch.tensor([5])
        bias = self.encoder(hist_codes, hist_len, current)
        bias.sum().backward()
        self.assertIsNone(self.codebook.grad)

    def test_out_of_range_codes_fail(self):
        with self.assertRaises(ValueError):
            self.encoder(torch.full((1, 4), 99), torch.tensor([1]),
                         torch.tensor([0]))

    def test_rng_isolated_construction(self):
        torch.manual_seed(7)
        before = torch.rand(4)
        torch.manual_seed(7)
        VQTransitionEncoder(self.codebook, num_experts=2)
        after = torch.rand(4)
        self.assertTrue(torch.equal(before, after))


# --------------------------------------------------------------------------
# Baseline equivalence & integration
# --------------------------------------------------------------------------

class BaselineEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.base = build_model(transition=False).eval()
        self.model = build_model(transition=True).eval()
        torch.manual_seed(3)
        self.feature = torch.randn(6, 5, 8)
        self.prior = torch.randn(6, 3)
        self.hist_codes = torch.randint(0, 16, (6, 4))
        self.hist_len = torch.tensor([0, 1, 2, 3, 4, 4])

    def test_existing_parameters_bitwise_identical(self):
        base_state = self.base.state_dict()
        new_state = self.model.state_dict()
        extra = [k for k in new_state if k.startswith("transition_encoder.")]
        self.assertTrue(extra, "transition branch must exist")
        stripped = {k: v for k, v in new_state.items()
                    if k not in set(extra)}
        self.assertEqual(set(base_state), set(stripped))
        for key, value in base_state.items():
            self.assertTrue(torch.equal(value, stripped[key]), msg=key)

    def test_zero_init_full_forward_bitwise_equal(self):
        with torch.no_grad():
            base_out = self.base(self.feature, self.prior)
            new_out = self.model(self.feature, self.prior,
                                 self.hist_codes, self.hist_len)
        for left, right in zip(base_out, new_out):
            self.assertTrue(torch.equal(left, right))

    def test_zero_init_clean_logits_equal(self):
        moe_base = self.base.loadings.fusion.moe
        moe_new = self.model.loadings.fusion.moe
        z = torch.randn(6, 8)
        with torch.no_grad():
            bias = self.model.transition_encoder(
                self.hist_codes, self.hist_len, torch.randint(0, 16, (6,))
            )
            self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))
            clean_base = moe_base.clean_routing_logits(z)
            clean_new = moe_new.clean_routing_logits(z, bias)
        self.assertTrue(torch.equal(clean_base, clean_new))

    def test_noisy_path_and_wh_unchanged(self):
        moe_base = self.base.loadings.fusion.moe
        moe_new = self.model.loadings.fusion.moe
        self.assertTrue(torch.equal(moe_base.W_h, moe_new.W_h))
        moe_base.train()
        moe_new.train()
        z = torch.randn(6, 8)
        zero_bias = torch.zeros(6, 2)
        torch.manual_seed(2024)
        gates_base, load_base = moe_base.noisy_top_k_gating(z, True)
        torch.manual_seed(2024)
        gates_new, load_new = moe_new.noisy_top_k_gating(
            z, True, transition_bias=zero_bias
        )
        self.assertTrue(torch.equal(gates_base, gates_new))
        self.assertTrue(torch.equal(load_base, load_new))

    def test_current_code_comes_from_live_quantizer(self):
        captured = {}
        original = VQTransitionEncoder.forward

        def spy(self, hist_codes, hist_len, current_code):
            captured["current"] = current_code
            return original(self, hist_codes, hist_len, current_code)

        with mock.patch.object(VQTransitionEncoder, "forward", spy):
            with torch.no_grad():
                self.model(self.feature, self.prior,
                           self.hist_codes, self.hist_len)
        with torch.no_grad():
            hidden = self.model.encoder(self.model.revin(self.feature, mode="norm"))
            _, _, (_, _, expected) = self.model.quantizer(hidden)
        self.assertTrue(torch.equal(captured["current"], expected))

    def test_codebook_reference_is_frozen_quantizer(self):
        self.assertIs(self.model.transition_encoder.codebook_weight,
                      self.model.quantizer.embedding.weight)

    def test_nonzero_projection_reroutes(self):
        encoder = self.model.transition_encoder
        current = torch.randint(0, 16, (6,))
        with torch.no_grad():
            state = encoder.transition_state(self.hist_codes, self.hist_len,
                                             current)
        # Different non-empty histories must yield different states.
        pairwise = (state[1:].unsqueeze(0) - state[1:].unsqueeze(1)).abs().max()
        self.assertGreater(float(pairwise), 0)
        # Craft the projection so the probed row routes to the opposite
        # expert of the zero-history row (whose bias stays exactly zero).
        probe_row = None
        with torch.no_grad():
            for row in range(1, state.shape[0]):
                if torch.count_nonzero(state[row]):
                    direction = state[row] / (state[row] ** 2).sum()
                    encoder.proj.weight[0] = -direction
                    encoder.proj.weight[1] = direction
                    probe_row = row
                    break
        if probe_row is None:
            self.fail("need at least one non-empty history")
        moe = self.model.loadings.fusion.moe
        for param in moe.gate.parameters():
            nn.init.zeros_(param)
        z = torch.zeros(6, 8)
        with torch.no_grad():
            bias = encoder(self.hist_codes, self.hist_len, current)
            logits = moe.clean_routing_logits(z, bias)
            gates, _ = moe.noisy_top_k_gating(z, False,
                                              transition_bias=bias)
        self.assertTrue(torch.equal(logits, bias))
        self.assertTrue(torch.equal(bias[0], torch.zeros_like(bias[0])))
        self.assertNotEqual(int(gates[0].argmax()), int(gates[probe_row].argmax()))

    def test_branch_receives_finite_gradient_and_updates(self):
        model = build_model(transition=True)
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        encoder = model.transition_encoder
        before = encoder.proj.weight.detach().clone()
        y_pred, _, _, _, aux = model(self.feature, self.prior,
                                     self.hist_codes, self.hist_len)
        label = torch.randn(6)
        loss = model.rank_loss(y_pred, label) + model.aux_weight * aux
        loss.backward()
        grad = encoder.proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)
        self.assertIsNotNone(encoder.gru.weight_ih_l0.grad)
        for module in (model.encoder, model.quantizer, model.revin):
            for param in module.parameters():
                self.assertIsNone(param.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, encoder.proj.weight.detach()))
        # Once W_t is non-zero, the GRU itself must receive a finite,
        # non-zero gradient through the transition path.
        optimizer.zero_grad()
        y_pred, _, _, _, aux = model(self.feature, self.prior,
                                     self.hist_codes, self.hist_len)
        loss = model.rank_loss(y_pred, torch.randn(6)) + model.aux_weight * aux
        loss.backward()
        gru_grad = encoder.gru.weight_ih_l0.grad
        self.assertIsNotNone(gru_grad)
        self.assertTrue(torch.isfinite(gru_grad).all())
        self.assertGreater(float(gru_grad.abs().sum()), 0)

    def test_transition_repr_does_not_reach_other_paths(self):
        model = self.model
        seen = {}

        def capture(name):
            return lambda _module, inputs: seen.setdefault(name, tuple(inputs))

        handles = [
            model.loadings.temporal_transformer.register_forward_pre_hook(capture("temporal")),
            model.latent_value_head.register_forward_pre_hook(capture("latent_head")),
            model.return_predictor.register_forward_pre_hook(capture("return_predictor")),
        ]
        with torch.no_grad():
            model(self.feature, self.prior, self.hist_codes, self.hist_len)
        for handle in handles:
            handle.remove()
        int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64,
                      torch.uint8)
        for name, inputs in seen.items():
            for tensor in inputs:
                if isinstance(tensor, torch.Tensor):
                    self.assertNotIn(tensor.dtype, int_dtypes, msg=name)

    def test_checkpoint_strict_roundtrip(self):
        import pytorch_lightning as pl
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ckpt"
            torch.save({"state_dict": self.model.state_dict(),
                        "pytorch-lightning_version": pl.__version__}, path)
            with mock.patch.object(
                GenerateReturn, "load_pretrained_vqvae",
                lambda self, checkpoint_path=None: None,
            ):
                restored = GenerateReturn.load_from_checkpoint(
                    path, config=tiny_config(True), T_max=10, strict=True
                )
            restored.eval()
            with torch.no_grad():
                expected = self.model(self.feature, self.prior,
                                      self.hist_codes, self.hist_len)
                actual = restored(self.feature, self.prior,
                                  self.hist_codes, self.hist_len)
            for left, right in zip(expected, actual):
                self.assertTrue(torch.equal(left, right))

    def test_run_inference_with_history_loader(self):
        samplers = make_samplers()
        stub = StubStage1Model()
        tables = build_code_history(stub, samplers, history_len=4,
                                    batch_size=5)
        loader, _ = init_data_loader(
            samplers["test"], shuffle=False, num_workers=0,
            history=tables["test"],
        )
        model = build_model(transition=True, num_features=158, seq_len=20,
                            num_prior=13)
        config = tiny_config(True, num_features=158, seq_len=20, num_prior=13)
        pred, _, metrics = run_inference(model, loader, config, "cpu")
        self.assertEqual(len(pred), len(samplers["test"]))
        self.assertIn("RankIC", metrics)

    def test_missing_history_arguments_fail_loudly(self):
        with self.assertRaises(ValueError):
            self.model(self.feature, self.prior)

    def test_invalid_transition_config_fails(self):
        config = tiny_config(True)
        config["predictor"]["transition_gru_hidden"] = 32
        torch.manual_seed(0)
        with mock.patch.object(
            GenerateReturn, "load_pretrained_vqvae",
            lambda self, checkpoint_path=None: None,
        ):
            with self.assertRaises(ValueError):
                GenerateReturn(config, T_max=10)


if __name__ == "__main__":
    unittest.main()
