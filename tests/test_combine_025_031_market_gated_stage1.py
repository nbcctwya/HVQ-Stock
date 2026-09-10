"""Tests for experiment 035: 031 market-gated MASTER Stage 1 + 025 Stage 2.

Experiment 035 keeps experiment 025's Stage 2 (Shared-Routed MoE +
Quantization Confidence Adapter) unchanged and replaces only Stage 1: the
original PRISM-VQ Stage 1 is swapped for experiment 031's frozen market-gated
MASTER-style Stage 1 (Market Gate -> FeatureTransform -> TAttention ->
SAttention -> TemporalAttention).
"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parents[1]
STAGE1_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "031"
    / "run"
    / "checkpoints"
    / "infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=15-val_loss=0.5977.ckpt"
)

from module.layers.encoder import (
    Gate,
    MASTERStyleEncoder,
    PositionalEncoding,
    SAttention,
    TAttention,
    TemporalAttention,
)
from module.layers.moe import FactorGatedMoE, SimpleMLP
from module.quantise import VectorQuantiser
from trainer.train_ypred import GenerateReturn


def tiny_config(adapter=True, shared_expert=True):
    return {
        "vqvae": {
            "num_features": 8,
            "seq_len": 5,
            "hidden_size": 8,
            "num_prior_factors": 3,
            "vq_embed_dim": 8,
            "num_embed": 16,
            "encoder": {
                "type": "master",
                "num_heads": 2,
                "num_layers": 1,
                "temporal_num_heads": 2,
                "spatial_num_heads": 2,
                "temporal_dropout": 0.1,
                "spatial_dropout": 0.1,
                "market_gate": {
                    "input_dim": 3,
                    "beta": {"csi300": 10, "sp500": 5},
                },
            },
            "quantizer": {
                "decay": 0.95,
                "commit_weight": 0.25,
                "distance": "l2",
                "anchor": "probrandom",
                "first_batch": False,
                "contras_loss": True,
            },
            "decoder": {"initial_T": 2, "hidden_channels": 8},
        },
        "predictor": {
            "saved_model": "unused.ckpt",
            "num_features": 8,
            "individual": False,
            "aux_weight": 0.01,
            "aux_imp": 3,
            "kernel_size": 3,
            "n_expert": 2,
            "k": 1,
            "pred_len": 4,
            "moe_hidden": 8,
            "shared_expert": shared_expert,
            "dropout": 0.1,
            "rank": 0,
            "target_day": 2,
            "use_prior": True,
            "quantization_confidence_adapter": adapter,
            "transformer": {
                "num_heads": 2,
                "num_layers": 1,
                "d_model": 8,
                "dim_feedforward": 16,
                "dropout": 0.1,
                "batch_first": True,
                "prepend_structure_token": True,
            },
        },
        "data": {"universe": "csi300"},
        "train": {"learning_rate": 0.0001},
    }


def build_model(adapter=True, shared_expert=True, seed=0):
    torch.manual_seed(seed)
    config = copy.deepcopy(tiny_config(adapter=adapter, shared_expert=shared_expert))
    with mock.patch.object(
        GenerateReturn,
        "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(config, T_max=10)


def real_config():
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    return OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )


def build_real_model():
    """Construct GenerateReturn from the default config without Stage 1 I/O."""
    config = real_config()
    config["predictor"]["saved_model"] = str(STAGE1_CHECKPOINT)
    with mock.patch.object(
        GenerateReturn,
        "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(config, T_max=10)


def capture_strict_load(model, checkpoint_path):
    """Run load_pretrained_vqvae while recording strict load statistics."""
    captured = {}
    for name, module in (
        ("encoder", model.encoder),
        ("quantizer", model.quantizer),
        ("revin", model.revin),
    ):
        original = module.load_state_dict

        def wrapper(state_dict, strict=True, _name=name, _original=original):
            result = _original(state_dict, strict)
            captured[_name] = {
                "missing": len(result.missing_keys),
                "unexpected": len(result.unexpected_keys),
            }
            return result

        module.load_state_dict = wrapper
    model.load_pretrained_vqvae(str(checkpoint_path))
    return captured


class DefaultConfigTests(unittest.TestCase):
    """The default configs/config.yaml must fully represent experiment 035."""

    def setUp(self):
        config_path = ROOT / "configs" / "config.yaml"
        with config_path.open() as stream:
            self.config = yaml.safe_load(stream)

    def test_encoder_selects_market_gated_master_stage1(self):
        encoder = self.config["vqvae"]["encoder"]
        self.assertEqual(encoder["type"], "master")
        self.assertEqual(encoder["num_heads"], 2)
        self.assertEqual(encoder["num_layers"], 1)
        self.assertEqual(encoder["temporal_num_heads"], 2)
        self.assertEqual(encoder["spatial_num_heads"], 2)
        self.assertEqual(encoder["temporal_dropout"], 0.1)
        self.assertEqual(encoder["spatial_dropout"], 0.1)
        self.assertEqual(encoder["market_gate"], {
            "input_dim": 63,
            "beta": {"csi300": 10, "sp500": 5},
        })

    def test_stage1_dimensions_unchanged(self):
        vqvae = self.config["vqvae"]
        self.assertEqual(vqvae["hidden_size"], 128)
        self.assertEqual(vqvae["vq_embed_dim"], 128)
        self.assertEqual(vqvae["num_embed"], 512)

    def test_025_stage2_mechanisms_still_enabled(self):
        predictor = self.config["predictor"]
        self.assertIs(predictor["shared_expert"], True)
        self.assertIs(predictor["quantization_confidence_adapter"], True)
        self.assertEqual(predictor["n_expert"], 2)

    def test_seed_and_data_splits_unchanged(self):
        self.assertEqual(self.config["train"]["seed"], 0)
        data = self.config["data"]
        self.assertEqual(data["universe"], "csi300")
        self.assertEqual(data["train_period"], ["2009-01-01", "2020-12-31"])
        self.assertEqual(data["valid_period"], ["2021-01-01", "2022-12-31"])
        self.assertEqual(data["test_period"], ["2023-01-01", "2025-12-31"])


class EncoderStructureTests(unittest.TestCase):
    """GenerateReturn's Stage 1 must be the 031 market-gated MASTER encoder."""

    def setUp(self):
        self.model = build_model().eval()
        self.encoder = self.model.encoder

    def test_master_encoder_present_with_required_components(self):
        master = self.encoder.master_encoder
        self.assertIsInstance(master, MASTERStyleEncoder)
        self.assertIsInstance(master.pe, PositionalEncoding)
        self.assertIsInstance(master.tatten, TAttention)
        self.assertIsInstance(master.satten, SAttention)
        self.assertIsInstance(master.temporalatten, TemporalAttention)

    def test_no_gru_and_market_gate_present(self):
        self.assertFalse(
            any(isinstance(module, nn.GRU) for module in self.encoder.modules())
        )
        self.assertIsInstance(self.encoder.feature_gate, Gate)
        self.assertEqual(self.encoder.feature_gate.trans.in_features, 3)
        self.assertEqual(self.encoder.feature_gate.trans.out_features, 8)
        self.assertEqual(self.encoder.feature_gate.t, 10)

    def test_feature_transform_and_out_layer_preserved(self):
        transform = self.encoder.feature_transform
        self.assertIsInstance(transform.linear, nn.Linear)
        self.assertIsInstance(transform.normalize, nn.LayerNorm)
        self.assertIsInstance(transform.leakyrelu, nn.LeakyReLU)
        self.assertEqual(len(self.encoder.out_layer), 3)
        self.assertIsInstance(self.encoder.out_layer[0], nn.Linear)
        self.assertIsInstance(self.encoder.out_layer[1], nn.GELU)
        self.assertIsInstance(self.encoder.out_layer[2], nn.Linear)

    def test_gate_then_attention_stack_order(self):
        encoder = self.encoder
        master = encoder.master_encoder
        seen = []
        modules = [
            ("market_gate", encoder.feature_gate),
            ("feature_transform", encoder.feature_transform),
            ("x2y", master.x2y),
            ("pe", master.pe),
            ("tatten", master.tatten),
            ("satten", master.satten),
            ("temporalatten", master.temporalatten),
            ("out_layer", encoder.out_layer),
        ]
        handles = [
            module.register_forward_hook(
                lambda _module, _inputs, _output, name=name: seen.append(name)
            )
            for name, module in modules
        ]
        try:
            with torch.no_grad():
                output = self.encoder(torch.randn(6, 5, 8), torch.randn(6, 5, 3))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(seen, [name for name, _ in modules])
        self.assertEqual(output.shape, (6, 8))

    def test_only_last_market_timestep_controls_gate_and_latent(self):
        torch.manual_seed(23)
        encoder = build_model().eval().encoder
        stock = torch.randn(6, 5, 8)
        market = torch.randn(6, 5, 3)
        changed_history = market.clone()
        changed_history[:, :-1, :] += 1000
        changed_current = market.clone()
        changed_current[:, -1, 0] += 1000

        with torch.no_grad():
            base_gate = encoder.feature_gate(market[:, -1, :])
            history_gate = encoder.feature_gate(changed_history[:, -1, :])
            current_gate = encoder.feature_gate(changed_current[:, -1, :])
            base_latent = encoder(stock, market)
            history_latent = encoder(stock, changed_history)
            current_latent = encoder(stock, changed_current)

        self.assertTrue(torch.equal(base_gate, history_gate))
        self.assertTrue(torch.equal(base_latent, history_latent))
        self.assertFalse(torch.equal(base_gate, current_gate))
        self.assertFalse(torch.equal(base_latent, current_latent))


class Stage1CheckpointTests(unittest.TestCase):
    """The 031 official Stage 1 checkpoint must strict-load with zero diffs."""

    def setUp(self):
        if not STAGE1_CHECKPOINT.is_file():
            raise unittest.SkipTest(
                f"031 official Stage 1 checkpoint missing: {STAGE1_CHECKPOINT}"
            )
        self.model = build_real_model().eval()
        self.captured = capture_strict_load(self.model, STAGE1_CHECKPOINT)

    def test_strict_load_has_no_missing_or_unexpected_keys(self):
        self.assertEqual(set(self.captured), {"encoder", "quantizer", "revin"})
        for name, counts in self.captured.items():
            self.assertEqual(counts["missing"], 0, msg=name)
            self.assertEqual(counts["unexpected"], 0, msg=name)

    def test_loaded_encoder_is_market_gated_master_structure(self):
        self.assertIsInstance(self.model.encoder.master_encoder, MASTERStyleEncoder)
        self.assertIsInstance(self.model.encoder.feature_gate, Gate)
        self.assertFalse(
            any(isinstance(module, nn.GRU) for module in self.model.encoder.modules())
        )

    def test_forward_h_is_128_dim_and_z_q_from_same_quantizer(self):
        model = self.model
        self.assertIsInstance(model.quantizer, VectorQuantiser)
        quantizers = [
            module for module in model.modules() if isinstance(module, VectorQuantiser)
        ]
        self.assertEqual(len(quantizers), 1)

        feature = torch.randn(4, 20, 158)
        prior = torch.randn(4, 13)
        market = torch.randn(4, 20, 63)
        seen = []
        handle = model.encoder.register_forward_hook(
            lambda _module, _inputs, output: seen.append(output.detach().clone())
        )
        with torch.no_grad():
            y_pred, _, _, z_stage2, _ = model(feature, prior, market)
        handle.remove()

        self.assertEqual(len(seen), 1)
        h_batch = seen[0]
        self.assertEqual(h_batch.shape, (4, 128))

        with torch.no_grad():
            z_q = model.quantizer(h_batch)[0]
        self.assertEqual(z_q.shape, (4, 128))
        # Zero-initialized adapter: z_conf is bitwise equal to z_q.
        self.assertTrue(torch.equal(z_stage2, z_q))
        self.assertEqual(y_pred.shape, (4,))

    def test_quantization_error_matches_required_definition(self):
        h = torch.randn(6, 128)
        z_q = torch.randn(6, 128)
        actual = GenerateReturn.quantization_error(h, z_q)
        expected = torch.mean((h - z_q) ** 2, dim=-1, keepdim=True)
        self.assertEqual(actual.shape, (6, 1))
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(actual.requires_grad)


class Stage1FreezeTests(unittest.TestCase):
    """Stage 1 (encoder/quantizer/revin) must stay frozen and in eval mode."""

    def setUp(self):
        self.model = build_model().eval()
        self.feature = torch.randn(11, 5, 8)
        self.prior = torch.randn(11, 3)
        self.market = torch.randn(11, 5, 3)

    def test_stage1_parameters_frozen_and_eval(self):
        for module in (self.model.encoder, self.model.quantizer, self.model.revin):
            self.assertFalse(module.training)
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)

    def test_backward_isolates_stage1_and_trains_adapter(self):
        model = self.model
        model.train()
        # The train() override must keep frozen Stage 1 modules in eval mode.
        for module in (model.encoder, model.quantizer, model.revin):
            self.assertFalse(module.training)

        y_pred, _, _, _, aux_loss = model(self.feature, self.prior, self.market)
        label = torch.randn(11)
        loss = model.rank_loss(y_pred, label) + model.aux_weight * aux_loss
        loss.backward()

        for module in (model.encoder, model.quantizer, model.revin):
            for name, parameter in module.named_parameters():
                self.assertIsNone(parameter.grad, msg=name)

        adapter = model.quantization_confidence_adapter
        self.assertIsNotNone(adapter.weight.grad)
        self.assertIsNotNone(adapter.bias.grad)
        self.assertTrue(torch.isfinite(adapter.weight.grad).all())
        self.assertTrue(torch.isfinite(adapter.bias.grad).all())
        self.assertGreater(adapter.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(adapter.bias.grad.abs().sum().item(), 0.0)


class Stage2StructureTests(unittest.TestCase):
    """The 025 Stage 2 (010 Shared-Routed MoE) structure must be intact."""

    def setUp(self):
        self.model = build_model().eval()
        self.moe = self.model.loadings.fusion.moe

    def test_moe_is_factor_gated_with_shared_expert(self):
        moe = self.moe
        self.assertIsInstance(moe, FactorGatedMoE)
        self.assertIsInstance(moe.shared_expert, SimpleMLP)
        self.assertEqual(repr(moe.shared_expert), repr(moe.experts[0]))
        self.assertNotIn(moe.shared_expert, list(moe.experts))

    def test_routed_configuration_unchanged(self):
        moe = self.moe
        self.assertEqual(moe.num_experts, 2)
        self.assertEqual(moe.k, 1)
        self.assertEqual(len(moe.experts), 2)
        self.assertTrue(moe.noisy_gating)
        self.assertTrue(hasattr(moe, "gate"))
        self.assertTrue(hasattr(moe, "noise"))
        self.assertTrue(hasattr(moe, "W_h"))
        self.assertTrue(hasattr(moe, "softplus"))
        self.assertTrue(hasattr(moe, "mean"))
        self.assertTrue(hasattr(moe, "std"))

    def test_confidence_adapter_is_the_only_extra_module(self):
        base = build_model(adapter=False, seed=11)
        adapted = build_model(adapter=True, seed=11)

        base_modules = {name for name, _ in base.named_modules()}
        adapted_modules = {name for name, _ in adapted.named_modules()}
        self.assertEqual(adapted_modules - base_modules, {"quantization_confidence_adapter"})
        self.assertEqual(base_modules - adapted_modules, set())

        adapter = adapted.quantization_confidence_adapter
        self.assertIsInstance(adapter, nn.Linear)
        self.assertEqual(adapter.in_features, 1)
        self.assertEqual(adapter.out_features, 8)
        self.assertEqual(torch.count_nonzero(adapter.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.bias).item(), 0)

    def test_zero_init_stage2_latent_bitwise_equal_z_q(self):
        model = build_model().eval()
        feature = torch.randn(9, 5, 8)
        market = torch.randn(9, 5, 3)
        with torch.no_grad():
            h_batch = model.encoder(model.revin(feature, mode="norm"), market)
            z_q = model.quantizer(h_batch)[0]
        z_stage2 = model.build_stage2_latent(h_batch, z_q)
        self.assertTrue(torch.equal(z_stage2, z_q))


class Stage2CheckpointRoundTripTests(unittest.TestCase):
    """A Stage 2 checkpoint must strict-load and reproduce outputs bitwise."""

    def test_strict_checkpoint_round_trip(self):
        model = build_model().eval()
        feature = torch.randn(6, 5, 8)
        prior = torch.randn(6, 3)
        market = torch.randn(6, 5, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "stage2.ckpt"
            torch.save(
                {
                    "epoch": 0,
                    "global_step": 1,
                    "pytorch-lightning_version": pl.__version__,
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            with mock.patch.object(
                GenerateReturn,
                "load_pretrained_vqvae",
                lambda self, checkpoint_path=None: None,
            ):
                restored = GenerateReturn.load_from_checkpoint(
                    str(checkpoint_path),
                    config=copy.deepcopy(tiny_config()),
                    T_max=10,
                    strict=True,
                )
        restored.freeze_vqvae()
        restored.eval()
        with torch.no_grad():
            reference = model(feature, prior, market)
            reloaded = restored(feature, prior, market)
        for reference_value, reloaded_value in zip(reference, reloaded):
            self.assertTrue(torch.equal(reference_value, reloaded_value))


if __name__ == "__main__":
    unittest.main()
