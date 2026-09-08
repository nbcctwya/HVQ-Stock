"""Minimal synthetic + real-history smoke for experiment 024.

Uses the exact corrected PRISM-VQ Stage 1 checkpoint, synthetic canonical
244-field splits for the training-equivalence checks, and the real canonical
CSI300 pickles for the historical-code context diagnostics.  No formal
training or backtest is run.
"""

import copy
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from unittest import mock

# Match stage2.py: must be set before PyTorch is imported so that
# torch.use_deterministic_algorithms(True) works with cuBLAS on CUDA.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_qlib import _normalize_prediction_frame
from dataset.dataset import init_data_loader
from dataset.schema import TOTAL_DIM
from module.code_history import build_code_history
from module.layers.moe import FactorGatedMoE
from module.quantise import VectorQuantiser
from module.transition import VQTransitionEncoder
from trainer.train_ypred import GenerateReturn
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "024" / "smoke"
STAGE1_CHECKPOINT = (
    ROOT / "artifacts" / "baseline" / "run" / "checkpoints"
    / "infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt"
)
EXPECTED_CHECKPOINT_BYTES = 14584929
EXPECTED_CHECKPOINT_MD5 = "6b9d9dbfd938c7bd2c7dc5ee33cb38af"
EXPECTED_SPLITS = {
    "train_period": ["2009-01-01", "2020-12-31"],
    "valid_period": ["2021-01-01", "2022-12-31"],
    "test_period": ["2023-01-01", "2025-12-31"],
}
REAL_PICKLES = {
    split: ROOT / "dataset" / "processed" / "CN" / f"csi300_20_h10_{split}.pkl"
    for split in ("train", "valid", "test")
}


class SyntheticCanonicalDataset(torch.utils.data.Dataset):
    def __init__(self, seed, start_date, days=6, stocks_per_day=6):
        generator = torch.Generator().manual_seed(seed)
        count = days * stocks_per_day
        self.values = torch.randn(count, 20, TOTAL_DIM, generator=generator)
        dates = pd.bdate_range(start_date, periods=days)
        self.index = pd.MultiIndex.from_product(
            [dates, [f"SMOKE{i:03d}" for i in range(stocks_per_day)]],
            names=["datetime", "instrument"],
        )

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index].numpy().copy()

    def get_index(self):
        return self.index


def make_synthetic_samplers():
    return {
        "train": SyntheticCanonicalDataset(10, "2020-01-06", days=6),
        "valid": SyntheticCanonicalDataset(20, "2021-01-04", days=3),
        "test": SyntheticCanonicalDataset(30, "2022-01-03", days=3),
    }


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(transition=True):
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )
    config["predictor"]["saved_model"] = str(STAGE1_CHECKPOINT)
    config["predictor"]["transition_aware_routing"] = transition
    return config


def build_model(transition=True):
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    return GenerateReturn(copy.deepcopy(build_config(transition)), T_max=2)


def expected_history(samplers, tables, split, row, history_len=4):
    """Positional re-derivation of one sample's history from the tables."""
    index = samplers[split].get_index()
    instrument = index[row][1]
    current_dt = index[row][0]
    visible = {"train": ["train"], "valid": ["train", "valid"],
               "test": ["train", "valid", "test"]}[split]
    pool = []
    for s in visible:
        other = samplers[s].get_index()
        for pos in range(len(other)):
            dt, inst = other[pos]
            if inst == instrument and dt < current_dt:
                pool.append((dt, int(tables[s]["codes"][pos])))
    pool.sort()
    return [code for _, code in pool[-history_len:]]


def history_diagnostics(tables, samplers):
    """Phase 2 diagnostics: history length distribution, adjacent code change
    rate, zero-transition ratio, and the number of valid transition sequences."""
    report = {}
    for split, table in tables.items():
        hist = table["hist_codes"].numpy()
        lens = table["hist_len"].numpy()
        codes = table["codes"].numpy()
        n = len(lens)
        length_dist = {str(k): int((lens == k).sum()) for k in range(5)}
        has_history = lens > 0
        # Adjacent code change rate P(k_t != k_{t-1}) over samples with history.
        last_hist = hist[np.arange(n), np.clip(lens - 1, 0, None)]
        change_rate = float((last_hist[has_history] != codes[has_history]).mean()) \
            if has_history.any() else 0.0
        # Zero-transition ratio over all valid transitions.
        slots = np.arange(3)
        within_valid = slots[None, :] < (lens - 1)[:, None]
        within_same = ((hist[:, 1:] == hist[:, :-1]) & within_valid).sum()
        last_same = ((last_hist == codes) & has_history).sum()
        total_transitions = int(lens.sum())
        zero_transitions = int(within_same + last_same)
        report[split] = {
            "samples": n,
            "history_length_distribution": length_dist,
            "adjacent_code_change_rate": change_rate,
            "zero_transition_ratio": (
                zero_transitions / total_transitions if total_transitions else 0.0
            ),
            "valid_transition_sequences": int(has_history.sum()),
            "total_valid_transitions": total_transitions,
        }
    return report


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "vq_transition_routing_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not STAGE1_CHECKPOINT.is_file() or STAGE1_CHECKPOINT.stat().st_size == 0:
        raise FileNotFoundError(f"External Stage 1 checkpoint missing: {STAGE1_CHECKPOINT}")
    checkpoint_bytes = STAGE1_CHECKPOINT.stat().st_size
    checkpoint_md5 = file_md5(STAGE1_CHECKPOINT)
    if checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES:
        raise AssertionError("External Stage 1 checkpoint size changed")
    if checkpoint_md5 != EXPECTED_CHECKPOINT_MD5:
        raise AssertionError("External Stage 1 checkpoint hash changed")

    config = build_config(True)
    if config["predictor"]["transition_aware_routing"] is not True:
        raise AssertionError("Default config must enable experiment 024")
    if config["predictor"]["transition_history_len"] != 4 \
            or config["predictor"]["transition_gru_hidden"] != 64:
        raise AssertionError("Experiment 024 fixes history_len=4 and GRU hidden=64")
    if config["vqvae"]["num_embed"] != 512 or config["vqvae"]["vq_embed_dim"] != 128:
        raise AssertionError("Stage 1 must remain single VQ512 with 128-dimensional codes")
    for key, expected in EXPECTED_SPLITS.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    base = build_model(False).eval()
    model = build_model(True).eval()
    quantizers = [module for module in model.modules() if isinstance(module, VectorQuantiser)]
    if len(quantizers) != 1 or tuple(model.quantizer.embedding.weight.shape) != (512, 128):
        raise AssertionError("Stage 1 is not the required single VQ512 configuration")

    # ---- historical-code context on synthetic canonical splits ----
    samplers = make_synthetic_samplers()
    tables = build_code_history(model, samplers, history_len=4, batch_size=12)
    history_ok = True
    for split in ("train", "valid", "test"):
        for row in range(len(samplers[split])):
            expected = expected_history(samplers, tables, split, row)
            length = int(tables[split]["hist_len"][row])
            actual = tables[split]["hist_codes"][row][:length].tolist()
            if actual != expected:
                history_ok = False
    if not history_ok:
        raise AssertionError("Historical code context violates causality/identity")

    def collect(seed):
        np.random.seed(seed)
        loader, _ = init_data_loader(samplers["train"], shuffle=True,
                                     num_workers=0, history=tables["train"])
        mapping = {}
        for hist_codes, hist_len, data in loader:
            for i in range(data.shape[0]):
                mapping[data[i].numpy().tobytes()] = (
                    hist_codes[i].clone(), int(hist_len[i]))
        return mapping

    first, second = collect(1), collect(2)
    shuffle_invariant = set(first) == set(second) and all(
        torch.equal(first[key][0], second[key][0])
        and first[key][1] == second[key][1]
        for key in first
    )
    if not shuffle_invariant:
        raise AssertionError("History binding depends on date shuffle order")

    # ---- baseline equivalence at zero init ----
    branch_prefix = "transition_encoder."
    adapted_state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith(branch_prefix)}
    if set(base.state_dict()) != set(adapted_state):
        raise AssertionError("Transition branch changed baseline state keys")
    existing_init_exact = all(
        torch.equal(value, adapted_state[key])
        for key, value in base.state_dict().items()
    )
    if not existing_init_exact:
        raise AssertionError("Transition branch perturbed baseline initialization")

    encoder = model.transition_encoder
    proj_zero_init = bool(
        tuple(encoder.proj.weight.shape) == (config["predictor"]["n_expert"], 64)
        and torch.count_nonzero(encoder.proj.weight.detach()) == 0
        and torch.count_nonzero(encoder.proj.bias.detach()) == 0
    )
    if not proj_zero_init:
        raise AssertionError("W_t must be zero-initialized [n_expert, 64] with zero bias")
    if tuple(encoder.gru.weight_ih_l0.shape) != (3 * 64, 128):
        raise AssertionError("Transition GRU must be single-layer input=128 hidden=64")
    gru_is_single_layer_uni = not hasattr(encoder.gru, "weight_ih_l2") \
        and not encoder.gru.bidirectional
    if not gru_is_single_layer_uni:
        raise AssertionError("Transition encoder must be a single-layer unidirectional GRU")

    # Mix zero-history rows (first day) with full-history rows (last day) so
    # both退化 paths and真实 transition paths are exercised.
    rows = [0, 1, 2, 30, 31, 32]
    train_batch = samplers["train"].values[rows].float()
    from dataset.schema import unpack_batch
    unpacked = unpack_batch(train_batch)
    feature, prior = unpacked.stock_feature, unpacked.prior_factor
    hist_codes = tables["train"]["hist_codes"][rows]
    hist_len = tables["train"]["hist_len"][rows]

    # vq_idx pass-through: the current code must come from the live quantizer.
    captured = {}
    original_forward = VQTransitionEncoder.forward

    def spy(self, codes_arg, len_arg, current_code):
        captured["current"] = current_code
        return original_forward(self, codes_arg, len_arg, current_code)

    with mock.patch.object(VQTransitionEncoder, "forward", spy):
        with torch.no_grad():
            model(feature, prior, hist_codes, hist_len)
    with torch.no_grad():
        h_batch = model.encoder(model.revin(feature, mode="norm"))
        _, _, (_, _, expected_vq_idx) = model.quantizer(h_batch)
    vq_idx_live = torch.equal(captured["current"], expected_vq_idx)
    if not vq_idx_live:
        raise AssertionError("Current code did not come from the live quantizer")

    with torch.no_grad():
        bias = model.transition_encoder(hist_codes, hist_len, expected_vq_idx)
    zero_bias_exact = bool(torch.equal(bias, torch.zeros_like(bias)))

    z = torch.randn(6, 128)
    x = torch.randn(6, 64)
    moe = model.loadings.fusion.moe
    base_moe = base.loadings.fusion.moe
    with torch.no_grad():
        clean_base = base_moe.clean_routing_logits(z)
        clean_adapted = moe.clean_routing_logits(z, bias)
        base_gates, base_load = base_moe.noisy_top_k_gating(z, False)
        adapted_gates, adapted_load = moe.noisy_top_k_gating(
            z, False, transition_bias=bias)
        base_moe_output, base_moe_aux = base_moe(x, z)
        adapted_moe_output, adapted_moe_aux = moe(x, z, transition_bias=bias)
        base_output = base(feature, prior)
        adapted_output = model(feature, prior, hist_codes, hist_len)

    zero_clean_exact = torch.equal(clean_base, clean_adapted)
    zero_routing_exact = (torch.equal(base_gates, adapted_gates)
                          and torch.equal(base_load, adapted_load))
    zero_moe_exact = (torch.equal(base_moe_output, adapted_moe_output)
                      and torch.equal(base_moe_aux, adapted_moe_aux))
    zero_forward_exact = all(
        torch.equal(left, right) for left, right in zip(base_output, adapted_output)
    )
    if not all((zero_bias_exact, zero_clean_exact, zero_routing_exact,
                zero_moe_exact, zero_forward_exact)):
        raise AssertionError("Zero-init routing is not bitwise equivalent to baseline")

    base_moe.train()
    moe.train()
    torch.manual_seed(2024)
    noisy_base_gates, noisy_base_load = base_moe.noisy_top_k_gating(z, True)
    torch.manual_seed(2024)
    noisy_adapted_gates, noisy_adapted_load = moe.noisy_top_k_gating(
        z, True, transition_bias=torch.zeros(6, config["predictor"]["n_expert"])
    )
    noisy_behavior_exact = (
        torch.equal(noisy_base_gates, noisy_adapted_gates)
        and torch.equal(noisy_base_load, noisy_adapted_load)
        and torch.equal(base_moe.W_h, moe.W_h)
    )
    if not noisy_behavior_exact:
        raise AssertionError("Original noisy top-k/noise/W_h/load behavior changed")

    # ---- backward / update ----
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["train"]["learning_rate"], weight_decay=1e-5,
    )
    proj_before = encoder.proj.weight.detach().clone()
    optimizer.zero_grad()
    y_pred, _, _, _, aux_loss = model(feature, prior, hist_codes, hist_len)
    loss = model.rank_loss(y_pred, unpacked.target(5)) + model.aux_weight * aux_loss
    loss.backward()
    proj_grad_l1 = float(encoder.proj.weight.grad.abs().sum())
    if proj_grad_l1 <= 0 or not torch.isfinite(encoder.proj.weight.grad).all():
        raise AssertionError("W_t did not receive a valid gradient")
    frozen_stage1_has_grad = any(
        p.grad is not None
        for module in (model.encoder, model.quantizer, model.revin)
        for p in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 received gradients")
    codebook_before = model.quantizer.embedding.weight.detach().clone()
    optimizer.step()
    proj_updated = not torch.equal(proj_before, encoder.proj.weight.detach())
    if not proj_updated:
        raise AssertionError("W_t did not update")
    if not torch.equal(codebook_before, model.quantizer.embedding.weight.detach()):
        raise AssertionError("Frozen codebook changed after optimizer step")
    optimizer.zero_grad()
    y_pred, _, _, _, aux_loss = model(feature, prior, hist_codes, hist_len)
    loss = model.rank_loss(y_pred, unpacked.target(5)) + model.aux_weight * aux_loss
    loss.backward()
    gru_grad_l1 = float(encoder.gru.weight_ih_l0.grad.abs().sum())
    if gru_grad_l1 <= 0 or not torch.isfinite(encoder.gru.weight_ih_l0.grad).all():
        raise AssertionError("Transition GRU did not receive a valid gradient")

    # ---- non-zero projection re-routes experts at fixed z_q ----
    model.eval()
    probe = copy.deepcopy(model).eval()
    probe_enc = probe.transition_encoder
    current = expected_vq_idx
    with torch.no_grad():
        state = probe_enc.transition_state(hist_codes, hist_len, current)
    reroute_ok = False
    for row in range(1, state.shape[0]):
        if torch.count_nonzero(state[row]):
            with torch.no_grad():
                direction = state[row] / (state[row] ** 2).sum()
                probe_enc.proj.weight[0] = -direction
                probe_enc.proj.weight[1] = direction
                probe_enc.proj.bias.zero_()
                for param in probe.loadings.fusion.moe.gate.parameters():
                    torch.nn.init.zeros_(param)
                bias_probe = probe_enc(hist_codes, hist_len, current)
                logits = probe.loadings.fusion.moe.clean_routing_logits(
                    torch.zeros(6, 128), bias_probe)
                gates, _ = probe.loadings.fusion.moe.noisy_top_k_gating(
                    torch.zeros(6, 128), False, transition_bias=bias_probe)
            zero_row = next(r for r in range(6) if int(hist_len[r]) == 0)
            reroute_ok = bool(
                torch.equal(logits, bias_probe)
                and torch.equal(bias_probe[zero_row],
                                torch.zeros_like(bias_probe[zero_row]))
                and int(gates[zero_row].argmax()) != int(gates[row].argmax())
            )
            break
    if not reroute_ok:
        raise AssertionError("Non-zero W_t did not change routing at fixed z_q")

    # ---- isolation: history codes / bias reach only the routing logits ----
    seen = {}

    def capture(name):
        return lambda _module, inputs: seen.setdefault(name, tuple(inputs))

    handles = [
        model.loadings.temporal_transformer.register_forward_pre_hook(capture("temporal")),
        model.latent_value_head.register_forward_pre_hook(capture("latent_head")),
        model.return_predictor.register_forward_pre_hook(capture("return_predictor")),
    ]
    with torch.no_grad():
        isolated_output = model(feature, prior, hist_codes, hist_len)
    for handle in handles:
        handle.remove()
    int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    codes_isolated = not any(
        isinstance(tensor, torch.Tensor) and tensor.dtype in int_dtypes
        for inputs in seen.values() for tensor in inputs
    )
    if not codes_isolated:
        raise AssertionError("Code ids bypassed the router into other prediction paths")

    # ---- GPU deterministic backward probe (formal run uses
    # torch.use_deterministic_algorithms(True) + cuDNN) ----
    gpu_probe = "skipped (no CUDA)"
    if torch.cuda.is_available():
        # (a) transition branch alone: cuDNN GRU backward under deterministic
        # algorithms must work, since the formal Stage 2 run trains this GRU.
        # Deepcopy the whole model so the branch's codebook reference tracks
        # the (in-place moved) quantizer Parameter, exactly as in the real
        # training flow.
        probe_model = copy.deepcopy(model).to("cuda")
        probe_model.train()  # cuDNN RNN backward requires training mode;
        # the train() override keeps the frozen Stage 1 modules in eval.
        probe_enc = probe_model.transition_encoder
        try:
            bias_gpu = probe_enc(hist_codes.cuda(), hist_len.cuda(),
                                 expected_vq_idx.cuda())
            bias_gpu.sum().backward()
            gru_ok = torch.isfinite(probe_enc.gru.weight_ih_l0.grad).all()
        except RuntimeError as error:
            raise AssertionError(
                f"GPU deterministic GRU backward failed: {error}"
            )
        if not gru_ok:
            raise AssertionError("GPU GRU backward produced non-finite grads")
        # (b) full model forward+backward on GPU.
        gpu_model = probe_model
        try:
            out = gpu_model(feature.cuda(), prior.cuda(),
                            hist_codes.cuda(), hist_len.cuda())
            (out[0].sum() + out[4]).backward()
            gpu_probe = "PASS"
        except RuntimeError as error:
            raise AssertionError(f"GPU deterministic model backward failed: {error}")
        del gpu_model
        torch.cuda.empty_cache()

    # ---- checkpoint strict round-trip + standard inference + metrics ----
    checkpoint_path = checkpoint_dir / "vq-transition-routing-smoke.ckpt"
    torch.save(
        {
            "epoch": 0,
            "global_step": 1,
            "pytorch-lightning_version": pl.__version__,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    restored = GenerateReturn.load_from_checkpoint(
        checkpoint_path, config=copy.deepcopy(config), T_max=2, strict=True
    ).eval()
    with torch.no_grad():
        restored_output = restored(feature, prior, hist_codes, hist_len)
    checkpoint_exact = all(
        torch.equal(left, right)
        for left, right in zip(isolated_output, restored_output)
    )
    if not checkpoint_exact:
        raise AssertionError("Strict checkpoint round-trip changed output")

    test_loader, _ = init_data_loader(samplers["test"], shuffle=False,
                                      num_workers=0, history=tables["test"])
    prediction, _, metrics = run_inference(restored, test_loader, config,
                                           device="cpu")
    prediction_path = result_dir / "0_best.pkl"
    metric_path = result_dir / "0_metric.csv"
    prediction.to_pickle(prediction_path)
    pd.DataFrame([metrics], index=["values"]).transpose().to_csv(metric_path)
    normalized, signal = _normalize_prediction_frame(prediction_path)
    if len(normalized) != len(samplers["test"]) or len(signal) != len(samplers["test"]):
        raise AssertionError("Prediction output is incompatible with backtest normalizer")

    # ---- real canonical history build (leakage validation + diagnostics) ----
    real_report = {}
    if all(path.is_file() for path in REAL_PICKLES.values()):
        real_samplers = {
            split: pickle.load(open(path, "rb"))
            for split, path in REAL_PICKLES.items()
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        real_tables = build_code_history(
            model, real_samplers, history_len=4, batch_size=2048,
            device=device,
            cache_path=checkpoint_dir / "vq_code_history.pt",
            provenance_key=f"{STAGE1_CHECKPOINT.name}:{checkpoint_md5}",
        )
        model.to("cpu")
        real_report = history_diagnostics(real_tables, real_samplers)
    else:
        real_report = {"status": "real pickles missing, skipped"}

    report = {
        "status": "PASS",
        "stage1": {
            "source": "external",
            "checkpoint": str(STAGE1_CHECKPOINT.relative_to(ROOT)),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_md5": checkpoint_md5,
            "strict_load": "strict=True (missing=0/unexpected=0 or load raises)",
            "single_vq512": True,
            "embedding_dimension": 128,
            "data_splits": EXPECTED_SPLITS,
            "frozen_parameters_have_grad": frozen_stage1_has_grad,
        },
        "transition_routing": {
            "gru": "nn.GRU(input=128, hidden=64, layers=1, unidirectional)",
            "projection": "zero-init nn.Linear(64, n_expert)",
            "proj_zero_initialized": proj_zero_init,
            "zero_init_bias_is_zero": zero_bias_exact,
            "history_context_causal": history_ok,
            "shuffle_invariant_history": shuffle_invariant,
            "current_code_from_live_quantizer": vq_idx_live,
            "clean_logits_bitwise_equal_base": bool(zero_clean_exact),
            "routing_and_load_bitwise_equal_base": bool(zero_routing_exact),
            "moe_output_and_aux_bitwise_equal_base": bool(zero_moe_exact),
            "full_forward_bitwise_equal_base": bool(zero_forward_exact),
            "existing_init_bitwise_equal_base": existing_init_exact,
            "noisy_topk_noise_wh_load_bitwise_equal_base": bool(noisy_behavior_exact),
            "code_ids_isolated_to_router": codes_isolated,
            "nonzero_projection_reroutes": reroute_ok,
            "proj_grad_l1": proj_grad_l1,
            "gru_grad_l1_after_first_update": gru_grad_l1,
            "proj_updated": proj_updated,
            "gpu_deterministic_backward": gpu_probe,
        },
        "diagnostics": {
            "synthetic": history_diagnostics(tables, samplers),
            "real": real_report,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "strict_round_trip": True,
            "output_bitwise_equal": bool(checkpoint_exact),
        },
        "standard_outputs": {
            "prediction": str(prediction_path.relative_to(ROOT)),
            "metric": str(metric_path.relative_to(ROOT)),
            "rows": len(prediction),
            "metrics": {key: float(value) for key, value in metrics.items()},
            "backtest_normalizer": "PASS",
        },
    }
    report_path = ARTIFACT_ROOT / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
