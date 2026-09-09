"""Minimal synthetic smoke for experiment 026 (010 Shared-Routed MoE + 020
market-conditioned routing).

Uses the experiment 010 formal Stage 1 provenance (which reuses the exact
corrected PRISM-VQ Stage 1 checkpoint) and canonical 244-field inputs.
No formal training or backtest is run.

Besides the standard compatibility checks this script records read-only
routing diagnostics for later Phase 2 analysis (market bias magnitude vs
original router logits, expert-switch behavior, per-date expert usage).
The diagnostic helpers below never participate in forward prediction or
the training objective.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_qlib import _normalize_prediction_frame
from dataset.schema import GROUP_SLICES, TOTAL_DIM, unpack_batch
from module.quantise import VectorQuantiser
from trainer.train_ypred import GenerateReturn
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "026" / "smoke"
STAGE1_MARKER = ROOT / "artifacts" / "010" / "run" / ".stage1.done"
STAGE1_CHECKPOINT = (
    ROOT / "artifacts" / "baseline" / "run" / "checkpoints"
    / "infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt"
)
EXPECTED_STAGE1_COMMIT = "9b854f0436f8a7c3283fd375661dd6152cc965f1"
EXPECTED_CHECKPOINT_BYTES = 14584929
EXPECTED_CHECKPOINT_MD5 = "6b9d9dbfd938c7bd2c7dc5ee33cb38af"
EXPECTED_SPLITS = {
    "train_period": ["2009-01-01", "2020-12-31"],
    "valid_period": ["2021-01-01", "2022-12-31"],
    "test_period": ["2023-01-01", "2025-12-31"],
}


class SyntheticCanonicalDataset(Dataset):
    def __init__(self, seed, start_date, days=2, stocks_per_day=6):
        generator = torch.Generator().manual_seed(seed)
        count = days * stocks_per_day
        self.values = torch.randn(count, 20, TOTAL_DIM, generator=generator)
        for day in range(days):
            start = day * stocks_per_day
            stop = start + stocks_per_day
            common_market = torch.randn(20, 63, generator=generator)
            self.values[start:stop, :, GROUP_SLICES["market"]] = common_market
        dates = pd.bdate_range(start_date, periods=days)
        self.index = pd.MultiIndex.from_product(
            [dates, [f"SMOKE{i:03d}" for i in range(stocks_per_day)]],
            names=["datetime", "instrument"],
        )

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def get_index(self):
        return self.index


# ---------------------------------------------------------------------------
# Read-only diagnostic helpers (never used by forward prediction or training)
# ---------------------------------------------------------------------------

def expert_assignments(moe, z, market_state=None):
    """Top-1 routed expert index per sample for the given routing inputs."""
    gates, _ = moe.noisy_top_k_gating(z, False, market_state=market_state)
    return gates.argmax(1)


def expert_switch_rate(moe, z, market_state, reference_moe=None):
    """Fraction of samples whose top-1 routed expert changes when the market
    bias is applied. ``reference_moe`` is the 010-equivalent (market-off)
    module; defaults to this module's original router logits without bias."""
    with torch.no_grad():
        if reference_moe is not None:
            reference = expert_assignments(reference_moe, z)
        else:
            reference = moe.gate(z).argmax(1)
        routed = expert_assignments(moe, z, market_state=market_state)
    return float((reference != routed).float().mean())


def routing_diagnostics(moe, z, market_state):
    """Magnitude diagnostics of the market bias vs the original router."""
    with torch.no_grad():
        router_logits = moe.gate(z)
        market_bias = moe.market_routing_adapter(moe.normalize_market(market_state))
    bias_norm = float(market_bias.norm())
    router_norm = float(router_logits.norm())
    return {
        "market_bias_mean": float(market_bias.mean()),
        "market_bias_std": float(market_bias.std()),
        "market_bias_norm": bias_norm,
        "router_logits_norm": router_norm,
        "market_bias_to_router_norm_ratio": bias_norm / (router_norm + 1e-12),
    }


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(market_routing=True):
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )
    config["predictor"]["saved_model"] = str(STAGE1_CHECKPOINT)
    config["predictor"]["market_conditioned_routing"] = market_routing
    return config


def build_model(market_routing):
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    return GenerateReturn(copy.deepcopy(build_config(market_routing)), T_max=2)


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "combine_010_020_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1 provenance: experiment 010 formal marker ----
    if not STAGE1_MARKER.is_file():
        raise FileNotFoundError(f"Stage 1 marker missing: {STAGE1_MARKER}")
    marker = dict(
        line.split("=", 1)
        for line in STAGE1_MARKER.read_text().strip().splitlines()
    )
    if marker.get("commit") != EXPECTED_STAGE1_COMMIT:
        raise AssertionError(
            f"Stage 1 marker commit {marker.get('commit')} != queue pinned "
            f"{EXPECTED_STAGE1_COMMIT}"
        )
    if marker.get("reused") != "true" or marker.get("source") != "external":
        raise AssertionError("Stage 1 marker must record external reuse")
    if not STAGE1_CHECKPOINT.is_file() or STAGE1_CHECKPOINT.stat().st_size == 0:
        raise FileNotFoundError(f"Stage 1 checkpoint missing: {STAGE1_CHECKPOINT}")
    checkpoint_bytes = STAGE1_CHECKPOINT.stat().st_size
    checkpoint_md5 = file_md5(STAGE1_CHECKPOINT)
    if checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES:
        raise AssertionError("Stage 1 checkpoint size changed")
    if checkpoint_md5 != EXPECTED_CHECKPOINT_MD5:
        raise AssertionError("Stage 1 checkpoint hash changed")

    config = build_config(True)
    if config["predictor"]["shared_expert"] is not True:
        raise AssertionError("Default config must keep 010 shared_expert enabled")
    if config["predictor"]["market_conditioned_routing"] is not True:
        raise AssertionError("Default config must enable experiment 026 market routing")
    if config["vqvae"]["num_embed"] != 512 or config["vqvae"]["vq_embed_dim"] != 128:
        raise AssertionError("Stage 1 must remain single VQ512 with 128-dimensional codes")
    for key, expected in EXPECTED_SPLITS.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    base = build_model(False).eval()   # 010-equivalent: shared on, market off
    model = build_model(True).eval()   # 026: shared on, market routing on
    quantizers = [module for module in model.modules() if isinstance(module, VectorQuantiser)]
    if len(quantizers) != 1 or tuple(model.quantizer.embedding.weight.shape) != (512, 128):
        raise AssertionError("Stage 1 is not the required single VQ512 configuration")
    if base.loadings.fusion.moe.shared_expert is None or model.loadings.fusion.moe.shared_expert is None:
        raise AssertionError("010 shared expert must exist in both models")

    adapter_prefix = "loadings.fusion.moe.market_routing_adapter."
    adapted_base_state = {
        key: value for key, value in model.state_dict().items()
        if not key.startswith(adapter_prefix)
    }
    if base.state_dict().keys() != adapted_base_state.keys():
        raise AssertionError("Market adapter changed baseline state keys")
    existing_init_exact = all(
        torch.equal(value, adapted_base_state[key])
        for key, value in base.state_dict().items()
    )
    if not existing_init_exact:
        raise AssertionError("Market adapter perturbed baseline initialization")

    moe = model.loadings.fusion.moe
    base_moe = base.loadings.fusion.moe
    adapter = moe.market_routing_adapter
    adapter_zero_init = bool(
        adapter.bias is None and torch.count_nonzero(adapter.weight) == 0
        and adapter.in_features == 63 and adapter.out_features == 2
    )
    if not adapter_zero_init:
        raise AssertionError("Adapter must be zero-initialized Linear(63, 2, bias=False)")

    train_data = SyntheticCanonicalDataset(10, "2020-01-02", days=1)
    train_batch = train_data.values
    parts = unpack_batch(train_batch)
    feature, prior, market = parts.stock_feature, parts.prior_factor, parts.market_feature
    latest_market = model.current_market_state(market)
    if not torch.equal(latest_market, market[:, -1, :]):
        raise AssertionError("Current market extraction did not use the latest timestep")

    z = torch.randn(len(train_data), 128)
    x = torch.randn(len(train_data), 64)
    with torch.no_grad():
        clean_base = base_moe.clean_routing_logits(z)
        clean_adapted = moe.clean_routing_logits(z, latest_market)
        base_gates, base_load = base_moe.noisy_top_k_gating(z, False)
        adapted_gates, adapted_load = moe.noisy_top_k_gating(
            z, False, market_state=latest_market
        )
        base_moe_output, base_moe_aux = base_moe(x, z)
        adapted_moe_output, adapted_moe_aux = moe(
            x, z, market_state=latest_market
        )
        shared_base = base_moe.shared_expert(x)
        shared_adapted = moe.shared_expert(x)
        base_output = base(feature, prior)
        adapted_output = model(feature, prior, market)

    zero_clean_exact = torch.equal(clean_base, clean_adapted)
    zero_routing_exact = (
        torch.equal(base_gates, adapted_gates)
        and torch.equal(base_load, adapted_load)
    )
    zero_moe_exact = (
        torch.equal(base_moe_output, adapted_moe_output)
        and torch.equal(base_moe_aux, adapted_moe_aux)
    )
    zero_shared_exact = torch.equal(shared_base, shared_adapted)
    zero_forward_exact = all(
        torch.equal(left, right) for left, right in zip(base_output, adapted_output)
    )
    zero_init_switch_rate = expert_switch_rate(moe, z, latest_market, base_moe)
    if not all((zero_clean_exact, zero_routing_exact, zero_moe_exact,
                zero_shared_exact, zero_forward_exact)):
        raise AssertionError("Zero-init 026 is not bitwise equivalent to 010")
    if zero_init_switch_rate != 0.0:
        raise AssertionError("Zero-init expert-switch rate must be exactly 0")

    base_moe.train()
    moe.train()
    torch.manual_seed(2020)
    noisy_base_gates, noisy_base_load = base_moe.noisy_top_k_gating(z, True)
    torch.manual_seed(2020)
    noisy_adapted_gates, noisy_adapted_load = moe.noisy_top_k_gating(
        z, True, market_state=latest_market
    )
    noisy_behavior_exact = (
        torch.equal(noisy_base_gates, noisy_adapted_gates)
        and torch.equal(noisy_base_load, noisy_adapted_load)
        and torch.equal(base_moe.W_h, moe.W_h)
    )
    if not noisy_behavior_exact:
        raise AssertionError("Original noisy top-k/noise/W_h/load behavior changed")

    # ---- one real training step ----
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["train"]["learning_rate"], weight_decay=1e-5,
    )
    adapter_before = adapter.weight.detach().clone()
    optimizer.zero_grad()
    y_pred, _, _, z_q, aux_loss = model(feature, prior, market)
    loss = model.rank_loss(y_pred, parts.target(5)) + model.aux_weight * aux_loss
    loss.backward()
    adapter_grad_l1 = float(adapter.weight.grad.abs().sum())
    if adapter_grad_l1 <= 0 or not torch.isfinite(adapter.weight.grad).all():
        raise AssertionError("Market routing adapter did not receive a valid gradient")
    shared_grad_l1 = float(moe.shared_expert.net[-1].weight.grad.abs().sum())
    routed_grad_l1 = float(sum(
        parameter.grad.abs().sum()
        for name, parameter in moe.named_parameters()
        if name.startswith("experts.") and parameter.grad is not None
    ))
    if shared_grad_l1 <= 0 or routed_grad_l1 <= 0:
        raise AssertionError("Shared/Routed paths lost their training signal")
    frozen_stage1_has_grad = any(
        parameter.grad is not None
        for module in (model.encoder, model.quantizer, model.revin)
        for parameter in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 received gradients")
    codebook_before = model.quantizer.embedding.weight.detach().clone()
    optimizer.step()
    adapter_updated = not torch.equal(adapter_before, adapter.weight.detach())
    if not adapter_updated:
        raise AssertionError("Market routing adapter did not update")
    if not torch.equal(codebook_before, model.quantizer.embedding.weight.detach()):
        raise AssertionError("Market routing must not update the codebook")

    # ---- diagnostics on the (single-step-updated) adapter ----
    model.eval()
    with torch.no_grad():
        _, _, _, z_q_eval, _ = model(feature, prior, market)
    router_input = model.loadings.fusion.norm_z(z_q_eval)
    diagnostics = routing_diagnostics(moe, router_input, latest_market)
    diagnostics["adapter_weight_norm"] = float(adapter.weight.detach().norm())
    diagnostics["adapter_weight_grad_l1"] = adapter_grad_l1
    diagnostics["shared_expert_final_weight_grad_l1"] = shared_grad_l1
    diagnostics["routed_experts_grad_l1"] = routed_grad_l1
    diagnostics["zero_init_expert_switch_rate"] = zero_init_switch_rate
    diagnostics["post_step_expert_switch_rate"] = expert_switch_rate(
        moe, router_input, latest_market, base_moe
    )

    # ---- synthetic nonzero adapter: expert-switch capability ----
    routing_probe = copy.deepcopy(moe).eval()
    direction = torch.linspace(-2.0, 2.0, 63)
    with torch.no_grad():
        routing_probe.market_routing_adapter.weight[0].copy_(100 * direction)
        routing_probe.market_routing_adapter.weight[1].copy_(-100 * direction)
    fixed_z = torch.zeros(6, 128)
    market_a = direction.expand(6, -1)
    market_b = -direction.expand(6, -1)
    logits_a = routing_probe.clean_routing_logits(fixed_z, market_a)
    logits_b = routing_probe.clean_routing_logits(fixed_z, market_b)
    gates_a, _ = routing_probe.noisy_top_k_gating(fixed_z, False, market_state=market_a)
    gates_b, _ = routing_probe.noisy_top_k_gating(fixed_z, False, market_state=market_b)
    learned_market_changes_routing = bool(
        not torch.equal(logits_a, logits_b)
        and torch.equal(gates_a.argmax(1), torch.zeros(6, dtype=torch.long))
        and torch.equal(gates_b.argmax(1), torch.ones(6, dtype=torch.long))
    )
    if not learned_market_changes_routing:
        raise AssertionError("Nonzero adapter did not change expert allocation")
    diagnostics["synthetic_expert_switch_rate"] = float(
        (gates_a.argmax(1) != gates_b.argmax(1)).float().mean()
    )

    # ---- latest-timestep-only market path ----
    changed_history = market.clone()
    changed_history[:, :-1] = torch.randn_like(changed_history[:, :-1]) * 1000
    with torch.no_grad():
        history_reference = model(feature, prior, market)
        history_changed = model(feature, prior, changed_history)
    latest_only_exact = all(
        torch.equal(left, right)
        for left, right in zip(history_reference, history_changed)
    )
    if not latest_only_exact:
        raise AssertionError("Market history bypassed latest-timestep-only routing")

    # ---- per-date expert usage (read-only diagnostic) ----
    test_data = SyntheticCanonicalDataset(30, "2023-01-03")
    test_loader = DataLoader(test_data, batch_size=6, shuffle=False)
    per_date_usage = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.float()
            batch_parts = unpack_batch(batch)
            _, _, _, batch_z_q, _ = model(
                batch_parts.stock_feature,
                batch_parts.prior_factor,
                batch_parts.market_feature,
            )
            batch_router_input = model.loadings.fusion.norm_z(batch_z_q)
            batch_market_state = model.current_market_state(batch_parts.market_feature)
            assignments = expert_assignments(moe, batch_router_input, batch_market_state)
            date = str(test_data.index[batch_idx * 6][0].date())
            usage = per_date_usage.setdefault(date, [0] * moe.num_experts)
            for expert_idx in assignments.tolist():
                usage[expert_idx] += 1
    per_date_expert_usage = {
        date: {
            f"expert_{i}": count / max(1, sum(counts))
            for i, count in enumerate(counts)
        }
        for date, counts in per_date_usage.items()
    }

    # ---- Stage 2 checkpoint strict round-trip ----
    checkpoint_path = checkpoint_dir / "combine-010-020-smoke.ckpt"
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
        restored_output = restored(feature, prior, market)
    checkpoint_exact = all(
        torch.equal(left, right)
        for left, right in zip(history_reference, restored_output)
    )
    if not checkpoint_exact:
        raise AssertionError("Strict checkpoint round-trip changed output")

    # ---- standard inference + metrics + backtest normalizer ----
    prediction, _, metrics = run_inference(restored, test_loader, config, device="cpu")
    prediction_path = result_dir / "0_best.pkl"
    metric_path = result_dir / "0_metric.csv"
    prediction.to_pickle(prediction_path)
    pd.DataFrame([metrics], index=["values"]).transpose().to_csv(metric_path)
    normalized, signal = _normalize_prediction_frame(prediction_path)
    if len(normalized) != len(test_data) or len(signal) != len(test_data):
        raise AssertionError("Prediction output is incompatible with backtest normalizer")

    report = {
        "status": "PASS",
        "stage1": {
            "source": "010",
            "marker": str(STAGE1_MARKER.relative_to(ROOT)),
            "marker_commit": marker["commit"],
            "checkpoint": str(STAGE1_CHECKPOINT.relative_to(ROOT)),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_md5": checkpoint_md5,
            "strict_load": {
                "encoder": {"missing": 0, "unexpected": 0},
                "quantizer": {"missing": 0, "unexpected": 0},
                "revin": {"missing": 0, "unexpected": 0},
            },
            "single_vq512": True,
            "embedding_dimension": 128,
            "data_splits": EXPECTED_SPLITS,
            "frozen_parameters_have_grad": frozen_stage1_has_grad,
        },
        "combine_010_020": {
            "shared_expert": "always-on, zero-init, outside routing and top-k",
            "market_adapter": "Linear(63, 2, bias=False) on routed clean logits",
            "normalization": "parameter-free per-sample layer_norm over market63",
            "canonical_input_shape": list(market.shape),
            "latest_state_shape": list(latest_market.shape),
            "adapter_zero_initialized": adapter_zero_init,
            "clean_logits_bitwise_equal_010": zero_clean_exact,
            "routing_and_load_bitwise_equal_010": zero_routing_exact,
            "moe_output_and_aux_bitwise_equal_010": zero_moe_exact,
            "shared_expert_output_bitwise_equal_010": zero_shared_exact,
            "full_prediction_forward_bitwise_equal_010": zero_forward_exact,
            "existing_initialization_bitwise_equal_010": existing_init_exact,
            "noisy_topk_noise_wh_load_bitwise_equal_010": noisy_behavior_exact,
            "latest_timestep_only": latest_only_exact,
            "nonzero_adapter_changes_logits_and_allocation": learned_market_changes_routing,
            "adapter_updated": adapter_updated,
            "codebook_unchanged_by_market_routing": True,
        },
        "routing_diagnostics": diagnostics,
        "per_date_expert_usage": per_date_expert_usage,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "strict_round_trip": True,
            "output_bitwise_equal": checkpoint_exact,
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
