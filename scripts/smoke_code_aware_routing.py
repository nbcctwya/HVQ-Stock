"""Minimal synthetic smoke for experiment 022.

Uses the exact corrected PRISM-VQ Stage 1 checkpoint and canonical 244-field
inputs.  No formal training or backtest is run.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_qlib import _normalize_prediction_frame
from dataset.schema import TOTAL_DIM, unpack_batch
from module.layers.moe import FactorGatedMoE
from module.quantise import VectorQuantiser
from trainer.train_ypred import GenerateReturn
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "022" / "smoke"
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


class SyntheticCanonicalDataset(Dataset):
    def __init__(self, seed, start_date, days=2, stocks_per_day=6):
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
        return self.values[index]

    def get_index(self):
        return self.index


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(code_aware_routing=True):
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )
    config["predictor"]["saved_model"] = str(STAGE1_CHECKPOINT)
    config["predictor"]["code_aware_routing"] = code_aware_routing
    return config


def build_model(code_aware_routing):
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    return GenerateReturn(copy.deepcopy(build_config(code_aware_routing)), T_max=2)


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "code_aware_routing_smoke"
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
    if config["predictor"]["code_aware_routing"] is not True:
        raise AssertionError("Default config must enable experiment 022")
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

    bias_prefix = "loadings.fusion.moe.code_bias"
    adapted_base_state = {
        key: value for key, value in model.state_dict().items()
        if not key.startswith(bias_prefix)
    }
    if base.state_dict().keys() != adapted_base_state.keys():
        raise AssertionError("Code-bias table changed baseline state keys")
    existing_init_exact = all(
        torch.equal(value, adapted_base_state[key])
        for key, value in base.state_dict().items()
    )
    if not existing_init_exact:
        raise AssertionError("Code-bias table perturbed baseline initialization")

    moe = model.loadings.fusion.moe
    base_moe = base.loadings.fusion.moe
    table = moe.code_bias
    table_zero_init = bool(
        tuple(table.shape) == (config["vqvae"]["num_embed"], config["predictor"]["n_expert"])
        and torch.count_nonzero(table.detach()) == 0
    )
    if not table_zero_init:
        raise AssertionError("Code bias must be a zero-initialized [num_embed, n_expert] table")

    train_data = SyntheticCanonicalDataset(10, "2020-01-02", days=1)
    train_batch = train_data.values
    unpacked = unpack_batch(train_batch.float())
    feature, prior = unpacked.stock_feature, unpacked.prior_factor

    # vq_idx pass-through: the ids the MoE router receives must be exactly the
    # ids the frozen Stage 1 quantizer produced.
    captured = {}
    original_clean = FactorGatedMoE.clean_routing_logits

    def spy(self, x, vq_idx=None):
        captured["vq_idx"] = vq_idx
        return original_clean(self, x, vq_idx=vq_idx)

    with mock.patch.object(FactorGatedMoE, "clean_routing_logits", spy):
        with torch.no_grad():
            model(feature, prior)
    with torch.no_grad():
        h_batch = model.encoder(model.revin(feature, mode="norm"))
        _, _, (_, _, expected_vq_idx) = model.quantizer(h_batch)
    vq_idx_pass_through = torch.equal(captured["vq_idx"], expected_vq_idx)
    if not vq_idx_pass_through:
        raise AssertionError("vq_idx did not pass from quantizer to MoE router")

    z = torch.randn(len(train_data), 128)
    x = torch.randn(len(train_data), 64)
    vq_idx = captured["vq_idx"]
    with torch.no_grad():
        clean_base = base_moe.clean_routing_logits(z)
        clean_adapted = moe.clean_routing_logits(z, vq_idx)
        base_gates, base_load = base_moe.noisy_top_k_gating(z, False)
        adapted_gates, adapted_load = moe.noisy_top_k_gating(z, False, vq_idx=vq_idx)
        base_moe_output, base_moe_aux = base_moe(x, z)
        adapted_moe_output, adapted_moe_aux = moe(x, z, vq_idx=vq_idx)
        base_output = base(feature, prior)
        adapted_output = model(feature, prior)

    zero_clean_exact = torch.equal(clean_base, clean_adapted)
    zero_routing_exact = (
        torch.equal(base_gates, adapted_gates)
        and torch.equal(base_load, adapted_load)
    )
    zero_moe_exact = (
        torch.equal(base_moe_output, adapted_moe_output)
        and torch.equal(base_moe_aux, adapted_moe_aux)
    )
    zero_forward_exact = all(
        torch.equal(left, right) for left, right in zip(base_output, adapted_output)
    )
    if not all((zero_clean_exact, zero_routing_exact, zero_moe_exact, zero_forward_exact)):
        raise AssertionError("Zero-init routing is not bitwise equivalent to baseline")

    base_moe.train()
    moe.train()
    torch.manual_seed(2020)
    noisy_base_gates, noisy_base_load = base_moe.noisy_top_k_gating(z, True)
    torch.manual_seed(2020)
    noisy_adapted_gates, noisy_adapted_load = moe.noisy_top_k_gating(
        z, True, vq_idx=vq_idx
    )
    noisy_behavior_exact = (
        torch.equal(noisy_base_gates, noisy_adapted_gates)
        and torch.equal(noisy_base_load, noisy_adapted_load)
        and torch.equal(base_moe.W_h, moe.W_h)
    )
    if not noisy_behavior_exact:
        raise AssertionError("Original noisy top-k/noise/W_h/load behavior changed")

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["train"]["learning_rate"], weight_decay=1e-5,
    )
    table_before = table.detach().clone()
    optimizer.zero_grad()
    y_pred, _, _, _, aux_loss = model(feature, prior)
    loss = model.rank_loss(y_pred, unpacked.target(5)) + model.aux_weight * aux_loss
    loss.backward()
    table_grad_l1 = float(table.grad.abs().sum())
    if table_grad_l1 <= 0 or not torch.isfinite(table.grad).all():
        raise AssertionError("Code-bias table did not receive a valid gradient")
    frozen_stage1_has_grad = any(
        parameter.grad is not None
        for module in (model.encoder, model.quantizer, model.revin)
        for parameter in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 received gradients")
    optimizer.step()
    table_updated = not torch.equal(table_before, table.detach())
    if not table_updated:
        raise AssertionError("Code-bias table did not update")

    routing_probe = copy.deepcopy(moe).eval()
    for parameter in routing_probe.gate.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        routing_probe.code_bias[0].copy_(torch.tensor([2.0, -2.0]))
        routing_probe.code_bias[1].copy_(torch.tensor([-2.0, 2.0]))
    fixed_z = torch.zeros(6, 128)
    idx_a = torch.zeros(6, dtype=torch.long)
    idx_b = torch.ones(6, dtype=torch.long)
    logits_a = routing_probe.clean_routing_logits(fixed_z, idx_a)
    logits_b = routing_probe.clean_routing_logits(fixed_z, idx_b)
    gates_a, _ = routing_probe.noisy_top_k_gating(fixed_z, False, vq_idx=idx_a)
    gates_b, _ = routing_probe.noisy_top_k_gating(fixed_z, False, vq_idx=idx_b)
    nonzero_bias_changes_routing = bool(
        not torch.equal(logits_a, logits_b)
        and torch.equal(gates_a.argmax(1), torch.zeros(6, dtype=torch.long))
        and torch.equal(gates_b.argmax(1), torch.ones(6, dtype=torch.long))
    )
    if not nonzero_bias_changes_routing:
        raise AssertionError("Nonzero code bias did not change expert allocation")

    invalid_id_raises = True
    for bad in (torch.tensor([-1] * 6), torch.tensor([512] * 6),
                torch.zeros(6, 1, dtype=torch.long), torch.zeros(5, dtype=torch.long)):
        try:
            moe.clean_routing_logits(z[:6], bad)
            invalid_id_raises = False
        except ValueError:
            pass
    try:
        moe.clean_routing_logits(z[:6], torch.zeros(6))
        invalid_id_raises = False
    except TypeError:
        pass
    if not invalid_id_raises:
        raise AssertionError("Invalid code ids must raise")

    # Isolation: the discrete code id must not reach any non-routing module.
    model.eval()
    seen = {}

    def capture(name):
        return lambda _module, inputs: seen.setdefault(name, tuple(inputs))

    handles = [
        model.loadings.temporal_transformer.register_forward_pre_hook(capture("temporal")),
        model.latent_value_head.register_forward_pre_hook(capture("latent_head")),
        model.return_predictor.register_forward_pre_hook(capture("return_predictor")),
    ]
    with torch.no_grad():
        isolated_output = model(feature, prior)
    for handle in handles:
        handle.remove()
    int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    code_id_isolated = not any(
        isinstance(tensor, torch.Tensor) and tensor.dtype in int_dtypes
        for inputs in seen.values() for tensor in inputs
    )
    if not code_id_isolated:
        raise AssertionError("Code id bypassed the router into other prediction paths")

    checkpoint_path = checkpoint_dir / "code-aware-routing-smoke.ckpt"
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
        restored_output = restored(feature, prior)
    checkpoint_exact = all(
        torch.equal(left, right)
        for left, right in zip(isolated_output, restored_output)
    )
    if not checkpoint_exact:
        raise AssertionError("Strict checkpoint round-trip changed output")

    test_data = SyntheticCanonicalDataset(30, "2023-01-03")
    test_loader = DataLoader(test_data, batch_size=6, shuffle=False)
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
            "source": "external",
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
        "code_aware_routing": {
            "table": f"nn.Parameter zeros([num_embed={config['vqvae']['num_embed']}, "
                     f"n_expert={config['predictor']['n_expert']}])",
            "zero_initialized": table_zero_init,
            "vq_idx_pass_through_from_quantizer": vq_idx_pass_through,
            "clean_logits_bitwise_equal_base": zero_clean_exact,
            "routing_and_load_bitwise_equal_base": zero_routing_exact,
            "moe_output_and_aux_bitwise_equal_base": zero_moe_exact,
            "full_prediction_forward_bitwise_equal_base": zero_forward_exact,
            "existing_initialization_bitwise_equal_base": existing_init_exact,
            "noisy_topk_noise_wh_load_bitwise_equal_base": noisy_behavior_exact,
            "code_id_isolated_to_router": code_id_isolated,
            "invalid_code_id_raises": invalid_id_raises,
            "nonzero_bias_changes_logits_and_allocation": nonzero_bias_changes_routing,
            "weight_grad_l1": table_grad_l1,
            "updated": table_updated,
        },
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
