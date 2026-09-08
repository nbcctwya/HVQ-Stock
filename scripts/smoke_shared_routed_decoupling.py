"""Minimal Stage 2 smoke for experiment 017.

Reuses experiment 010's exact Stage 1 checkpoint, exercises the additional
Shared-Routed decoupling loss, and verifies checkpoint/inference compatibility.
"""

import copy
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
from dataset.schema import TOTAL_DIM
from module.layers.moe import FactorGatedMoE
from module.quantise import VectorQuantiser
from trainer.train_ypred import GenerateReturn
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "017" / "smoke"
SOURCE_MARKER = ROOT / "artifacts" / "010" / "run" / ".stage1.done"
SOURCE_COMMIT = "9b854f0436f8a7c3283fd375661dd6152cc965f1"
EXPECTED_SPLITS = {
    "train_period": ["2009-01-01", "2020-12-31"],
    "valid_period": ["2021-01-01", "2022-12-31"],
    "test_period": ["2023-01-01", "2025-12-31"],
}


class SyntheticCanonicalDataset(Dataset):
    def __init__(self, seed, start_date, days=2, stocks_per_day=8):
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


def read_marker(path):
    if not path.is_file():
        raise FileNotFoundError(f"Experiment 010 Stage 1 marker missing: {path}")
    marker = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            marker[key] = value
    return marker


def resolve_stage1_checkpoint():
    marker = read_marker(SOURCE_MARKER)
    if marker.get("commit") != SOURCE_COMMIT:
        raise AssertionError(
            "Experiment 010 Stage 1 marker commit does not match its frozen commit"
        )
    checkpoint = Path(marker.get("best", ""))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / "artifacts" / "010" / "run" / checkpoint
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Experiment 010 Stage 1 checkpoint missing: {checkpoint}")
    return marker, checkpoint.resolve()


def build_config(stage1_checkpoint):
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    config = OmegaConf.to_container(cfg, resolve=True)
    config["predictor"]["saved_model"] = str(stage1_checkpoint)
    return config


def training_loss(model, batch):
    y_pred, _, _, _, aux_loss = model(
        batch[:, :, :158], batch[:, -1, 158:171]
    )
    return model.rank_loss(y_pred, batch[:, -1, 238]) + model.aux_weight * aux_loss


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "shared_routed_decoupling_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    marker, stage1_checkpoint = resolve_stage1_checkpoint()
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    config = build_config(stage1_checkpoint)

    if config["predictor"]["shared_expert"] is not True:
        raise AssertionError("Shared Expert must remain enabled")
    if config["predictor"]["decoupling_lambda"] != 0.01:
        raise AssertionError("Default decoupling_lambda must be exactly 0.01")
    if config["vqvae"]["num_embed"] != 512:
        raise AssertionError("Stage 1 must retain single VQ512")
    for key, expected in EXPECTED_SPLITS.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    # GenerateReturn performs strict Encoder/Quantizer/RevIN Stage 1 loading.
    model = GenerateReturn(config, T_max=2)
    if not isinstance(model.quantizer, VectorQuantiser):
        raise AssertionError("Stage 1 quantizer is not the single VectorQuantiser")
    if tuple(model.quantizer.embedding.weight.shape) != (512, 128):
        raise AssertionError("Stage 1 codebook is not VQ512 with dimension 128")

    moe = model.loadings.fusion.moe
    if moe.shared_expert is None or moe.decoupling_lambda != 0.01:
        raise AssertionError("Shared-Routed decoupling is not enabled")

    # Compare against 010 with a deliberately nonzero Shared Expert so the
    # equality check proves that only the returned auxiliary loss changed.
    experiment_probe = copy.deepcopy(moe).eval()
    with torch.no_grad():
        experiment_probe.shared_expert.net[-1].weight.fill_(0.02)
        experiment_probe.shared_expert.net[-1].bias.fill_(0.01)
    base_probe = copy.deepcopy(experiment_probe)
    base_probe.decoupling_lambda = 0.0
    probe_x = torch.randn(9, 64)
    probe_z = torch.randn(9, 128)
    experiment_out, experiment_loss = experiment_probe(probe_x, probe_z)
    base_out, route_loss = base_probe(probe_x, probe_z)

    routed_probe = copy.deepcopy(base_probe)
    routed_probe.shared_expert = None
    routed_probe.use_shared_expert = False
    routed_out, routed_route_loss = routed_probe(probe_x, probe_z)
    shared_out = base_probe.shared_expert(probe_x)
    decoupling_penalty = FactorGatedMoE.shared_routed_decoupling_loss(
        shared_out, routed_out
    )
    prediction_equivalent = bool(torch.equal(experiment_out, base_out))
    route_loss_unchanged = bool(torch.equal(route_loss, routed_route_loss))
    loss_formula_correct = bool(
        torch.allclose(
            experiment_loss,
            route_loss + 0.01 * decoupling_penalty,
            atol=1e-7,
            rtol=0,
        )
    )
    if not prediction_equivalent:
        raise AssertionError("017 prediction path differs from 010 fixed addition")
    if not route_loss_unchanged:
        raise AssertionError("Routed load-balancing loss changed")
    if not loss_formula_correct:
        raise AssertionError("MoE loss is not route_loss + 0.01 * decoupling_loss")

    # Directly verify that the regularizer supplies finite gradients to both
    # representations, independently of the main prediction objective.
    shared_leaf = torch.randn(9, 64, requires_grad=True)
    routed_leaf = torch.randn(9, 64, requires_grad=True)
    gradient_probe_loss = FactorGatedMoE.shared_routed_decoupling_loss(
        shared_leaf, routed_leaf
    )
    gradient_probe_loss.backward()
    shared_dec_grad_l1 = float(shared_leaf.grad.abs().sum())
    routed_dec_grad_l1 = float(routed_leaf.grad.abs().sum())
    if shared_dec_grad_l1 <= 0 or routed_dec_grad_l1 <= 0:
        raise AssertionError("Decoupling loss did not produce gradients on both paths")

    train_data = SyntheticCanonicalDataset(10, "2020-01-02", days=1)
    train_batch = train_data.values
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["train"]["learning_rate"],
        weight_decay=1e-5,
    )
    shared_final = moe.shared_expert.net[-1]
    before_weight = shared_final.weight.detach().clone()
    optimizer.zero_grad()
    training_loss(model, train_batch).backward()
    shared_weight_grad_l1 = float(shared_final.weight.grad.abs().sum())
    if shared_weight_grad_l1 <= 0:
        raise AssertionError("Shared Expert did not receive a training gradient")
    optimizer.step()
    shared_updated = not torch.equal(before_weight, shared_final.weight.detach())
    if not shared_updated:
        raise AssertionError("Shared Expert did not update")

    frozen_stage1_has_grad = any(
        parameter.grad is not None
        for module in (model.encoder, model.quantizer, model.revin)
        for parameter in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 parameters received gradients")

    checkpoint_path = checkpoint_dir / "shared-routed-decoupling-smoke.ckpt"
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
        checkpoint_path, config=config, T_max=2, strict=True
    )
    restored.freeze_vqvae()
    restored.eval()
    model.eval()
    with torch.no_grad():
        reference = model(train_batch[:, :, :158], train_batch[:, -1, 158:171])[0]
        reloaded = restored(train_batch[:, :, :158], train_batch[:, -1, 158:171])[0]
    checkpoint_exact = torch.equal(reference, reloaded)
    if not checkpoint_exact:
        raise AssertionError("Strict checkpoint round-trip changed prediction")

    inference_results = {}
    for split, split_seed, start_date in (
        ("valid", 20, "2021-01-04"),
        ("test", 30, "2023-01-03"),
    ):
        dataset = SyntheticCanonicalDataset(split_seed, start_date)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        prediction, _, metrics = run_inference(restored, loader, config, device="cpu")
        prediction_path = result_dir / f"0_{split}.pkl"
        metric_path = result_dir / f"0_{split}_metric.csv"
        prediction.to_pickle(prediction_path)
        pd.DataFrame([metrics], index=["values"]).transpose().to_csv(metric_path)
        normalized, signal = _normalize_prediction_frame(prediction_path)
        if len(normalized) != len(dataset) or len(signal) != len(dataset):
            raise AssertionError(f"{split} output is incompatible with backtest")
        inference_results[split] = {
            "rows": len(prediction),
            "prediction": str(prediction_path.relative_to(ROOT)),
            "metric": str(metric_path.relative_to(ROOT)),
            "metrics": {key: float(value) for key, value in metrics.items()},
        }

    test_prediction = pd.read_pickle(result_dir / "0_test.pkl")
    standard_prediction_path = result_dir / "0_best.pkl"
    test_prediction.to_pickle(standard_prediction_path)
    test_metrics = pd.read_csv(result_dir / "0_test_metric.csv", index_col=0)
    standard_metric_path = result_dir / "0_metric.csv"
    test_metrics.to_csv(standard_metric_path)
    _normalize_prediction_frame(standard_prediction_path)

    report = {
        "status": "PASS",
        "stage1": {
            "source": "010",
            "source_commit": marker["commit"],
            "checkpoint": str(stage1_checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": stage1_checkpoint.stat().st_size,
            "strict_load": {
                "encoder": {"missing": 0, "unexpected": 0},
                "quantizer": {"missing": 0, "unexpected": 0},
                "revin": {"missing": 0, "unexpected": 0},
            },
            "single_vq512": True,
            "data_splits": EXPECTED_SPLITS,
            "frozen_parameters_have_grad": frozen_stage1_has_grad,
        },
        "shared_routed_decoupling": {
            "lambda": 0.01,
            "prediction_bitwise_equal_to_010": prediction_equivalent,
            "nonzero_shared_output_used_for_equivalence_check": True,
            "route_loss_bitwise_unchanged": route_loss_unchanged,
            "loss_formula_correct": loss_formula_correct,
            "probe_penalty": float(decoupling_penalty.detach()),
            "probe_penalty_finite": bool(torch.isfinite(decoupling_penalty)),
            "shared_representation_grad_l1": shared_dec_grad_l1,
            "routed_representation_grad_l1": routed_dec_grad_l1,
            "shared_training_weight_grad_l1": shared_weight_grad_l1,
            "shared_training_weight_updated": shared_updated,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "strict_round_trip": True,
            "output_bitwise_equal": checkpoint_exact,
        },
        "inference": inference_results,
        "standard_outputs": {
            "prediction": str(standard_prediction_path.relative_to(ROOT)),
            "metric": str(standard_metric_path.relative_to(ROOT)),
            "backtest_normalizer": "PASS",
        },
    }
    report_path = ARTIFACT_ROOT / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
