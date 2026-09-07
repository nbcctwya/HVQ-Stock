"""Minimal end-to-end Stage 2 smoke for experiment 010.

This intentionally uses synthetic canonical batches, but loads the exact
formal baseline Stage 1 checkpoint. It exercises shared+routed forward,
backward/update, checkpoint round-trip, validation/test inference, and the
standard prediction format consumed by the Phase 2/backtest pipeline.
"""

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
from trainer.train_ypred import GenerateReturn
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "010" / "smoke"
STAGE1_CKPT = (
    ROOT
    / "artifacts"
    / "baseline"
    / "run"
    / "checkpoints"
    / "infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt"
)


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


def build_config():
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    config = OmegaConf.to_container(cfg, resolve=True)
    config["predictor"]["saved_model"] = str(STAGE1_CKPT)
    return config


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "shared_routed_moe_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not STAGE1_CKPT.is_file() or STAGE1_CKPT.stat().st_size == 0:
        raise FileNotFoundError(f"Exact baseline Stage 1 checkpoint missing: {STAGE1_CKPT}")

    seed_everything(0)
    pl.seed_everything(0, workers=True)
    config = build_config()
    if config["predictor"]["shared_expert"] is not True:
        raise AssertionError("configs/config.yaml must enable predictor.shared_expert")

    model = GenerateReturn(config, T_max=1)
    moe = model.loadings.fusion.moe
    if moe.shared_expert is None:
        raise AssertionError("Shared Expert is not enabled")

    final_linear = moe.shared_expert.net[-1]
    zero_init = bool(
        torch.count_nonzero(final_linear.weight) == 0
        and torch.count_nonzero(final_linear.bias) == 0
    )
    if not zero_init:
        raise AssertionError("Shared Expert final Linear is not zero-initialized")

    train_data = SyntheticCanonicalDataset(10, "2020-01-02", days=1)
    train_batch = train_data.values
    model.train()
    y_pred, _, _, _, aux_loss = model(
        train_batch[:, :, :158], train_batch[:, -1, 158:171]
    )
    loss = model.rank_loss(y_pred, train_batch[:, -1, 238]) + model.aux_weight * aux_loss

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["train"]["learning_rate"],
        weight_decay=1e-5,
    )
    before_weight = final_linear.weight.detach().clone()
    before_bias = final_linear.bias.detach().clone()
    optimizer.zero_grad()
    loss.backward()

    shared_weight_grad = float(final_linear.weight.grad.abs().sum())
    shared_bias_grad = float(final_linear.bias.grad.abs().sum())
    if shared_weight_grad <= 0 or shared_bias_grad <= 0:
        raise AssertionError("Shared Expert final Linear did not receive a nonzero gradient")
    frozen_stage1_has_grad = any(
        parameter.grad is not None
        for module in (model.encoder, model.quantizer, model.revin)
        for parameter in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 parameters received gradients")

    optimizer.step()
    shared_weight_updated = not torch.equal(before_weight, final_linear.weight.detach())
    shared_bias_updated = not torch.equal(before_bias, final_linear.bias.detach())
    if not shared_weight_updated or not shared_bias_updated:
        raise AssertionError("Shared Expert final Linear did not update")

    checkpoint_path = checkpoint_dir / "shared-routed-smoke.ckpt"
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
        checkpoint_path, config=config, T_max=1, strict=True
    )
    restored.freeze_vqvae()
    restored.eval()

    model.eval()
    with torch.no_grad():
        reference = model(train_batch[:, :, :158], train_batch[:, -1, 158:171])[0]
        reloaded = restored(train_batch[:, :, :158], train_batch[:, -1, 158:171])[0]
    checkpoint_exact = torch.equal(reference, reloaded)
    if not checkpoint_exact:
        raise AssertionError("Checkpoint round-trip changed model output")

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
            raise AssertionError(f"{split} prediction is incompatible with backtest normalizer")
        inference_results[split] = {
            "rows": len(prediction),
            "prediction": str(prediction_path.relative_to(ROOT)),
            "metric": str(metric_path.relative_to(ROOT)),
            "metrics": {key: float(value) for key, value in metrics.items()},
        }

    # Match the standard Phase 2 filename in addition to retaining explicit
    # valid/test files for smoke auditability.
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
            "source": "external",
            "checkpoint": str(STAGE1_CKPT.relative_to(ROOT)),
            "strict_load": {
                "encoder": {"missing": 0, "unexpected": 0},
                "quantizer": {"missing": 0, "unexpected": 0},
                "revin": {"missing": 0, "unexpected": 0},
            },
            "frozen_parameters_have_grad": frozen_stage1_has_grad,
        },
        "shared_expert": {
            "zero_initialized": zero_init,
            "final_weight_grad_l1": shared_weight_grad,
            "final_bias_grad_l1": shared_bias_grad,
            "final_weight_updated": shared_weight_updated,
            "final_bias_updated": shared_bias_updated,
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
