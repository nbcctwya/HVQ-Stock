#!/usr/bin/env python3
"""End-to-end AlphaMaster smoke using tiny canonical PKLs."""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_qlib import _normalize_prediction_frame
from dataset.market import CanonicalSampler, market_feature_config
from dataset.schema import GROUP_DIMS, GROUP_SLICES, MARKET_INDICES, TOTAL_DIM, unpack_batch
from experiments.runner import find_best_stage1_ckpt, find_stage2_outputs
from trainer.train_alphamaster import AlphaMasterModule


def build_sampler(universe):
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2020-01-01", periods=30, name="datetime")
    instruments = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    fields, _ = market_feature_config(MARKET_INDICES[universe])
    columns = pd.MultiIndex.from_tuples([
        (group, fields[i] if group == "market" else f"{group}_{i}")
        for group, width in GROUP_DIMS.items()
        for i in range(width)
    ])
    frame = pd.DataFrame(
        rng.normal(size=(len(index), TOTAL_DIM)).astype(np.float32),
        index=index,
        columns=columns,
    )
    market = pd.DataFrame(
        rng.normal(size=(len(dates), 63)).astype(np.float32),
        index=dates,
        columns=pd.MultiIndex.from_product([["market"], fields]),
    )
    frame.loc[:, "market"] = market.loc[
        index.get_level_values("datetime")
    ].to_numpy()
    return CanonicalSampler(
        frame, market, universe, dates[20], dates[-1]
    )


def write_data(data_root):
    for universe, region in (("csi300", "CN"), ("sp500", "US")):
        sampler = build_sampler(universe)
        region_dir = data_root / region
        region_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid", "test"):
            path = region_dir / f"{universe}_20_h10_{split}.pkl"
            with path.open("wb") as stream:
                pickle.dump(sampler, stream)


def run_logged(command, log_path, env):
    with log_path.open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}); see {log_path}"
        )


def verify_inputs(data_root):
    with (ROOT / "configs" / "config.yaml").open() as stream:
        base_config = yaml.safe_load(stream)
    results = {}
    for universe, region, beta in (
        ("csi300", "CN", 10), ("sp500", "US", 5)
    ):
        with (data_root / region / f"{universe}_20_h10_test.pkl").open("rb") as stream:
            dataset = pickle.load(stream)
        dates = dataset.get_index().get_level_values("datetime")
        positions = np.flatnonzero(dates == dates[0])
        batch = torch.as_tensor(dataset[positions]).float()
        parts = unpack_batch(batch)
        config = dict(base_config)
        config["data"] = dict(base_config["data"], universe=universe)
        model = AlphaMasterModule(config).eval()
        if model.master.feature_gate.t != beta:
            raise AssertionError(f"unexpected {universe} beta")
        prediction, alpha, loadings, prior_contribution, hidden = model(
            parts.stock_feature,
            parts.market_feature,
            parts.prior_factor,
            return_components=True,
        )
        if parts.prior_factor.shape != (4, 13):
            raise AssertionError("canonical prior_factor is not [N,13]")
        if alpha.shape != (4,) or loadings.shape != (4, 13):
            raise AssertionError("prior-factor head output shape mismatch")
        if prediction.shape != (4,):
            raise AssertionError("prediction shape mismatch")
        torch.testing.assert_close(
            prior_contribution,
            torch.sum(loadings * parts.prior_factor, dim=-1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            prediction,
            alpha + torch.sum(loadings * parts.prior_factor, dim=-1),
            rtol=0,
            atol=0,
        )

        changed_prior = batch.clone()
        changed_prior[..., GROUP_SLICES["prior"]] += 123456
        changed_parts = unpack_batch(changed_prior)
        changed_output = model(
            changed_parts.stock_feature,
            changed_parts.market_feature,
            changed_parts.prior_factor,
            return_components=True,
        )
        prediction_changed_prior, new_alpha, new_loadings, _, new_hidden = changed_output
        if torch.equal(prediction, prediction_changed_prior):
            raise AssertionError("prior13 did not affect final prediction")
        for expected, actual in (
            (hidden, new_hidden), (alpha, new_alpha), (loadings, new_loadings)
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

        repeated = model(
            parts.stock_feature,
            parts.market_feature,
            parts.prior_factor,
            return_components=True,
        )
        for expected, actual in zip(
            (prediction, alpha, loadings, prior_contribution, hidden), repeated
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        if torch.equal(loadings[0], loadings[1]):
            raise AssertionError("prior loadings are not dynamically sample-dependent")
        torch.testing.assert_close(
            loadings,
            model.master.prior_loading_head(hidden),
            rtol=0,
            atol=0,
        )

        changed_market = parts.market_feature.clone()
        changed_market[:, -1, 0] += 100
        if torch.equal(
            prediction,
            model(parts.stock_feature, changed_market, parts.prior_factor),
        ):
            raise AssertionError("market63 did not affect AlphaMaster prediction")
        if len(set(dates[positions])) != 1:
            raise AssertionError("smoke batch spans multiple trading days")
        results[universe] = {
            "input_shape": list(batch.shape),
            "stock_shape": list(parts.stock_feature.shape),
            "prior_shape": list(parts.prior_factor.shape),
            "market_shape": list(parts.market_feature.shape),
            "prediction_shape": list(prediction.shape),
            "alpha_shape": list(alpha.shape),
            "prior_loading_shape": list(loadings.shape),
            "beta": beta,
            "factor_formula_exact": True,
            "prior_sensitive": True,
            "prior_backbone_isolated": True,
            "deterministic_forward": True,
            "dynamic_loadings": True,
            "market_sensitive": True,
            "single_day_cross_section": True,
            "diagnostics": {
                "alpha_mean": alpha.mean().item(),
                "prior_contribution_mean": prior_contribution.mean().item(),
                "prior_loading_mean": loadings.mean().item(),
                "prior_loading_std": loadings.std().item(),
                "prior_loading_abs_mean": loadings.abs().mean().item(),
            },
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="artifacts/008/smoke",
        help="Smoke artifact directory relative to the repository root.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = output_dir / "data"
    if not data_root.exists():
        write_data(data_root)
    forward_results = verify_inputs(data_root)

    artifact_override = output_dir.relative_to(ROOT)
    env = dict(os.environ, WANDB_MODE="offline", MPLCONFIGDIR=str(output_dir / "mpl"))
    common = [
        f"data.data_path={data_root}",
        f"artifact_root={artifact_override}",
        "train.num_workers=0",
    ]
    best = find_best_stage1_ckpt(output_dir)
    if best is None:
        run_logged([
            sys.executable, "stage1.py", *common,
            "train.num_epochs=1",
            "train.accelerator=cpu",
            "train.gpu_counts=1",
            "train.limit_train_batches=2",
            "train.limit_val_batches=2",
            "train.run_name=alphamaster_smoke",
        ], output_dir / "stage1.log", env)
        best = find_best_stage1_ckpt(output_dir)
    if best is None:
        raise AssertionError("runner did not discover the Stage 1 checkpoint")
    checkpoint = best[1]
    run_logged([
        sys.executable, "stage2.py", *common,
        "train.seed=0",
        f'predictor.saved_model="{checkpoint.name}"',
    ], output_dir / "stage2.log", env)

    result_dir = find_stage2_outputs(output_dir)
    if result_dir is None:
        raise AssertionError("runner did not discover Stage 2 outputs")
    prediction_path = result_dir / "0_best.pkl"
    metric_path = result_dir / "0_metric.csv"
    normalized, signal = _normalize_prediction_frame(prediction_path)
    if signal.empty or not signal.index.names == ["datetime", "instrument"]:
        raise AssertionError("prediction is incompatible with backtest_qlib.py")

    with (ROOT / "configs" / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    AlphaMasterModule.load_strict_checkpoint(checkpoint, config)

    report = {
        "status": "PASS",
        "forward": forward_results,
        "stage1": {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "runner_discovery": True,
            "minimal_training": True,
        },
        "stage2": {
            "strict_checkpoint_load": True,
            "prediction": str(prediction_path.relative_to(ROOT)),
            "metric": str(metric_path.relative_to(ROOT)),
        },
        "backtest_compatibility": {
            "normalized_rows": len(normalized),
            "accepted_by_normalizer": True,
        },
    }
    with (output_dir / "smoke_report.json").open("w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
