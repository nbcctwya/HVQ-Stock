#!/usr/bin/env python3
"""End-to-end day-level EMA AlphaMaster smoke using tiny canonical PKLs."""

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

        def capture_forward(candidate_model, stock, market):
            values = {}
            handles = [
                candidate_model.master.feature_gate.register_forward_pre_hook(
                    lambda module, args: values.__setitem__(
                        "gate_input", args[0].detach().clone()
                    )
                ),
                candidate_model.master.market_encoder.register_forward_pre_hook(
                    lambda module, args: values.__setitem__(
                        "history", args[0].detach().clone()
                    )
                ),
                candidate_model.master.market_encoder.register_forward_hook(
                    lambda module, args, output: values.__setitem__(
                        "market_state", output.detach().clone()
                    )
                ),
                candidate_model.master.market_quantizer.register_forward_hook(
                    lambda module, args, output: values.update({
                        "vq_input": args[0].detach().clone(),
                        "quantized_state": output.quantized.detach().clone(),
                        "code_indices": output.indices.detach().clone(),
                        "vq_loss": output.loss.detach().clone(),
                        "codebook_loss": output.codebook_loss.detach().clone(),
                        "commitment_loss": output.commitment_loss.detach().clone(),
                    })
                ),
                candidate_model.master.temporalatten.register_forward_hook(
                    lambda module, args, output: values.__setitem__(
                        "hidden", output.detach().clone()
                    )
                ),
                candidate_model.master.market_adapter.register_forward_pre_hook(
                    lambda module, args: values.__setitem__(
                        "adapter_input", args[0].detach().clone()
                    )
                ),
                candidate_model.master.market_adapter.register_forward_hook(
                    lambda module, args, output: values.__setitem__(
                        "delta_weight", output.detach().clone()
                    )
                ),
            ]
            values["prediction"] = candidate_model(stock, market).detach()
            for handle in handles:
                handle.remove()
            return values

        zero_values = capture_forward(
            model, parts.stock_feature, parts.market_feature
        )
        prediction = zero_values["prediction"]
        if zero_values["history"].shape != (len(positions), 19, 63):
            raise AssertionError("historical market input is not [N,19,63]")
        if zero_values["market_state"].shape != (len(positions), 63):
            raise AssertionError("market state is not [N,63]")
        if zero_values["vq_input"].shape != (len(positions), 63):
            raise AssertionError("VQ input is not [N,63]")
        if zero_values["quantized_state"].shape != (len(positions), 63):
            raise AssertionError("quantized market state is not [N,63]")
        if zero_values["code_indices"].shape != (len(positions),):
            raise AssertionError("VQ indices are not [N]")
        if zero_values["hidden"].shape != (len(positions), 256):
            raise AssertionError("TemporalAttention hidden is not [N,256]")
        if zero_values["delta_weight"].shape != (len(positions), 256):
            raise AssertionError("Market Adapter output is not [N,256]")
        torch.testing.assert_close(
            zero_values["gate_input"], parts.market_feature[:, -1, :],
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            zero_values["history"], parts.market_feature[:, :-1, :],
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            zero_values["vq_input"], zero_values["market_state"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            zero_values["adapter_input"], zero_values["quantized_state"],
            rtol=0, atol=0,
        )
        if not torch.isfinite(zero_values["vq_loss"]):
            raise AssertionError("VQ loss is not finite")
        if torch.count_nonzero(model.master.market_adapter.weight):
            raise AssertionError("Market Adapter is not zero-initialized")
        if torch.count_nonzero(zero_values["delta_weight"]):
            raise AssertionError("zero-initialized adapter emitted nonzero weights")
        base_prediction = model.master.decoder(zero_values["hidden"]).squeeze(-1)
        torch.testing.assert_close(prediction, base_prediction, rtol=0, atol=0)

        changed_prior = batch.clone()
        changed_prior[..., GROUP_SLICES["prior"]] += 123456
        changed_parts = unpack_batch(changed_prior)
        prediction_changed_prior = model(
            changed_parts.stock_feature, changed_parts.market_feature
        )
        if not torch.equal(prediction, prediction_changed_prior):
            raise AssertionError("prior13 changed AlphaMaster prediction")

        if len(set(dates[positions])) != 1:
            raise AssertionError("smoke batch spans multiple trading days")
        torch.testing.assert_close(
            zero_values["market_state"],
            zero_values["market_state"][:1].expand_as(zero_values["market_state"]),
            rtol=1e-5,
            atol=2e-7,
        )
        torch.testing.assert_close(
            zero_values["quantized_state"],
            zero_values["quantized_state"][:1].expand_as(
                zero_values["quantized_state"]
            ),
            rtol=0,
            atol=0,
        )
        if not torch.equal(
            zero_values["code_indices"],
            zero_values["code_indices"][:1].expand_as(zero_values["code_indices"]),
        ):
            raise AssertionError("cross-section did not share one VQ regime")

        changed_history = parts.market_feature.clone()
        changed_history[:, :-1, 0] += 10
        with torch.no_grad():
            original_state = model.master.market_encoder(
                parts.market_feature[:, :-1, :]
            )[0]
            changed_state = model.master.market_encoder(
                changed_history[:, :-1, :]
            )[0]
            model.master.market_quantizer.embedding.weight.fill_(1000)
            model.master.market_quantizer.embedding.weight[0].copy_(original_state)
            model.master.market_quantizer.embedding.weight[1].copy_(changed_state)
            model.master.market_adapter.weight.normal_(std=0.01)
        original = capture_forward(model, parts.stock_feature, parts.market_feature)
        torch.testing.assert_close(
            original["delta_weight"],
            original["delta_weight"][:1].expand_as(original["delta_weight"]),
            rtol=1e-5,
            atol=2e-7,
        )
        torch.testing.assert_close(
            original["quantized_state"],
            original["quantized_state"][:1].expand_as(original["quantized_state"]),
            rtol=0,
            atol=0,
        )

        historical = capture_forward(model, parts.stock_feature, changed_history)
        torch.testing.assert_close(
            historical["gate_input"], original["gate_input"], rtol=0, atol=0
        )
        if torch.equal(historical["delta_weight"], original["delta_weight"]):
            raise AssertionError("historical market did not change dynamic weights")
        if torch.equal(historical["code_indices"], original["code_indices"]):
            raise AssertionError("controlled histories did not select different regimes")
        if torch.equal(historical["prediction"], original["prediction"]):
            raise AssertionError("historical market did not affect prediction")

        changed_current = parts.market_feature.clone()
        changed_current[:, -1, 0] += 100
        current = capture_forward(model, parts.stock_feature, changed_current)
        torch.testing.assert_close(
            current["history"], original["history"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            current["market_state"], original["market_state"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            current["quantized_state"], original["quantized_state"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            current["code_indices"], original["code_indices"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            current["delta_weight"], original["delta_weight"], rtol=0, atol=0
        )
        if torch.equal(current["gate_input"], original["gate_input"]):
            raise AssertionError("current market did not change Feature Gate input")
        if torch.equal(current["prediction"], original["prediction"]):
            raise AssertionError("current market did not affect prediction")

        gradient_model = AlphaMasterModule(config).train()
        optimizer = torch.optim.SGD(gradient_model.parameters(), lr=0.1)
        target = parts.target(config["predictor"]["target_day"])
        initial_parameters = {
            "adapter": gradient_model.master.market_adapter.weight.detach().clone(),
            "codebook": gradient_model.master.market_quantizer.embedding.weight.detach().clone(),
            "gru": gradient_model.master.market_encoder.gru.weight_ih_l0.detach().clone(),
        }
        gradient_prediction, gradient_vq = gradient_model(
            parts.stock_feature,
            parts.market_feature,
            return_vq_output=True,
        )
        codebook_after_ema = (
            gradient_model.master.market_quantizer.embedding.weight.detach().clone()
        )
        if torch.equal(initial_parameters["codebook"], codebook_after_ema):
            raise AssertionError("codebook did not update through EMA")
        one_state_quantizer = type(gradient_model.master.market_quantizer)(
            codebook_size=8,
            embedding_dim=63,
            commitment_weight=0.25,
            decay=0.99,
            statistics_level="trading_day",
        )
        full_day_quantizer = type(gradient_model.master.market_quantizer)(
            codebook_size=8,
            embedding_dim=63,
            commitment_weight=0.25,
            decay=0.99,
            statistics_level="trading_day",
        )
        full_day_quantizer.load_state_dict(one_state_quantizer.state_dict(), strict=True)
        shared_state = zero_values["market_state"][:1]
        one_state_quantizer.train()(shared_state)
        full_day_quantizer.train()(shared_state.expand(len(positions), -1))
        torch.testing.assert_close(
            full_day_quantizer.ema_cluster_size,
            one_state_quantizer.ema_cluster_size,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            full_day_quantizer.ema_embedding_sum,
            one_state_quantizer.ema_embedding_sum,
            rtol=0,
            atol=0,
        )
        first_loss = (
            gradient_model.loss_fn(gradient_prediction, target) + gradient_vq.loss
        )
        if not torch.isfinite(first_loss):
            raise AssertionError("combined prediction + VQ loss is not finite")
        first_loss.backward()
        gradients = {
            "adapter": gradient_model.master.market_adapter.weight.grad,
            "gru": gradient_model.master.market_encoder.gru.weight_ih_l0.grad,
        }
        gradient_l1 = {}
        for name, gradient in gradients.items():
            if gradient is None or gradient.abs().sum().item() <= 0:
                raise AssertionError(f"{name} did not receive a valid gradient")
            gradient_l1[name] = gradient.abs().sum().item()
        optimizer.step()
        torch.testing.assert_close(
            gradient_model.master.market_quantizer.embedding.weight,
            codebook_after_ema,
            rtol=0,
            atol=0,
        )
        updated_parameters = {
            "adapter": gradient_model.master.market_adapter.weight,
            "gru": gradient_model.master.market_encoder.gru.weight_ih_l0,
        }
        for name, parameter in updated_parameters.items():
            if torch.equal(initial_parameters[name], parameter):
                raise AssertionError(f"{name} did not update")

        gru = model.master.market_encoder.gru
        quantizer = model.master.market_quantizer
        results[universe] = {
            "input_shape": list(batch.shape),
            "stock_shape": list(parts.stock_feature.shape),
            "market_shape": list(parts.market_feature.shape),
            "historical_market_shape": list(zero_values["history"].shape),
            "market_state_shape": list(zero_values["market_state"].shape),
            "vq_input_shape": list(zero_values["vq_input"].shape),
            "quantized_state_shape": list(zero_values["quantized_state"].shape),
            "code_indices_shape": list(zero_values["code_indices"].shape),
            "hidden_shape": list(zero_values["hidden"].shape),
            "delta_weight_shape": list(zero_values["delta_weight"].shape),
            "prediction_shape": list(prediction.shape),
            "beta": beta,
            "prior_invariant": True,
            "adapter_zero_initialized": True,
            "zero_init_007_equivalence_max_abs_diff": float(
                (prediction - base_prediction).abs().max().item()
            ),
            "current_market_gate_uses_last_day_only": True,
            "historical_branch_uses_previous_19_days_only": True,
            "historical_market_sensitive_with_nonzero_adapter": True,
            "current_market_sensitive_through_feature_gate": True,
            "paths_decoupled": True,
            "single_day_cross_section": True,
            "cross_section_shared_market_state_quantized_regime_and_delta_weight": True,
            "market_encoder": {
                "input_size": gru.input_size,
                "hidden_size": gru.hidden_size,
                "num_layers": gru.num_layers,
                "batch_first": gru.batch_first,
                "bidirectional": gru.bidirectional,
                "dropout": gru.dropout,
            },
            "market_quantizer": {
                "codebook_size": quantizer.codebook_size,
                "embedding_dim": quantizer.embedding_dim,
                "distance": "l2",
                "straight_through": True,
                "commitment_weight": quantizer.commitment_weight,
                "update": "ema",
                "decay": quantizer.decay,
                "statistics_level": quantizer.statistics_level,
                "one_observation_per_trading_day": True,
                "cross_section_size_invariant_ema_update": True,
                "initial_vq_loss": float(zero_values["vq_loss"].item()),
                "initial_codebook_loss": float(zero_values["codebook_loss"].item()),
                "initial_commitment_loss": float(
                    zero_values["commitment_loss"].item()
                ),
                "controlled_history_regime_changed": True,
                "embedding_requires_grad": quantizer.embedding.weight.requires_grad,
                "codebook_updated_by_ema": True,
            },
            "market_adapter": {
                "input_size": model.master.market_adapter.in_features,
                "output_size": model.master.market_adapter.out_features,
                "bias": model.master.market_adapter.bias is not None,
                "adapter_first_gradient_l1": gradient_l1["adapter"],
                "gru_first_gradient_l1": gradient_l1["gru"],
                "ema_vq_gru_adapter_updated": True,
            },
        }
    return results


def measure_codebook_usage(checkpoint, config, data_root):
    """Count one regime assignment per trading day after minimal training."""
    model = AlphaMasterModule.load_strict_checkpoint(checkpoint, config).eval()
    path = data_root / "CN" / "csi300_20_h10_test.pkl"
    with path.open("rb") as stream:
        dataset = pickle.load(stream)
    dates = dataset.get_index().get_level_values("datetime")
    assignments = []
    with torch.no_grad():
        for date in dates.unique():
            positions = np.flatnonzero(dates == date)
            batch = torch.as_tensor(dataset[positions]).float()
            parts = unpack_batch(batch)
            _, vq_output = model(
                parts.stock_feature,
                parts.market_feature,
                return_vq_output=True,
            )
            if not torch.equal(
                vq_output.indices,
                vq_output.indices[:1].expand_as(vq_output.indices),
            ):
                raise AssertionError(f"cross-section has multiple regimes on {date}")
            assignments.append(int(vq_output.indices[0].item()))
    counts = np.bincount(assignments, minlength=model.master.market_quantizer.codebook_size)
    active_codes = np.flatnonzero(counts).tolist()
    return {
        "level": "trading_day",
        "total_assignments": len(assignments),
        "code_counts": counts.tolist(),
        "active_codes": active_codes,
        "active_code_count": len(active_codes),
        "active_fraction": len(active_codes) / len(counts),
        "all_cross_sections_share_one_regime": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="artifacts/014/smoke",
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
    env = dict(
        os.environ,
        WANDB_MODE="offline",
        MPLCONFIGDIR=str(output_dir / "mpl"),
        CUDA_VISIBLE_DEVICES="",
    )
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
    codebook_usage = measure_codebook_usage(checkpoint, config, data_root)

    report = {
        "status": "PASS",
        "forward": forward_results,
        "stage1": {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "runner_discovery": True,
            "minimal_training": True,
            "codebook_usage": codebook_usage,
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
