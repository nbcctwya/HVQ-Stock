"""Formal evaluation entry point.

Usage:
    python -m evaluation.run --experiments baseline 010 019 025 034
    python -m evaluation.run --experiments 034 --seeds 0 1 --out results_034
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import backtest as bt
from . import protocol
from .discovery import discover
from .ensemble import SUPPORTED_METHODS, ensemble_scores
from .metrics import IC_COLUMNS, PORTFOLIO_COLUMNS, ranking_metrics, portfolio_metrics

SEED_METRICS_COLUMNS = [
    "market", "model", "seed",
    "IC", "ICIR", "RankIC", "RankICIR",
    "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar",
    "num_test_days", "pred_path_or_ckpt_path",
]
ENSEMBLE_METRICS_COLUMNS = [
    "market", "model", "ensemble_method",
    "IC", "ICIR", "RankIC", "RankICIR",
    "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar",
    "num_test_days", "seeds", "pred_paths",
]
TABLE_METRICS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=protocol.REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _evaluate_prediction(pred: pd.DataFrame, market: str, test_start: str, test_end: str):
    """Ranking metrics + unified backtest + portfolio metrics for one frame."""
    bt.check_coverage(pred, test_start, test_end)
    rank = ranking_metrics(pred["score"], pred["label"])
    signal = bt.make_signal(pred)
    curve, meta = bt.run_backtest(signal, market, test_start, test_end)
    port = portfolio_metrics(curve["daily_ret_net"])
    return rank, curve, port, meta


def _fmt_mean_std(mean, std) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="HVQ-Stock formal evaluation (Baseline Results Protocol v1.0)")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment IDs with formal predictions.")
    parser.add_argument("--market", default="csi300", choices=sorted(protocol.MARKET_SETTINGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=protocol.DEFAULT_SEEDS)
    parser.add_argument("--ensemble-methods", nargs="+", default=[protocol.ENSEMBLE_METHOD_DEFAULT],
                        choices=list(SUPPORTED_METHODS))
    parser.add_argument("--out", default=str(protocol.REPO_ROOT / "results"), help="Output results root.")
    args = parser.parse_args()

    out = Path(args.out)
    for sub in ("metrics", "tables", "curves/ensemble", "metadata", "diagnostics"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    periods = protocol.load_periods()
    test_start, test_end = periods["test"]
    label_horizon = protocol.load_label_horizon()

    print(f"[evaluation] market={args.market} experiments={args.experiments} seeds={args.seeds}")
    print(f"[evaluation] formal test split: {test_start} -> {test_end} (from configs/config.yaml)")

    predictions = discover(args.experiments, args.seeds)
    qlib_facts = bt.init_qlib(args.market)

    # --- per-seed evaluation ---
    seed_rows = []
    seed_preds: Dict[str, Dict[int, pd.DataFrame]] = {}
    for experiment in args.experiments:
        for seed in args.seeds:
            fp = predictions[(experiment, seed)]
            pred = bt.load_prediction(fp.path)
            rank, curve, port, meta = _evaluate_prediction(pred, args.market, test_start, test_end)
            row = {
                "market": args.market, "model": experiment, "seed": seed,
                **{k: rank[k] for k in IC_COLUMNS},
                **{k: port[k] for k in PORTFOLIO_COLUMNS},
                "num_test_days": port["num_test_days"],
                "pred_path_or_ckpt_path": fp.rel_path,
            }
            seed_rows.append(row)
            seed_preds.setdefault(experiment, {})[seed] = pred
            print(f"[evaluation] {experiment} seed{seed}: IC={row['IC']:.4f} AR={row['AR']:.4f} "
                  f"Sharpe={row['Sharpe']:.4f} days={row['num_test_days']} ({fp.source})")

    seed_df = pd.DataFrame(seed_rows, columns=SEED_METRICS_COLUMNS)
    seed_df = seed_df.sort_values(["market", "model", "seed"]).reset_index(drop=True)
    seed_df.to_csv(out / "metrics" / "seed_metrics.csv", index=False)

    # --- seed aggregation ---
    agg_rows = []
    for (market, model), grp in seed_df.groupby(["market", "model"]):
        row = {"market": market, "model": model}
        for col in TABLE_METRICS:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std(ddof=protocol.DDOF) if len(grp) > 1 else np.nan
        agg_rows.append(row)
    agg_cols = ["market", "model"] + [f"{c}_{s}" for c in TABLE_METRICS for s in ("mean", "std")]
    agg_df = pd.DataFrame(agg_rows, columns=agg_cols)
    agg_df.to_csv(out / "metrics" / "aggregate_metrics.csv", index=False)
    agg_df = pd.read_csv(out / "metrics" / "aggregate_metrics.csv")  # table formats CSV-round-tripped values

    table_rows = []
    for _, r in agg_df.iterrows():
        row = {"market": r["market"], "model": r["model"]}
        for col in TABLE_METRICS:
            row[col] = _fmt_mean_std(r[f"{col}_mean"], r[f"{col}_std"])
        table_rows.append(row)
    pd.DataFrame(table_rows, columns=["market", "model"] + TABLE_METRICS).to_csv(
        out / "tables" / "seed_mean_std.csv", index=False
    )

    # --- ensemble ---
    ens_rows = []
    table_ens_rows = []
    curve_files = []
    for experiment in args.experiments:
        preds = seed_preds[experiment]
        for method in args.ensemble_methods:
            ens = ensemble_scores(preds, method=method)
            rank, curve, port, meta = _evaluate_prediction(ens, args.market, test_start, test_end)
            seeds_used = sorted(preds)
            pred_paths = [predictions[(experiment, s)].rel_path for s in seeds_used]
            ens_rows.append({
                "market": args.market, "model": experiment, "ensemble_method": method,
                **{k: rank[k] for k in IC_COLUMNS},
                **{k: port[k] for k in PORTFOLIO_COLUMNS},
                "num_test_days": port["num_test_days"],
                "seeds": ";".join(str(s) for s in seeds_used),
                "pred_paths": ";".join(pred_paths),
            })
            curve_name = f"{args.market}_{experiment}.csv"
            curve_out = curve.copy()
            curve_out.insert(0, "datetime", curve_out.index)
            curve_out.to_csv(out / "curves" / "ensemble" / curve_name, index=False)
            curve_files.append(f"curves/ensemble/{curve_name}")
            print(f"[evaluation] {experiment} ensemble({method}): IC={rank['IC']:.4f} "
                  f"AR={port['AR']:.4f} Sharpe={port['Sharpe']:.4f}")

    ens_df = pd.DataFrame(ens_rows, columns=ENSEMBLE_METRICS_COLUMNS)
    ens_df.to_csv(out / "metrics" / "ensemble_metrics.csv", index=False)
    # format the display table from the CSV-round-tripped values so tables are
    # byte-consistent with what validation re-reads (float repr boundary cases)
    ens_df = pd.read_csv(out / "metrics" / "ensemble_metrics.csv")

    for _, r in ens_df.iterrows():
        row = {"market": r["market"], "model": r["model"], "ensemble_method": r["ensemble_method"]}
        for col in TABLE_METRICS:
            row[col] = f"{r[col]:.4f}" if pd.notna(r[col]) else "NaN"
        table_ens_rows.append(row)
    pd.DataFrame(table_ens_rows, columns=["market", "model", "ensemble_method"] + TABLE_METRICS).to_csv(
        out / "tables" / "ensemble.csv", index=False
    )

    # --- metadata ---
    settings = protocol.MARKET_SETTINGS[args.market]
    calendar_end = None
    try:
        cal = bt.trading_days("2000-01-01", "2030-12-31")
        calendar_end = str(cal.max().date())
    except Exception:
        pass
    eval_config = {
        "protocol": "Baseline Results Protocol v1.0",
        "baseline": protocol.BASELINE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "market": args.market,
        "models": args.experiments,
        "seeds": args.seeds,
        "periods": periods,
        "label": {
            "horizon_days": label_horizon,
            "expression": f"Ref($close,-{label_horizon})/Ref($close,-1)-1",
            "normalization": "raw label on test split (CSRankNorm applied during training only)",
        },
        "strategy": {
            "class": protocol.STRATEGY_CLASS,
            **protocol.strategy_kwargs(),
            "freq": protocol.FREQ,
        },
        "exchange": {
            "provider_uri": qlib_facts["provider_uri"],
            "region": qlib_facts["region"],
            "instruments": qlib_facts["instruments"],
            "benchmark": qlib_facts["benchmark"],
            "deal_price": "close",
            "limit_threshold": None,
            "limit_threshold_resolved": qlib_facts["region_limit_threshold"],
            "limit_note": "limit_threshold=None resolves to the Qlib CN region default "
                          f"(C.limit_threshold={qlib_facts['region_limit_threshold']}); "
                          "stocks at the limit are not tradable",
            "trade_unit": qlib_facts["trade_unit"],
            "open_cost": protocol.OPEN_COST,
            "close_cost": protocol.CLOSE_COST,
            "min_cost": protocol.MIN_COST,
            "executor": "qlib.backtest.executor.SimulatorExecutor(time_per_step=day, "
                        "generate_portfolio_metrics=True)",
            "qlib_version": qlib_facts["qlib_version"],
            "universe_note": "instruments use Qlib dynamic historical constituents",
        },
        "account": {
            "initial_cash": protocol.ACCOUNT,
            "long_only": protocol.LONG_ONLY,
            "leverage": protocol.LEVERAGE,
        },
        "timing": {
            "signal_date": "t-1",
            "trade_date": "t",
            "qlib_internal_shift": protocol.QLIB_SIGNAL_SHIFT,
            "adapter_shift": protocol.ADAPTER_SHIFT,
            "label_horizon_days": label_horizon,
        },
        "return_semantics": {
            "report_return_is_gross": True,
            "evidence": "qlib 0.9.7 qlib/backtest/account.py: return_rate=(earning+cost)/last_account_value",
            "net_return": "daily_ret_net = report['return'] - report['cost'] (cost deducted exactly once)",
        },
        "metrics": {
            "trading_days": protocol.TRADING_DAYS,
            "ddof": protocol.DDOF,
            "risk_free_rate": protocol.RISK_FREE_RATE,
            "mar_daily": protocol.MAR_DAILY,
            "ic": "IC_t = Pearson(score, label) per day; IC = mean(IC_t); ICIR = mean/std(ddof=1), not annualized",
            "portfolio": {
                "AR": "exp(mean(log1p(r_net)) * 252) - 1",
                "STD": "std(log1p(r_net), ddof=1) * sqrt(252)",
                "MDD": "min(NAV / cummax(NAV) - 1), NAV = [1.0, exp(cumsum(g))]",
                "Sharpe": "sqrt(252) * mean(g) / std(g, ddof=1)",
                "Sortino": "sqrt(252) * mean(g - MAR) / sqrt(mean(min(g - MAR, 0)^2)) over ALL days",
                "Calmar": "AR / abs(MDD)",
            },
            "undefined": "mathematically undefined cases are NaN, never 0",
        },
        "ensemble": {
            "enabled": True,
            "methods": args.ensemble_methods,
            "join": protocol.ENSEMBLE_JOIN,
            "normalize": None,
            "score_formula": "mean of raw seed scores on the (datetime, instrument) inner join",
            "ranking_metrics_source": "recomputed from ensemble score; backtest re-run on ensemble score",
        },
        "data": {
            "calendar_end": calendar_end,
            "handler_config": "dataset/2025_csi300.yaml",
            "train_config": "configs/config.yaml",
        },
    }
    with open(out / "metadata" / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    manifest = {
        "schema_version": "1.0",
        "baseline": protocol.BASELINE_ID,
        "description": "HVQ-Stock unified evaluation results",
        "primary_keys": {
            "seed_metrics": ["market", "model", "seed"],
            "aggregate_metrics": ["market", "model"],
            "ensemble_metrics": ["market", "model", "ensemble_method"],
        },
        "files": {
            "seed_metrics": "metrics/seed_metrics.csv",
            "aggregate_metrics": "metrics/aggregate_metrics.csv",
            "seed_table": "tables/seed_mean_std.csv",
            "eval_config": "metadata/eval_config.json",
            "validation": "diagnostics/validation.json",
            "ensemble_metrics": "metrics/ensemble_metrics.csv",
            "ensemble_table": "tables/ensemble.csv",
            "ensemble_curves": "curves/ensemble/*.csv",
        },
    }
    with open(out / "metadata" / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # --- validation (part of every formal run) ---
    from .validate import run_validation

    validation = run_validation(out, expected_experiments=args.experiments, expected_seeds=args.seeds,
                                expected_methods=args.ensemble_methods)
    print(f"[evaluation] validation: passes={validation['passes']} failures={validation['failures']}")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
