"""Formal validation for HVQ-Stock evaluation results.

Usage:
    python -m evaluation.validate [--out results]

Exit code 0 iff every check passes. Always writes
``<out>/diagnostics/validation.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol
from .metrics import portfolio_metrics, ranking_metrics

NAN_TOLERATED_COLUMNS = {"ICIR", "RankICIR", "Sharpe", "Sortino", "Calmar"}  # may be mathematically undefined
TABLE_METRICS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
RTOL = 1e-9
ATOL = 1e-12


class _Checker:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=""):
        self.checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    def result(self):
        failures = sum(1 for c in self.checks if not c["passed"])
        return {
            "passed": failures == 0,
            "passes": len(self.checks) - failures,
            "failures": failures,
            "checks": self.checks,
        }


def _close(a, b):
    return np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)


def _fmt4(value) -> str:
    return "NaN" if pd.isna(value) else f"{value:.4f}"


def run_validation(out, expected_experiments=None, expected_seeds=None, expected_methods=None) -> dict:
    out = Path(out)
    c = _Checker()

    manifest_path = out / "metadata" / "manifest.json"
    if not manifest_path.is_file():
        c.check("manifest_exists", False, f"{manifest_path} missing")
        validation = c.result()
        _write(out, validation)
        return validation
    manifest = json.loads(manifest_path.read_text())

    # 1. manifest files exist (validation.json itself is written at the end of
    # this very run, so its parent dir is checked instead of the file)
    missing = []
    for key, rel in manifest["files"].items():
        if "*" in rel:
            if not list(out.glob(rel)):
                missing.append(rel)
        elif rel == "diagnostics/validation.json":
            if not (out / rel).parent.is_dir():
                missing.append(rel)
        elif not (out / rel).is_file():
            missing.append(rel)
    c.check("manifest_files_exist", not missing, f"missing: {missing}" if missing else "all present")

    eval_config = json.loads((out / "metadata" / "eval_config.json").read_text())
    experiments = expected_experiments or eval_config["models"]
    seeds = expected_seeds or eval_config["seeds"]
    methods = expected_methods or eval_config["ensemble"]["methods"]
    market = eval_config["market"]

    seed_df = pd.read_csv(out / "metrics" / "seed_metrics.csv")
    agg_df = pd.read_csv(out / "metrics" / "aggregate_metrics.csv")
    ens_df = pd.read_csv(out / "metrics" / "ensemble_metrics.csv")

    # 2. exact coverage of expected (model, seed) grid
    expected_pairs = {(market, e, s) for e in experiments for s in seeds}
    actual_pairs = set(zip(seed_df["market"], seed_df["model"].astype(str), seed_df["seed"]))
    c.check(
        "seed_metrics_coverage",
        actual_pairs == expected_pairs,
        f"rows={len(seed_df)} expected={len(expected_pairs)} "
        f"missing={sorted(expected_pairs - actual_pairs)} extra={sorted(actual_pairs - expected_pairs)}",
    )

    # 3. primary key uniqueness
    dup = seed_df.duplicated(subset=["market", "model", "seed"]).sum()
    c.check("seed_metrics_unique_keys", dup == 0, f"duplicates={int(dup)}")

    # 4. illegal NaN / Inf
    metric_cols = [c_ for c_ in seed_df.columns if c_ in TABLE_METRICS or c_ == "num_test_days"]
    inf_count = int(np.isinf(seed_df[metric_cols].to_numpy(dtype=float)).sum())
    nan_records = []
    nan_illegal = 0
    for col in metric_cols:
        n_nan = int(seed_df[col].isna().sum())
        if n_nan:
            nan_records.append(f"{col}:{n_nan}")
            if col not in NAN_TOLERATED_COLUMNS:
                nan_illegal += n_nan
    ens_metric_cols = [c_ for c_ in ens_df.columns if c_ in TABLE_METRICS or c_ == "num_test_days"]
    inf_count += int(np.isinf(ens_df[ens_metric_cols].to_numpy(dtype=float)).sum())
    for col in ens_metric_cols:
        n_nan = int(ens_df[col].isna().sum())
        if n_nan:
            nan_records.append(f"ensemble.{col}:{n_nan}")
            if col not in NAN_TOLERATED_COLUMNS:
                nan_illegal += n_nan
    c.check(
        "no_illegal_nan_inf",
        inf_count == 0 and nan_illegal == 0,
        f"inf={inf_count} illegal_nan={nan_illegal} recorded_exceptions={nan_records}",
    )

    # 5. value ranges
    ok = True
    details = []
    for df, tag in ((seed_df, "seed"), (ens_df, "ensemble")):
        for col, cond in (("IC", "abs<=1"), ("RankIC", "abs<=1")):
            bad = df[col].dropna().abs() > 1 + 1e-12
            if bad.any():
                ok = False
                details.append(f"{tag}.{col}")
        if (df["STD"].dropna() < 0).any():
            ok = False
            details.append(f"{tag}.STD<0")
        if (df["MDD"].dropna() > 1e-12).any():
            ok = False
            details.append(f"{tag}.MDD>0")
    c.check("metric_ranges", ok, ",".join(details) if details else "IC/RankIC<=1, STD>=0, MDD<=0")

    # 6. aggregate == mean/std of seed metrics
    mismatches = []
    for (mkt, model), grp in seed_df.groupby(["market", "model"]):
        row = agg_df[(agg_df["market"] == mkt) & (agg_df["model"] == model)]
        if len(row) != 1:
            mismatches.append(f"{model}:aggregate rows={len(row)}")
            continue
        row = row.iloc[0]
        for col in TABLE_METRICS:
            if not _close(row[f"{col}_mean"], grp[col].mean()):
                mismatches.append(f"{model}.{col}_mean")
            if not _close(row[f"{col}_std"], grp[col].std(ddof=protocol.DDOF)):
                mismatches.append(f"{model}.{col}_std")
    c.check("aggregate_consistency", not mismatches, ",".join(mismatches) if mismatches else "exact")

    # 7. tables match metrics at 4 decimals
    table_df = pd.read_csv(out / "tables" / "seed_mean_std.csv", dtype=str)
    mismatches = []
    for _, row in agg_df.iterrows():
        trow = table_df[(table_df["market"] == row["market"]) & (table_df["model"] == row["model"])]
        if len(trow) != 1:
            mismatches.append(f"{row['model']}:table row missing")
            continue
        cell = trow.iloc[0]
        for col in TABLE_METRICS:
            expected = f"{_fmt4(row[f'{col}_mean'])} ± {_fmt4(row[f'{col}_std'])}"
            if str(cell[col]) != expected:
                mismatches.append(f"{row['model']}.{col}: '{cell[col]}' != '{expected}'")
    c.check("seed_table_consistency", not mismatches, ",".join(mismatches) if mismatches else "exact")

    ens_table = pd.read_csv(out / "tables" / "ensemble.csv", dtype=str)
    mismatches = []
    for _, row in ens_df.iterrows():
        trow = ens_table[
            (ens_table["market"] == row["market"])
            & (ens_table["model"] == row["model"])
            & (ens_table["ensemble_method"] == row["ensemble_method"])
        ]
        if len(trow) != 1:
            mismatches.append(f"{row['model']}:table row missing")
            continue
        for col in TABLE_METRICS:
            if str(trow.iloc[0][col]) != _fmt4(row[col]):
                mismatches.append(f"{row['model']}.{col}")
    c.check("ensemble_table_consistency", not mismatches, ",".join(mismatches) if mismatches else "exact")

    # 8. ensemble expected rows, exactly one each
    expected_ens = {(market, e, m) for e in experiments for m in methods}
    actual_ens = list(zip(ens_df["market"], ens_df["model"].astype(str), ens_df["ensemble_method"]))
    dup_ens = len(actual_ens) - len(set(actual_ens))
    c.check(
        "ensemble_rows",
        set(actual_ens) == expected_ens and dup_ens == 0,
        f"rows={len(ens_df)} expected={len(expected_ens)} duplicates={dup_ens}",
    )

    # 9-13. per-curve checks
    curve_issues = []
    metric_issues = []
    for _, row in ens_df.iterrows():
        curve_path = out / "curves" / "ensemble" / f"{row['market']}_{row['model']}.csv"
        if not curve_path.is_file():
            curve_issues.append(f"{row['model']}:curve missing")
            continue
        curve = pd.read_csv(curve_path, parse_dates=["datetime"])
        dates = curve["datetime"]
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            curve_issues.append(f"{row['model']}:dates not ascending/unique")
        curve_cols = ["daily_ret_gross", "cost", "daily_ret_net", "bench_ret", "nav", "bench_nav"]
        if curve[curve_cols].isna().any().any():
            curve_issues.append(f"{row['model']}:NaN in curve")
        if np.isinf(curve[curve_cols].to_numpy()).any():
            curve_issues.append(f"{row['model']}:Inf in curve")
        if not np.allclose(curve["daily_ret_net"], curve["daily_ret_gross"] - curve["cost"], rtol=RTOL, atol=ATOL):
            curve_issues.append(f"{row['model']}:net != gross - cost")
        if not np.allclose(curve["nav"], (1 + curve["daily_ret_net"]).cumprod(), rtol=RTOL, atol=ATOL):
            curve_issues.append(f"{row['model']}:nav mismatch")
        if not np.allclose(curve["bench_nav"], (1 + curve["bench_ret"]).cumprod(), rtol=RTOL, atol=ATOL):
            curve_issues.append(f"{row['model']}:bench_nav mismatch")
        recomputed = portfolio_metrics(curve["daily_ret_net"])
        for col in PORTFOLIO_CHECK_COLUMNS:
            if not _close(recomputed[col], row[col]):
                metric_issues.append(f"{row['model']}.{col}: {recomputed[col]} != {row[col]}")
        if recomputed["num_test_days"] != row["num_test_days"] or recomputed["num_test_days"] != len(curve):
            metric_issues.append(f"{row['model']}.num_test_days")
    c.check("curve_integrity", not curve_issues, ";".join(curve_issues) if curve_issues else "dates/net/nav/bench_nav ok")
    c.check(
        "curve_metrics_recomputation",
        not metric_issues,
        ";".join(metric_issues) if metric_issues else "AR/STD/MDD/Sharpe/Sortino/Calmar recomputed from curves",
    )

    # 13 (protocol): ensemble ranking metrics come from the ensemble score
    from .backtest import load_prediction
    from .ensemble import ensemble_scores
    from .protocol import REPO_ROOT

    ens_rank_issues = []
    for _, row in ens_df.iterrows():
        pred_paths = [REPO_ROOT / p for p in str(row["pred_paths"]).split(";")]
        seeds_used = [int(s) for s in str(row["seeds"]).split(";")]
        preds = {s: load_prediction(p) for s, p in zip(seeds_used, pred_paths)}
        ens = ensemble_scores(preds, method=row["ensemble_method"])
        rank = ranking_metrics(ens["score"], ens["label"])
        for col in ("IC", "ICIR", "RankIC", "RankICIR"):
            if not _close(rank[col], row[col]):
                ens_rank_issues.append(f"{row['model']}.{col}: recomputed {rank[col]} != {row[col]}")
    c.check(
        "ensemble_ranking_from_score",
        not ens_rank_issues,
        ";".join(ens_rank_issues) if ens_rank_issues else "recomputed from ensembled predictions",
    )

    # 14. eval_config holds the full explicit protocol parameter set
    expected_strategy = {
        "class": protocol.STRATEGY_CLASS,
        "topk": protocol.TOPK,
        "n_drop": protocol.N_DROP,
        "method_sell": protocol.METHOD_SELL,
        "method_buy": protocol.METHOD_BUY,
        "hold_thresh": protocol.HOLD_THRESH,
        "only_tradable": protocol.ONLY_TRADABLE,
        "forbid_all_trade_at_limit": protocol.FORBID_ALL_TRADE_AT_LIMIT,
        "risk_degree": protocol.RISK_DEGREE,
        "freq": protocol.FREQ,
    }
    cfg_issues = []
    for k, v in expected_strategy.items():
        if eval_config["strategy"].get(k) != v:
            cfg_issues.append(f"strategy.{k}={eval_config['strategy'].get(k)!r} != {v!r}")
    for k, v in (("open_cost", protocol.OPEN_COST), ("close_cost", protocol.CLOSE_COST), ("min_cost", protocol.MIN_COST)):
        if eval_config["exchange"].get(k) != v:
            cfg_issues.append(f"exchange.{k}")
    if eval_config["account"].get("initial_cash") != protocol.ACCOUNT:
        cfg_issues.append("account.initial_cash")
    if eval_config["account"].get("long_only") is not True or eval_config["account"].get("leverage") is not False:
        cfg_issues.append("account.long_only/leverage")
    c.check("eval_config_protocol_params", not cfg_issues, ",".join(cfg_issues) if cfg_issues else "all explicit")

    # 16. backtest period == formal test split from configs/config.yaml
    periods = protocol.load_periods()
    period_ok = eval_config["periods"]["test"] == periods["test"]
    c.check(
        "test_period",
        period_ok,
        f"eval_config test={eval_config['periods']['test']} config.yaml test={periods['test']}",
    )

    # 17-18. coverage and timing against the real Qlib calendar
    from . import backtest as bt

    bt.init_qlib(market)
    calendar = bt.trading_days(periods["test"][0], periods["test"][1])
    coverage_issues = []
    for _, row in seed_df.iterrows():
        pred = load_prediction(REPO_ROOT / row["pred_path_or_ckpt_path"])
        actual = pd.DatetimeIndex(pred.index.get_level_values("datetime").unique()).sort_values()
        missing = calendar.difference(actual)
        if len(missing):
            coverage_issues.append(f"{row['model']}/seed{row['seed']}: {len(missing)} days missing")
    c.check(
        "prediction_coverage",
        not coverage_issues,
        ";".join(coverage_issues) if coverage_issues else f"all seeds cover {len(calendar)} trading days",
    )

    timing = eval_config["timing"]
    timing_ok = timing.get("adapter_shift") == 0 and timing.get("qlib_internal_shift") == 1
    first_signal_day = None
    if not coverage_issues:
        first_signal_day = actual.min()  # from last loaded prediction (same calendar for all)
    curve_first = pd.read_csv(
        out / "curves" / "ensemble" / f"{market}_{ens_df.iloc[0]['model']}.csv", parse_dates=["datetime"]
    )["datetime"].min()
    if first_signal_day is not None:
        if first_signal_day != calendar.min():
            timing_ok = False
        if curve_first != calendar.min():
            timing_ok = False
    c.check(
        "signal_timing",
        timing_ok,
        f"adapter_shift={timing.get('adapter_shift')} qlib_shift={timing.get('qlib_internal_shift')} "
        f"first_signal_day={first_signal_day} first_trade_day={curve_first} calendar_start={calendar.min()}",
    )

    validation = c.result()
    _write(out, validation)
    return validation


PORTFOLIO_CHECK_COLUMNS = ["AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]


def _write(out: Path, validation: dict) -> None:
    diag = Path(out) / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    with open(diag / "validation.json", "w") as f:
        json.dump(validation, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate formal evaluation results.")
    parser.add_argument("--out", default=str(protocol.REPO_ROOT / "results"))
    args = parser.parse_args()
    validation = run_validation(args.out)
    print(json.dumps({k: validation[k] for k in ("passed", "passes", "failures")}, indent=2))
    for chk in validation["checks"]:
        mark = "PASS" if chk["passed"] else "FAIL"
        print(f"[{mark}] {chk['name']}: {chk['detail']}")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
