"""Pure metric functions for Baseline Results Protocol v1.0."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

ANNUALIZATION = 252
PORTFOLIO_METRICS = ("AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar")
RANKING_METRICS = ("IC", "ICIR", "RankIC", "RankICIR")


def portfolio_metrics(simple_returns: pd.Series) -> Dict[str, float]:
    """Compute protocol portfolio metrics from daily net simple returns."""
    r = pd.Series(simple_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if (r <= -1).any():
        raise ValueError("daily net return must be greater than -1")
    g = np.log1p(r)
    n = len(g)
    mean = float(g.mean()) if n else np.nan
    std = float(g.std(ddof=1)) if n >= 2 else np.nan
    ar = float(np.expm1(mean * ANNUALIZATION)) if n else np.nan
    annual_std = std * math.sqrt(ANNUALIZATION) if np.isfinite(std) else np.nan
    nav = np.r_[1.0, np.exp(g.cumsum().to_numpy())]
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0)) if n else np.nan
    sharpe = math.sqrt(ANNUALIZATION) * mean / std if std and np.isfinite(std) else np.nan
    downside = float(np.sqrt(np.mean(np.minimum(g.to_numpy(), 0.0) ** 2))) if n else np.nan
    sortino = math.sqrt(ANNUALIZATION) * mean / downside if downside and np.isfinite(downside) else np.nan
    calmar = ar / abs(mdd) if mdd and np.isfinite(mdd) else np.nan
    return {"AR": ar, "STD": annual_std, "MDD": mdd, "Sharpe": sharpe,
            "Sortino": sortino, "Calmar": calmar, "num_test_days": int(n)}


def ranking_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    """Compute daily cross-sectional Pearson/Spearman metrics."""
    required = {"score", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"ranking frame lacks columns: {sorted(required - set(frame.columns))}")
    data = frame[["score", "label"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(data.index, pd.MultiIndex) or "datetime" not in data.index.names:
        raise ValueError("ranking frame must have a MultiIndex with a datetime level")
    grouped = data.groupby(level="datetime", sort=True)
    ic = grouped.apply(lambda x: x["score"].corr(x["label"], method="pearson"), include_groups=False).dropna()
    ric = grouped.apply(lambda x: x["score"].corr(x["label"], method="spearman"), include_groups=False).dropna()

    def mean_ir(values: pd.Series) -> tuple[float, float]:
        mean = float(values.mean()) if len(values) else np.nan
        std = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        return mean, mean / std if std and np.isfinite(std) else np.nan

    ic_mean, icir = mean_ir(ic)
    ric_mean, ricir = mean_ir(ric)
    return {"IC": ic_mean, "ICIR": icir, "RankIC": ric_mean, "RankICIR": ricir}


def curve_from_report(report: pd.DataFrame) -> pd.DataFrame:
    """Convert a Qlib report (whose return is gross) to the protocol curve."""
    curve = pd.DataFrame(index=pd.to_datetime(report.index))
    curve.index.name = "datetime"
    curve["daily_ret_gross"] = pd.to_numeric(report["return"]).astype(float)
    curve["cost"] = pd.to_numeric(report["cost"]).astype(float).fillna(0.0)
    curve["daily_ret_net"] = curve["daily_ret_gross"] - curve["cost"]
    # Qlib benchmark data may be float32.  Promote before cumprod so the saved
    # NAV is reproducible after CSV parsing (which yields float64).
    curve["bench_ret"] = pd.to_numeric(report["bench"]).astype(float)
    if (curve["daily_ret_net"] <= -1).any() or (curve["bench_ret"] <= -1).any():
        raise ValueError("curve contains a daily return <= -1")
    curve["nav"] = (1.0 + curve["daily_ret_net"]).cumprod()
    curve["bench_nav"] = (1.0 + curve["bench_ret"]).cumprod()
    return curve.reset_index()
