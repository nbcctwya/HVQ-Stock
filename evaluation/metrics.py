"""Metric formulas of Baseline Results Protocol v1.0.

All formulas live here and nowhere else. Conventions (fixed):
252 trading days, log returns, ddof=1, rf=0, MAR=0.
Mathematically undefined cases produce NaN, never 0.
"""

import numpy as np
import pandas as pd

from .protocol import DDOF, MAR_DAILY, TRADING_DAYS

IC_COLUMNS = ("IC", "ICIR", "RankIC", "RankICIR")
PORTFOLIO_COLUMNS = ("AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar")


def daily_ic(scores: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Per-day cross-sectional Pearson (IC_t) and Spearman (RankIC_t).

    Days where a correlation cannot be computed (too few pairs or zero
    variance) are skipped.
    """
    df = pd.DataFrame({"score": scores, "label": labels}).dropna()
    rows = {}
    for dt, day in df.groupby(level="datetime"):
        if len(day) < 2 or day["score"].nunique() < 2 or day["label"].nunique() < 2:
            continue
        rows[dt] = (
            day["score"].corr(day["label"], method="pearson"),
            day["score"].corr(day["label"], method="spearman"),
        )
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["IC", "RankIC"])
    out.index.name = "datetime"
    return out.sort_index()


def _mean_std(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan, np.nan
    mean = series.mean()
    std = series.std(ddof=DDOF) if len(series) > 1 else np.nan
    return mean, std


def ranking_metrics(scores: pd.Series, labels: pd.Series) -> dict:
    """IC / ICIR / RankIC / RankICIR from daily cross-sections.

    ICIR = mean(IC_t) / std(IC_t, ddof=1); NOT annualized (no sqrt(252)).
    """
    daily = daily_ic(scores, labels)
    ic_mean, ic_std = _mean_std(daily["IC"])
    ric_mean, ric_std = _mean_std(daily["RankIC"])
    return {
        "IC": ic_mean,
        "ICIR": ic_mean / ic_std if ic_std and np.isfinite(ic_std) else np.nan,
        "RankIC": ric_mean,
        "RankICIR": ric_mean / ric_std if ric_std and np.isfinite(ric_std) else np.nan,
    }


def portfolio_metrics(daily_ret_net: pd.Series) -> dict:
    """Portfolio metrics from cost-deducted daily simple returns.

    AR      = exp(mean(g) * 252) - 1
    STD     = std(g, ddof=1) * sqrt(252)
    NAV     = [1.0, exp(cumsum(g))];  MDD = min(NAV / cummax(NAV) - 1)
    Sharpe  = sqrt(252) * mean(g) / std(g, ddof=1)          (rf = 0)
    DownDev = sqrt(mean(min(g - MAR, 0)^2))  over ALL trading days
    Sortino = sqrt(252) * mean(g - MAR) / DownDev
    Calmar  = AR / abs(MDD)
    """
    r = pd.Series(daily_ret_net).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        raise ValueError("No valid net return observations.")
    if (r <= -1).any():
        bad = int((r <= -1).sum())
        raise ValueError(f"{bad} daily net return(s) <= -1; log return undefined.")

    g = np.log1p(r)
    n = len(g)
    mean_g = g.mean()
    std_g = g.std(ddof=DDOF) if n > 1 else np.nan

    nav = np.concatenate([[1.0], np.exp(g.cumsum().values)])
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))

    ar = float(np.expm1(mean_g * TRADING_DAYS))
    std = float(std_g * np.sqrt(TRADING_DAYS)) if np.isfinite(std_g) else np.nan
    sharpe = (
        float(np.sqrt(TRADING_DAYS) * mean_g / std_g)
        if np.isfinite(std_g) and std_g != 0
        else np.nan
    )
    downside = np.minimum(g.values - MAR_DAILY, 0.0)
    down_dev = float(np.sqrt(np.mean(downside**2)))
    sortino = (
        float(np.sqrt(TRADING_DAYS) * (mean_g - MAR_DAILY) / down_dev)
        if down_dev != 0
        else np.nan
    )
    calmar = float(ar / abs(mdd)) if mdd != 0 else np.nan

    return {
        "AR": ar,
        "STD": std,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "num_test_days": int(n),
    }
