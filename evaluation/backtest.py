"""Unified formal Qlib backtest for Baseline Results Protocol v1.0.

This module re-runs the backtest from the prediction signal with the full
explicit protocol parameter set. It never copies AR/STD/MDD/Sharpe/Sortino/
Calmar from any existing backtest output.

Timing: the prediction datetime is the signal date. Qlib's
TopkDropoutStrategy trades at date t using the signal of the previous
trading step (internal shift = 1). The adapter passes the prediction
UNSHIFTED; shifting here would double-lag the signal.

Verified return semantics (qlib 0.9.7, qlib/backtest/account.py):
report["return"] = (earning + cost) / last_account_value  -> GROSS return;
report["cost"] = now_cost / last_account_value. Therefore
daily_ret_net = report["return"] - report["cost"] deducts cost exactly once.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import protocol

_QLIB_INITIALIZED = False


def init_qlib(market: str) -> dict:
    """Initialize Qlib once for the market and return the resolved facts."""
    global _QLIB_INITIALIZED
    settings = protocol.MARKET_SETTINGS[market]
    provider_uri = str(Path(settings["provider_uri"]).expanduser())

    import qlib
    from qlib.constant import REG_CN, REG_US

    region = REG_US if settings["region"].lower() == "us" else REG_CN
    if not _QLIB_INITIALIZED:
        qlib.init(provider_uri=provider_uri, region=region)
        _QLIB_INITIALIZED = True

    from qlib.config import C

    return {
        "provider_uri": provider_uri,
        "region": settings["region"],
        "instruments": settings["instruments"],
        "benchmark": settings["benchmark"],
        "qlib_version": qlib.__version__,
        "trade_unit": getattr(C, "trade_unit", None),
        "region_limit_threshold": getattr(C, "limit_threshold", None),
        "region_deal_price": getattr(C, "deal_price", None),
    }


def trading_days(start_time: str, end_time: str) -> pd.DatetimeIndex:
    from qlib.data import D

    cal = D.calendar(start_time=start_time, end_time=end_time, freq=protocol.FREQ)
    return pd.DatetimeIndex(pd.to_datetime(cal))


def load_prediction(pred_path: Path) -> pd.DataFrame:
    """Load a formal prediction pickle -> DataFrame indexed by
    (datetime, instrument) with columns [score, label]."""
    pred = pd.read_pickle(pred_path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame(name=pred.name or "score")
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"Expected DataFrame/Series from {pred_path}, got {type(pred)!r}")
    pred = pred.copy()
    if not isinstance(pred.index, pd.MultiIndex) or pred.index.nlevels != 2:
        raise ValueError(f"Prediction must have a 2-level MultiIndex: {pred_path}")
    names = list(pred.index.names)
    pred.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(pred.index.get_level_values(0)),
            pred.index.get_level_values(1).astype(str),
        ],
        names=["datetime", "instrument"],
    )
    cols = {c.lower(): c for c in pred.columns}
    if "score" not in cols or "label" not in cols:
        raise ValueError(
            f"Prediction must contain score and label columns; found {list(pred.columns)} "
            f"(index names were {names}) in {pred_path}"
        )
    pred = pred.rename(columns={cols["score"]: "score", cols["label"]: "label"})
    pred = pred[["score", "label"]].astype(float)
    if pred.index.duplicated().any():
        raise ValueError(f"Duplicate (datetime, instrument) entries in {pred_path}")
    return pred.sort_index()


def check_coverage(pred: pd.DataFrame, start_time: str, end_time: str) -> None:
    """Prediction must cover every Qlib trading day of the formal test split.
    Insufficient coverage is a hard failure, never a silent shortening."""
    expected = trading_days(start_time, end_time)
    actual = pd.DatetimeIndex(pred.index.get_level_values("datetime").unique()).sort_values()
    missing = expected.difference(actual)
    if len(missing) > 0:
        raise ValueError(
            f"Prediction coverage is shorter than the formal test split: "
            f"{len(missing)} trading day(s) missing, first {missing[:5].strftime('%Y-%m-%d').tolist()}"
        )


def make_signal(pred: pd.DataFrame) -> pd.Series:
    """Qlib signal: (datetime, instrument) -> score, UNSHIFTED."""
    return pred["score"].rename("score")


def run_backtest(
    signal: pd.Series,
    market: str,
    start_time: str,
    end_time: str,
) -> Tuple[pd.DataFrame, dict]:
    """Run the unified backtest and return (curve, report_meta).

    curve columns: datetime, daily_ret_gross, cost, daily_ret_net,
    bench_ret, nav, bench_nav.
    """
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.strategy import TopkDropoutStrategy

    settings = protocol.MARKET_SETTINGS[market]
    strategy = TopkDropoutStrategy(signal=signal, **protocol.strategy_kwargs())
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": protocol.FREQ,
            "generate_portfolio_metrics": True,
        },
    }
    exchange_kwargs = protocol.exchange_kwargs()
    exchange_kwargs["codes"] = settings["instruments"]

    portfolio_metric_dict, _ = qlib_backtest(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        executor=executor_config,
        account=protocol.ACCOUNT,
        benchmark=settings["benchmark"],
        exchange_kwargs=exchange_kwargs,
    )
    entry = portfolio_metric_dict.get(protocol.FREQ)
    if entry is None:
        # qlib 0.9.7 keys the result by the executor freq alias (e.g. "1day")
        if len(portfolio_metric_dict) != 1:
            raise ValueError(f"Unexpected portfolio_metric_dict keys: {list(portfolio_metric_dict)}")
        entry = next(iter(portfolio_metric_dict.values()))
    report, _ = entry

    curve = pd.DataFrame(index=pd.to_datetime(report.index))
    curve.index.name = "datetime"
    curve["daily_ret_gross"] = report["return"].astype(float).values
    curve["cost"] = report["cost"].astype(float).fillna(0.0).values
    curve["daily_ret_net"] = curve["daily_ret_gross"] - curve["cost"]
    curve["bench_ret"] = report["bench"].astype(float).values
    curve["nav"] = (1.0 + curve["daily_ret_net"]).cumprod()
    curve["bench_nav"] = (1.0 + curve["bench_ret"]).cumprod()

    if (curve["daily_ret_net"] <= -1).any():
        raise ValueError("Backtest produced daily net return <= -1; log return undefined.")

    meta = {
        "first_trade_date": str(curve.index.min().date()),
        "last_trade_date": str(curve.index.max().date()),
        "num_trade_days": int(len(curve)),
    }
    return curve, meta
