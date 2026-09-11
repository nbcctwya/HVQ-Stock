"""Baseline Results Protocol v1.0 — fixed constants and project facts.

Every number here is part of the formal protocol (see RULES.md). Nothing in
this module may be tuned per model, per seed or per market.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

BASELINE_ID = "hvq_stock"
PROTOCOL_VERSION = "1.0"

# --- fixed strategy parameters (always passed explicitly to Qlib) ---
STRATEGY_CLASS = "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy"
TOPK = 30
N_DROP = 5
METHOD_SELL = "bottom"
METHOD_BUY = "top"
HOLD_THRESH = 1
ONLY_TRADABLE = False
FORBID_ALL_TRADE_AT_LIMIT = True
RISK_DEGREE = 0.95
FREQ = "day"

# --- fixed transaction costs ---
OPEN_COST = 0.0005
CLOSE_COST = 0.0015
MIN_COST = 0

# --- fixed account ---
ACCOUNT = 100000000
LONG_ONLY = True
LEVERAGE = False

# --- fixed metric conventions ---
TRADING_DAYS = 252
DDOF = 1
RISK_FREE_RATE = 0.0
MAR_DAILY = 0.0

# --- timing ---
QLIB_SIGNAL_SHIFT = 1  # trade_date = t uses signal from t-1 (Qlib internal)
ADAPTER_SHIFT = 0  # evaluation must NOT shift predictions again

ENSEMBLE_METHOD_DEFAULT = "avg_none"
ENSEMBLE_JOIN = "inner"

DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# market settings read from project facts (dataset/2025_csi300.yaml,
# backtest_qlib.py UNIVERSE_SETTINGS). These are project conventions, not
# protocol-tunable numbers.
MARKET_SETTINGS = {
    "csi300": {
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "region": "cn",
        "instruments": "csi300",
        "benchmark": "SH000300",
    },
}

CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
DATASET_CONFIG_PATH = REPO_ROOT / "dataset" / "2025_csi300.yaml"


def load_train_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_periods() -> dict:
    """Read the canonical train/valid/test split from configs/config.yaml."""
    cfg = load_train_config()
    data = cfg["data"]
    return {
        "train": [str(d) for d in data["train_period"]],
        "valid": [str(d) for d in data["valid_period"]],
        "test": [str(d) for d in data["test_period"]],
    }


def load_label_horizon() -> int:
    """Prediction target horizon in days (predictor.target_day)."""
    cfg = load_train_config()
    return int(cfg["predictor"]["target_day"])


def strategy_kwargs() -> dict:
    """Full explicit TopkDropoutStrategy kwargs (never rely on Qlib defaults)."""
    return {
        "topk": TOPK,
        "n_drop": N_DROP,
        "method_sell": METHOD_SELL,
        "method_buy": METHOD_BUY,
        "hold_thresh": HOLD_THRESH,
        "only_tradable": ONLY_TRADABLE,
        "forbid_all_trade_at_limit": FORBID_ALL_TRADE_AT_LIMIT,
        "risk_degree": RISK_DEGREE,
    }


def exchange_kwargs() -> dict:
    """Explicit Exchange kwargs. limit_threshold=None resolves to the Qlib CN
    region default (C.limit_threshold = 0.095); deal_price is pinned to close
    per project convention."""
    return {
        "freq": FREQ,
        "limit_threshold": None,
        "deal_price": "close",
        "open_cost": OPEN_COST,
        "close_cost": CLOSE_COST,
        "min_cost": MIN_COST,
    }
