"""Generate the canonical CSI300/SP500 dataset directly from Qlib and JKP CSV."""

import argparse
import copy
import gc
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.market import CanonicalSampler, load_market_frame
from dataset.schema import (
    GROUP_DIMS, MARKET_INDICES, PRIOR_DIM, RETURN_DIM, STEP_LEN,
    dataset_basename, unpack_batch, validate_columns,
)


class GlobalFactorMerger(Processor):
    """Broadcast a daily factor matrix across the stock instruments."""

    def __init__(self, factor_mat, fields_group):
        self.factor_mat = factor_mat
        self.fields_group = fields_group

    def __call__(self, df):
        dates = df.index.get_level_values("datetime")
        block = pd.DataFrame(
            self.factor_mat.loc[dates].to_numpy(), index=df.index,
            columns=pd.MultiIndex.from_product([[self.fields_group], self.factor_mat.columns]),
        )
        return pd.concat([df, block], axis=1)


class CanonicalGroups(Processor):
    def __call__(self, df):
        df = df.loc[:, list(GROUP_DIMS)]
        validate_columns(df.columns)
        return df


class Alpha158WithJKP(Alpha158):
    """Keep HVQ's Alpha158 and prior processors; append independent market data."""

    def __init__(self, jkp_factor_mat, market_frame, **kwargs):
        if jkp_factor_mat.shape[1] != PRIOR_DIM:
            raise ValueError(f"Expected {PRIOR_DIM} JKP factors, got {jkp_factor_mat.shape[1]}")
        kwargs["infer_processors"] = (
            [GlobalFactorMerger(jkp_factor_mat, "prior")]
            + copy.deepcopy(kwargs.get("infer_processors", []))
            + [GlobalFactorMerger(market_frame["market"], "market"), CanonicalGroups()]
        )
        super().__init__(**kwargs)


def build_factor_matrix(raw_df: pd.DataFrame, query: str, qlib_calendar: pd.Index, window: int) -> pd.DataFrame:
    """Pivot raw factor returns and compute a rolling cumulative return without current-day data."""
    factor_returns = (
        raw_df
        .query(query)
        .pivot(index="date", columns="name", values="ret")
        .sort_index()
    )

    factor_returns.columns = [f"JKP_{c}" for c in factor_returns.columns]
    factor_returns = factor_returns.reindex(qlib_calendar)

    shifted = factor_returns.shift(1)
    rolling_log = np.log1p(shifted).rolling(window=window, min_periods=window).sum()
    cumulative_returns = np.expm1(rolling_log)

    cumulative_returns.columns = [f"{col}_RET{window}D" for col in cumulative_returns.columns]
    return cumulative_returns


def label_config():
    # Preserve HVQ's existing horizon definitions, including RET_1D.
    return ([f"Ref($close, -{h}) / Ref($close, -1) - 1" for h in range(1, RETURN_DIM + 1)],
            [f"RET_{h}D" for h in range(1, RETURN_DIM + 1)])


def load_config(universe):
    if universe not in MARKET_INDICES:
        raise ValueError(f"Only CSI300/SP500 are supported: {universe}")
    with (ROOT / f"dataset/2025_{universe}.yaml").open() as stream:
        return yaml.safe_load(stream)


def default_output_dir():
    with (ROOT / "configs/config.yaml").open() as stream:
        return ROOT / yaml.safe_load(stream)["data"]["data_path"]


def default_jkp_path(universe):
    location = {"csi300": "chn", "sp500": "usa"}[universe]
    return ROOT / "dataset/jkpdata" / f"[{location}]_[all_themes]_[daily]_[vw_cap].csv"


def build_handler(config, jkp_path):
    """Build all four processed Qlib groups; never read a serialized dataset."""
    universe = config["market"]
    location = {"csi300": "chn", "sp500": "usa"}[universe]
    stock = copy.deepcopy(config["data_handler_config"])
    raw = pd.read_csv(jkp_path, parse_dates=["date"])
    calendar = D.calendar(start_time=stock["start_time"], end_time=stock["end_time"])
    factors = build_factor_matrix(raw, f"location=='{location}' and weighting=='vw_cap' and freq=='daily'",
                                  calendar, STEP_LEN)
    market = load_market_frame(config, universe)
    stock["label"] = label_config()
    return Alpha158WithJKP(factors, market, **stock), market


def prepare_segment(handler, market, universe, segment, data_key):
    start, end = map(pd.Timestamp, segment)
    # Extra lookback rows provide the same complete stock windows as TSDatasetH.
    calendar = market.index
    extended_start = calendar[max(0, calendar.searchsorted(start) - STEP_LEN)]
    frame = handler.fetch(selector=slice(extended_start, end), col_set=list(GROUP_DIMS),
                          data_key=data_key).copy()
    return CanonicalSampler(frame, market, universe, start, end)


def prepare_dataset(universe, output_dir=None, jkp_path=None, config=None):
    """Build and serialize each split from raw sources, with exclusive output creation."""
    config = copy.deepcopy(config) if config is not None else load_config(universe)
    if config["market"] != universe or universe not in MARKET_INDICES:
        raise ValueError("Config market must match the requested CSI300/SP500 universe")
    settings = config["dataset"]
    base = dataset_basename(universe, settings["step_len"])
    region = config["qlib_init"]["region"]
    if region != {"csi300": "cn", "sp500": "us"}[universe]:
        raise ValueError("Qlib region does not match the universe")
    output_dir = Path(output_dir or default_output_dir()).resolve()
    jkp_path = Path(jkp_path or default_jkp_path(universe))
    if not jkp_path.is_file():
        raise FileNotFoundError(f"JKP factor CSV not found: {jkp_path}")
    splits = ("train", "valid", "test")
    targets = {split: output_dir / region.upper() / f"{base}_{split}.pkl" for split in splits}
    for path in targets.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite dataset: {path}")
    qlib.init(provider_uri=str(Path(config["qlib_init"]["provider_uri"]).expanduser()),
              region=region, kernels=2, exp_manager={"class": "MLflowExpManager",
              "module_path": "qlib.workflow.expm", "kwargs": {
                  "uri": str(output_dir / "qlib_runs"), "default_exp_name": "data_only"}})
    handler, market = build_handler(config, jkp_path)
    for split in splits:
        data_key = DataHandlerLP.DK_I if split == "test" else DataHandlerLP.DK_L
        sampler = prepare_segment(handler, market, universe, settings["segments"][split], data_key)
        unpack_batch(sampler[[0]])
        targets[split].parent.mkdir(parents=True, exist_ok=True)
        with targets[split].open("xb") as stream:
            pickle.dump(sampler, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{universe}/{split}: {len(sampler)} samples, {sampler[0].shape} -> {targets[split]}", flush=True)
        del sampler
        gc.collect()
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=tuple(MARKET_INDICES), default="csi300")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jkp-path", type=Path, help="Raw JKP CSV (not a pickle)")
    parser.add_argument("--data-handler-config", type=Path)
    args = parser.parse_args()
    config = None
    if args.data_handler_config:
        with args.data_handler_config.open() as stream:
            config = yaml.safe_load(stream)
    prepare_dataset(args.universe, args.output_dir, args.jkp_path, config)


if __name__ == "__main__":
    main()
