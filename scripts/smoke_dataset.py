"""Generate small canonical datasets from real raw feeds; no training/backtest."""

import argparse
import gc
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.dataset import init_data_loader
from dataset.get_dataset import load_config, prepare_dataset
from dataset.schema import GROUP_DIMS, MARKET_INDICES, TOTAL_DIM, unpack_batch, validate_columns


def smoke_config(universe):
    config = load_config(universe)
    config["data_handler_config"].update(
        start_time="2020-06-01", end_time="2021-06-10",
        fit_start_time="2020-07-01", fit_end_time="2020-12-31",
    )
    config["dataset"]["segments"] = {
        "train": ["2020-11-02", "2020-11-04"],
        "valid": ["2021-02-01", "2021-02-03"],
        "test": ["2021-06-01", "2021-06-03"],
    }
    return config


def check_sampler(sampler):
    validate_columns(sampler.columns)
    assert sampler.data_arr.shape[-1] == TOTAL_DIM
    assert sampler.market_indices == MARKET_INDICES[sampler.universe]
    assert sampler.columns.get_level_values(0).value_counts().to_dict() == GROUP_DIMS
    dates = sampler.get_index().get_level_values("datetime")
    for date in dates.unique():
        indices = np.flatnonzero(dates == date)
        parts = unpack_batch(sampler[indices])
        assert [x.shape[1:] for x in parts] == [(20, 158), (13,), (20, 63), (10,)]
        assert all(np.isfinite(x).all() for x in parts[:3])
        # DK_I intentionally retains missing future labels (e.g. suspensions).
        # Preserve that definition; only the information groups must be finite.
        expected_market = sampler.market_frame.loc[:date].iloc[-20:].to_numpy()
        np.testing.assert_array_equal(parts.market_feature,
                                      np.broadcast_to(expected_market, parts.market_feature.shape))
        np.testing.assert_array_equal(parts.target(5), parts.future_returns[:, 4])
        last = sampler.data_arr[[int(sampler._get_indices(*sampler._get_row_col(int(i)))[-1])
                                 for i in indices]].astype(float)
        label_column = sampler.columns.get_loc(("label", "RET_5D"))
        np.testing.assert_array_equal(parts.target(5), last[:, label_column])
    loader, _ = init_data_loader(sampler, shuffle=False)
    parts = unpack_batch(next(iter(loader)))
    assert parts.stock_feature.shape[1:] == (20, 158)


def run(universe, output_dir):
    # Any attempt to read an existing pickle during generation is a hard failure.
    with patch("pickle.load", side_effect=AssertionError("Generation must use raw sources only")):
        targets = prepare_dataset(universe, output_dir=output_dir, config=smoke_config(universe))
    for split, path in targets.items():
        with path.open("rb") as stream:
            sampler = pickle.load(stream)
        check_sampler(sampler)
        print(f"PASS {universe}/{split}: raw generation, groups, horizons, shared market sequence, DataLoader")
        del sampler
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=tuple(MARKET_INDICES), nargs="+", default=list(MARKET_INDICES))
    parser.add_argument("--output-dir", type=Path, help="Optional fresh directory to retain smoke pickles")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="hvq-dataset-smoke-") as temp:
        for universe in args.universe:
            run(universe, args.output_dir or Path(temp))


if __name__ == "__main__":
    main()
