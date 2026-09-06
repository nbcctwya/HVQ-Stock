"""AlphaMaster market63 formulas and an independent calendar-based channel."""

import copy

import numpy as np
import pandas as pd
from qlib.contrib.data.handler import check_transform_proc
from qlib.data.dataset import TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor

from dataset.schema import GROUP_SLICES, MARKET_INDICES, STEP_LEN, validate_columns


def market_feature_config(indices):
    """Keep AlphaMaster's formula order and Qlib Mask semantics verbatim."""
    if len(indices) != 3 or len(set(indices)) != 3:
        raise ValueError("market63 requires three distinct indices")
    exprs = ["$close/Ref($close,1)-1"]
    for w in (5, 10, 20, 30, 60):
        exprs += [
            f"Mean($close/Ref($close,1)-1,{w})",
            f"Std($close/Ref($close,1)-1,{w})",
            f"Mean($volume,{w})/$volume",
            f"Std($volume,{w})/$volume",
        ]
    fields = [f'Mask({expr}, "{inst}")' for inst in indices for expr in exprs]
    return fields, list(fields)


class ValidateMarket(Processor):
    """Catch missing index feeds before Fillna could hide them as zero features."""

    def __call__(self, df):
        missing = df.columns[df.isna().all()]
        if len(missing):
            raise ValueError(f"Unavailable market expressions: {list(missing)}")
        return df


class MarketDataHandler(DataHandlerLP):
    def __init__(self, *, universe, market_indices, start_time, end_time,
                 fit_start_time, fit_end_time, infer_processors, instruments=None):
        if tuple(market_indices) != MARKET_INDICES[universe]:
            raise ValueError(f"Incorrect market indices for {universe}: {market_indices}")
        processors = check_transform_proc(copy.deepcopy(infer_processors),
                                          fit_start_time, fit_end_time)
        super().__init__(
            instruments=universe if instruments is None else instruments,
            start_time=start_time, end_time=end_time,
            data_loader={"class": "QlibDataLoader", "kwargs": {
                "config": {"market": market_feature_config(market_indices)}, "freq": "day",
            }},
            infer_processors=[ValidateMarket()] + processors,
            learn_processors=[], process_type=DataHandlerLP.PTYPE_A,
        )


def load_market_frame(config, universe):
    stock = config["data_handler_config"]
    kwargs = {key: stock[key] for key in
              ("start_time", "end_time", "fit_start_time", "fit_end_time", "instruments")}
    kwargs.update(config["market_data_handler_config"])
    handler = MarketDataHandler(universe=universe, **kwargs)
    frame = handler.fetch(col_set=["market"], data_key=DataHandlerLP.DK_I)
    grouped = frame.groupby(level="datetime", sort=True)
    if (grouped.nunique(dropna=False) > 1).any().any():
        raise ValueError("Market information differs across stocks on the same date")
    daily = grouped.first()
    if not np.isfinite(daily.to_numpy()).all():
        raise ValueError("Market information contains non-finite values after preprocessing")
    return daily


class CanonicalSampler(TSDataSampler):
    """Sample canonical Qlib data with a shared market trading calendar.

    Stock/prior/label keep Qlib's ffill+bfill sampling. Market windows are
    selected by endpoint date, never by a stock's suspension/listing gaps.
    Both the input DataFrame and serialized data_arr have all four groups.
    """

    def __init__(self, data, market_frame, universe, start, end):
        validate_columns(data.columns)
        expected = pd.MultiIndex.from_product(
            [["market"], market_feature_config(MARKET_INDICES[universe])[1]])
        if not market_frame.columns.equals(expected) or not market_frame.index.is_unique:
            raise ValueError("Expected ordered market63 columns and one row per trading date")
        if not np.isfinite(market_frame.to_numpy()).all():
            raise ValueError("Market channel must be finite")
        self.columns = data.columns.copy()
        self.universe = universe
        self.market_indices = MARKET_INDICES[universe]
        self.market_frame = market_frame.sort_index()
        super().__init__(data, start=start, end=end, step_len=STEP_LEN, fillna_type="ffill+bfill")
        if not len(self):
            raise ValueError("Cannot create an empty dataset segment")
        missing = self.idx_df.index.difference(self.market_frame.index)
        if len(missing):
            raise ValueError(f"Missing market trading dates: {list(missing[:5])}")
        dates = self.get_index().get_level_values("datetime")
        self._market_positions = self.market_frame.index.get_indexer(dates)
        if (self._market_positions < STEP_LEN - 1).any():
            raise ValueError("Insufficient market history for a complete window")

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = self.get_index().get_loc((pd.Timestamp(index[0]), index[1]))
        batch = super().__getitem__(index).copy()
        endpoints = self._market_positions[index]
        rows = np.asarray(endpoints)[..., None] + np.arange(1 - self.step_len, 1)
        batch[..., GROUP_SLICES["market"]] = self.market_frame.to_numpy()[rows]
        return batch
