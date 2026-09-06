"""AlphaMaster market63 formulas and an independent calendar-based channel."""

import copy

import numpy as np
import pandas as pd
from qlib.contrib.data.handler import check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor

from dataset.schema import MARKET_INDICES, MARKET_DIM, SCHEMA_V1, SCHEMA_V2, unpack_batch


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


class MarketWindowSampler:
    """Wrap an unchanged v1 TSDataSampler with a separate market calendar.

    The v1 sampler is embedded in the new pickle, so no reference to an old
    pickle path is required. Its sampling, fill rules and row order remain
    exact. Market windows depend only on the endpoint date, including for
    suspended/newly listed stocks. A missing market date fails explicitly.
    """

    schema = SCHEMA_V2

    def __init__(self, stock_sampler, market_frame, universe):
        if universe not in MARKET_INDICES:
            raise ValueError(f"Unsupported v2 universe: {universe}")
        if len(stock_sampler) == 0:
            raise ValueError("Cannot wrap an empty stock sampler")
        unpack_batch(stock_sampler[0][None], version=1)
        expected_columns = pd.MultiIndex.from_product(
            [["market"], market_feature_config(MARKET_INDICES[universe])[1]])
        if not market_frame.columns.equals(expected_columns) or not market_frame.index.is_unique:
            raise ValueError("Expected one market63 row per trading date")
        self.stock_sampler = stock_sampler
        self.universe = universe
        self.market_indices = MARKET_INDICES[universe]
        self.step_len = stock_sampler.step_len
        self.market_frame = market_frame.sort_index()
        self._dates = stock_sampler.get_index().get_level_values("datetime")
        # Validate all required dates, including the complete lookback calendar.
        calendar = stock_sampler.idx_df.index
        missing = calendar.difference(self.market_frame.index)
        if len(missing):
            raise ValueError(f"Missing market trading dates: {list(missing[:5])}")
        positions = self.market_frame.index.get_indexer(self._dates)
        if (positions < self.step_len - 1).any():
            raise ValueError("Market history is too short for a complete window")
        if not np.isfinite(self.market_frame.to_numpy()).all():
            raise ValueError("Market channel must be finite")
        self._positions = positions
        self.columns = pd.MultiIndex.from_tuples([
            (group, self.market_frame.columns[i][1] if group == "market" else f"{group}_{i}")
            for group, width in self.schema.groups for i in range(width)
        ])

    def __len__(self):
        return len(self.stock_sampler)

    def get_index(self):
        return self.stock_sampler.get_index()

    def config(self, **kwargs):
        self.stock_sampler.config(**kwargs)

    def __getitem__(self, index):
        # Keep exact integer, vectorized and (date, instrument) sample identity.
        if isinstance(index, tuple):
            index = self.get_index().get_loc((pd.Timestamp(index[0]), index[1]))
        old = self.stock_sampler[index]
        endpoints = self._positions[index]
        offsets = np.arange(1 - self.step_len, 1)
        rows = np.asarray(endpoints)[..., None] + offsets
        market = self.market_frame.to_numpy()[rows].astype(old.dtype, copy=False)
        split = SCHEMA_V1.slices["label"].start
        return np.concatenate([old[..., :split], market, old[..., SCHEMA_V1.slices["label"]]], axis=-1)
