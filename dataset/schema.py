"""Versioned HVQ data layout. Market is an independent, currently unused group."""

from dataclasses import dataclass
from typing import NamedTuple, Any

STOCK_DIM = 158
PRIOR_DIM = 13
MARKET_DIM = 63
RETURN_DIM = 10
STEP_LEN = 20
MARKET_INDICES = {
    "csi300": ("sh000300", "sh000852", "sh000905"),
    "sp500": ("^gspc", "^dji", "^ndx"),
}


@dataclass(frozen=True)
class BatchSchema:
    version: int
    stock_dim: int = STOCK_DIM
    prior_dim: int = PRIOR_DIM
    return_dim: int = RETURN_DIM

    def __post_init__(self):
        if self.version not in (1, 2):
            raise ValueError(f"Unsupported schema version: {self.version}")

    @property
    def market_dim(self):
        return MARKET_DIM if self.version == 2 else 0

    @property
    def groups(self):
        groups = [("feature", self.stock_dim), ("prior", self.prior_dim)]
        if self.market_dim:
            groups.append(("market", self.market_dim))
        return groups + [("label", self.return_dim)]

    @property
    def slices(self):
        offset, result = 0, {}
        for name, width in self.groups:
            result[name] = slice(offset, offset + width)
            offset += width
        return result

    @property
    def total_dim(self):
        return sum(width for _, width in self.groups)


SCHEMA_V1 = BatchSchema(1)
SCHEMA_V2 = BatchSchema(2)


class BatchParts(NamedTuple):
    stock_feature: Any
    prior_factor: Any
    market_feature: Any  # None for v1
    future_returns: Any

    def target(self, target_day):
        if not 1 <= target_day <= self.future_returns.shape[-1]:
            raise ValueError(f"target_day must be within the future return block: {target_day}")
        return self.future_returns[:, target_day - 1]


def unpack_batch(batch, *, stock_dim=STOCK_DIM, prior_dim=PRIOR_DIM,
                 return_dim=RETURN_DIM, version=None):
    """Parse numpy/torch batches, accepting v1 or v2 and rejecting unknown widths.

    A legacy outer singleton loader dimension is accepted; the N=1 batch
    dimension is never squeezed away. No dtype/device changes are performed.
    """
    if batch.ndim == 4 and batch.shape[0] == 1:
        batch = batch[0]
    if batch.ndim != 3 or batch.shape[1] == 0:
        raise ValueError(f"Expected [N,T,C] batch, got {tuple(batch.shape)}")
    schemas = [BatchSchema(v, stock_dim, prior_dim, return_dim)
               for v in ((1, 2) if version is None else (version,))]
    schema = next((s for s in schemas if s.total_dim == batch.shape[-1]), None)
    if schema is None:
        raise ValueError(f"Unknown batch width {batch.shape[-1]}; expected "
                         f"{[s.total_dim for s in schemas]}")
    blocks = schema.slices
    return BatchParts(
        batch[:, :, blocks["feature"]], batch[:, -1, blocks["prior"]],
        batch[:, :, blocks["market"]] if schema.market_dim else None,
        batch[:, -1, blocks["label"]],
    )


def unpack_model_batch(batch, config):
    vq = config["vqvae"]
    return unpack_batch(
        batch, stock_dim=vq.get("num_features", STOCK_DIM),
        prior_dim=vq.get("num_prior_factors", PRIOR_DIM),
        return_dim=vq.get("predictor", {}).get("pred_len", RETURN_DIM),
    )


def dataset_basename(universe, step_len=STEP_LEN, horizon=RETURN_DIM, version=2):
    if version == 2 and universe not in MARKET_INDICES:
        raise ValueError(f"Schema v2 only supports CSI300/SP500, got {universe}")
    BatchSchema(version)
    suffix = "schema_v2" if version == 2 else "dl2"
    return f"{universe}_{step_len}_h{horizon}_{suffix}"


def dataset_location(data_config, universe, horizon):
    # Historical universes keep their existing v1 resolution.
    version = data_config.get("schema_version", 2) if universe in MARKET_INDICES else 1
    root = (data_config.get("schema_v2_path", "dataset/schema_v2_data")
            if version == 2 else data_config.get("data_path"))
    return root, dataset_basename(universe, data_config["window_size"], horizon, version)
