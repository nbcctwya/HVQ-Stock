"""The single canonical HVQ layout, shared by generation and all consumers."""

from typing import Any, NamedTuple

STOCK_DIM = 158
PRIOR_DIM = 13
MARKET_DIM = 63
RETURN_DIM = 10
STEP_LEN = 20
GROUP_DIMS = {"feature": STOCK_DIM, "prior": PRIOR_DIM, "market": MARKET_DIM, "label": RETURN_DIM}
TOTAL_DIM = sum(GROUP_DIMS.values())
GROUP_SLICES = {}
_offset = 0
for _group, _width in GROUP_DIMS.items():
    GROUP_SLICES[_group] = slice(_offset, _offset + _width)
    _offset += _width

MARKET_INDICES = {
    "csi300": ("sh000300", "sh000852", "sh000905"),
    "sp500": ("^gspc", "^dji", "^ndx"),
}


class BatchParts(NamedTuple):
    stock_feature: Any
    prior_factor: Any
    market_feature: Any
    future_returns: Any

    def target(self, target_day):
        if not isinstance(target_day, int) or not 1 <= target_day <= RETURN_DIM:
            raise ValueError(f"target_day must be an integer in [1, {RETURN_DIM}]: {target_day}")
        return self.future_returns[:, target_day - 1]


def unpack_batch(batch):
    """Parse [N,T,244] numpy/torch batches without changing dtype or device."""
    if batch.ndim != 3 or batch.shape[1] == 0 or batch.shape[-1] != TOTAL_DIM:
        raise ValueError(f"Expected [N,T,{TOTAL_DIM}] with T > 0, got {tuple(batch.shape)}")
    return BatchParts(
        batch[:, :, GROUP_SLICES["feature"]],
        batch[:, -1, GROUP_SLICES["prior"]],
        batch[:, :, GROUP_SLICES["market"]],
        batch[:, -1, GROUP_SLICES["label"]],
    )


def validate_columns(columns):
    """Validate group order and widths before Qlib loses column metadata."""
    expected = [group for group, width in GROUP_DIMS.items() for _ in range(width)]
    if columns.nlevels != 2 or list(columns.get_level_values(0)) != expected or not columns.is_unique:
        raise ValueError(f"Expected ordered, unique columns with groups {GROUP_DIMS}")


def dataset_basename(universe, step_len=STEP_LEN, horizon=RETURN_DIM):
    if universe not in MARKET_INDICES:
        raise ValueError(f"Only CSI300/SP500 are supported, got {universe}")
    if step_len != STEP_LEN or horizon != RETURN_DIM:
        raise ValueError(f"Canonical datasets require step_len={STEP_LEN}, horizon={RETURN_DIM}")
    return f"{universe}_{step_len}_h{horizon}"
