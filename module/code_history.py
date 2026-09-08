"""Deterministic historical VQ-code context for transition-aware routing (024).

For every sample ``(instrument, datetime)`` this module precomputes up to
``history_len`` strictly-earlier VQ codes of the same instrument, produced by
the exact frozen Stage 1 (RevIN -> SpatialEncoder -> VectorQuantiser) that the
formal forward uses.  History is bound to sample identity (the split sampler's
positional index), never to batch position or DataLoader iteration order, so
training-date shuffling cannot change any sample's history.

Causality rules (loud fail on any violation):
  * ``(instrument, datetime)`` keys must be unique within each split and
    across splits;
  * split chronology must be strict: all train datetimes < all valid
    datetimes < all test datetimes;
  * train samples read only earlier train observations; valid samples read
    earlier train + earlier valid observations; test samples read earlier
    train + valid + earlier test observations;
  * no sample ever reads an observation at or after its own datetime;
  * only the stock-feature window is encoded; future-return labels are never
    touched.

Samples with insufficient history keep their rows (never dropped): the
``hist_len`` table records the number of valid history codes (0..history_len)
and the routing branch emits a strictly-zero transition state for length 0.
"""

import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.schema import GROUP_SLICES

SPLIT_ORDER = ("train", "valid", "test")
# Split s (index i) may read observations from splits 0..i (inclusive),
# always restricted to datetime < the sample's own datetime.
_SPLIT_VISIBILITY = {"train": ("train",), "valid": ("train", "valid"),
                     "test": ("train", "valid", "test")}


def _index_frame(index, split):
    """Validate the canonical MultiIndex and return a (datetime, instrument) frame."""
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise ValueError(
            f"[{split}] expected a 2-level MultiIndex (datetime, instrument), "
            f"got {type(index).__name__}"
        )
    names = list(index.names)
    if "datetime" not in names or "instrument" not in names:
        raise ValueError(
            f"[{split}] index must have 'datetime' and 'instrument' levels, "
            f"got names={names}"
        )
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(index.get_level_values("datetime")),
        "instrument": index.get_level_values("instrument").astype(str),
    })
    if frame[["instrument", "datetime"]].duplicated().any():
        dup = frame[frame[["instrument", "datetime"]].duplicated(keep=False)]
        raise ValueError(
            f"[{split}] duplicate (instrument, datetime) keys, e.g.:\n{dup.head()}"
        )
    return frame


def _check_split_chronology(frames):
    """Loud fail unless train, valid and test occupy strictly increasing dates."""
    for earlier, later in zip(SPLIT_ORDER[:-1], SPLIT_ORDER[1:]):
        max_early = frames[earlier]["datetime"].max()
        min_late = frames[later]["datetime"].min()
        if not max_early < min_late:
            raise ValueError(
                f"Impossible split chronology: max({earlier})={max_early} is not "
                f"strictly before min({later})={min_late}; cannot guarantee "
                "causal history visibility"
            )


@torch.no_grad()
def encode_split_codes(model, sampler, split, batch_size=1024, device=None):
    """Run the model's frozen Stage 1 over every sample in positional order.

    Uses exactly the formal forward path (revin -> encoder -> quantizer) and
    only the stock-feature window; labels are never read.  Returns an int64
    numpy array of code ids, one per positional index.
    """
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    n = len(sampler)
    codes = np.empty(n, dtype=np.int64)
    num_embed = model.quantizer.num_embed
    for start in range(0, n, batch_size):
        positions = np.arange(start, min(start + batch_size, n))
        batch = torch.as_tensor(
            np.asarray(sampler[positions]), dtype=torch.float32
        ).to(device)
        if batch.ndim != 3 or batch.shape[-1] <= GROUP_SLICES["feature"].stop:
            raise ValueError(
                f"[{split}] malformed sample window shape {tuple(batch.shape)}"
            )
        feature = batch[:, :, GROUP_SLICES["feature"]]
        hidden = model.encoder(model.revin(feature, mode="norm"))
        _, _, (_, _, vq_idx) = model.quantizer(hidden)
        vq_idx = vq_idx.detach().long().cpu().numpy()
        if (vq_idx < 0).any() or (vq_idx >= num_embed).any():
            raise ValueError(
                f"[{split}] frozen Stage 1 produced out-of-range codes"
            )
        codes[positions] = vq_idx
    if was_training:
        model.train()
    return codes


def build_history_tables(frames, codes_by_split, history_len):
    """Pair each sample with its strictly-earlier same-instrument codes.

    Returns ``{split: {"hist_codes": int64 [N, history_len] (oldest first,
    zero-padded), "hist_len": int64 [N]}}``.  No sample is ever dropped.
    """
    tables = {}
    for split in SPLIT_ORDER:
        visible = _SPLIT_VISIBILITY[split]
        frame = frames[split]
        hist_codes = np.zeros((len(frame), history_len), dtype=np.int64)
        hist_len = np.zeros(len(frame), dtype=np.int64)
        # Pool all observations this split is allowed to see, keeping each
        # observation's original positional row as the second index level.
        pool = pd.concat(
            [frames[s].assign(code=codes_by_split[s]) for s in visible],
            keys=visible, names=["split"],
        )
        for instrument, group in pool.groupby("instrument", sort=False):
            group = group.sort_values("datetime", kind="mergesort")
            dt_values = group["datetime"].to_numpy()
            code_values = group["code"].to_numpy()
            own = (group.index.get_level_values("split") == split)
            own_rows = np.asarray(group.index.get_level_values(1))[own]
            own_dts = dt_values[own]
            # searchsorted with side='left' enforces datetime < t strictly,
            # so an observation can never read itself or any future code.
            ends = np.searchsorted(dt_values, own_dts, side="left")
            for row, end in zip(own_rows, ends):
                start = max(0, end - history_len)
                length = end - start
                hist_codes[row, :length] = code_values[start:end]
                hist_len[row] = length
        tables[split] = {"hist_codes": hist_codes, "hist_len": hist_len}
    return tables


def _index_digest(index):
    payload = pd.util.hash_pandas_object(index).values.tobytes()
    return hashlib.md5(payload).hexdigest()


def build_code_history(model, samplers, history_len=4, batch_size=1024,
                       device=None, cache_path=None, provenance_key=""):
    """Build (or load a verified cache of) per-split historical-code tables.

    ``samplers`` maps 'train'/'valid'/'test' to canonical split samplers.
    The cache is only trusted when the provenance key (Stage 1 checkpoint
    identity) and every split's sample identity digest match exactly.
    """
    if set(samplers) != set(SPLIT_ORDER):
        raise ValueError(f"samplers must provide exactly {SPLIT_ORDER}")
    if not isinstance(history_len, int) or history_len < 1:
        raise ValueError(f"history_len must be a positive int, got {history_len}")

    frames = {split: _index_frame(samplers[split].get_index(), split)
              for split in SPLIT_ORDER}
    _check_split_chronology(frames)
    digests = {split: _index_digest(samplers[split].get_index())
               for split in SPLIT_ORDER}

    if cache_path is not None:
        cache_path = str(cache_path)
        try:
            cache = torch.load(cache_path, map_location="cpu")
        except FileNotFoundError:
            cache = None
        if cache is not None:
            meta = cache.get("meta", {})
            if (meta.get("provenance_key") == provenance_key
                    and meta.get("history_len") == history_len
                    and meta.get("digests") == digests):
                return {split: {"hist_codes": cache[split]["hist_codes"],
                                "hist_len": cache[split]["hist_len"],
                                "codes": cache[split]["codes"]}
                        for split in SPLIT_ORDER}
            raise ValueError(
                f"Stale code-history cache at {cache_path}: provenance or "
                "sample identity changed; delete it to rebuild"
            )

    codes_by_split = {}
    for split in SPLIT_ORDER:
        codes = encode_split_codes(model, samplers[split], split,
                                   batch_size=batch_size, device=device)
        if len(codes) != len(frames[split]):
            raise ValueError(f"[{split}] encoded {len(codes)} codes for "
                             f"{len(frames[split])} samples")
        codes_by_split[split] = codes
    tables = build_history_tables(frames, codes_by_split, history_len)
    result = {}
    for split in SPLIT_ORDER:
        tables[split]["codes"] = codes_by_split[split]
        result[split] = {
            key: torch.as_tensor(value, dtype=torch.long)
            for key, value in tables[split].items()
        }

    if cache_path is not None:
        torch.save(
            {"meta": {"provenance_key": provenance_key,
                      "history_len": history_len, "digests": digests},
             **result},
            str(cache_path),
        )
    return result


class TransitionHistoryDataset(Dataset):
    """Bind per-sample history codes to dataset position (never batch order)."""

    def __init__(self, base, hist_codes, hist_len):
        if len(hist_codes) != len(base) or len(hist_len) != len(base):
            raise ValueError(
                f"history tables ({len(hist_codes)}, {len(hist_len)}) must "
                f"match dataset length ({len(base)})"
            )
        if hist_codes.ndim != 2:
            raise ValueError("hist_codes must have shape [N, history_len]")
        self.base = base
        self.hist_codes = torch.as_tensor(hist_codes, dtype=torch.long)
        self.hist_len = torch.as_tensor(hist_len, dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return self.hist_codes[index], self.hist_len[index], self.base[index]

    def get_index(self):
        return self.base.get_index()
