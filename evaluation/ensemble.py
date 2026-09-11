"""Prediction-level ensemble (Baseline Results Protocol v1.0).

Ensembles are built ONLY from prediction scores: seed predictions are
aligned on (datetime, instrument) with an inner join, then combined.
Ranking metrics and the portfolio backtest are re-computed from the
ensemble score — averaging per-seed IC/Sharpe/returns is forbidden.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

SUPPORTED_METHODS = ("avg_none",)  # avg_zscore / avg_rank reserved for later


def ensemble_scores(seed_preds: Dict[int, pd.DataFrame], method: str = "avg_none") -> pd.DataFrame:
    """Combine per-seed prediction frames into an ensemble frame.

    Returns a DataFrame indexed by (datetime, instrument) with columns
    [score, label], where score = mean of raw seed scores over the inner
    join of all seeds, and label is the shared label column (verified
    identical across seeds where present).
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported ensemble method: {method}. Supported: {SUPPORTED_METHODS}")
    if not seed_preds:
        raise ValueError("No seed predictions provided for ensemble.")

    seeds = sorted(seed_preds)
    base_index = None
    score_cols = []
    label = None
    for seed in seeds:
        pred = seed_preds[seed]
        if base_index is None:
            base_index = pred.index
        else:
            base_index = base_index.intersection(pred.index)  # inner join
    for seed in seeds:
        pred = seed_preds[seed]
        score_cols.append(pred["score"].reindex(base_index))
        seed_label = pred["label"].reindex(base_index)
        if label is None:
            label = seed_label
        elif not label.equals(seed_label):
            if not np.allclose(label.values, seed_label.values, equal_nan=True):
                raise ValueError(
                    f"Label mismatch between seed {seeds[0]} and seed {seed} on the ensemble join."
                )
    scores = pd.concat(score_cols, axis=1)
    if scores.isna().any().any():
        raise ValueError("NaN score after inner join; index alignment is inconsistent.")
    ensemble = pd.DataFrame({"score": scores.mean(axis=1), "label": label}, index=base_index)
    return ensemble.sort_index()
