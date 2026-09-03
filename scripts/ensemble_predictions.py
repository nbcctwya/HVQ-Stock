#!/usr/bin/env python
"""Average stage2 prediction scores from multiple seed runs."""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an ensemble *_best.pkl by averaging stage2 prediction scores."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Directory containing seed files such as 0_best.pkl, 1_best.pkl, ...",
    )
    source.add_argument(
        "--pred",
        type=Path,
        nargs="+",
        help="Explicit prediction pickle files to ensemble.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seed ids used with --run-dir. Default: 0 1 2 3 4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle path. Defaults to RUN_DIR/ensemble_<N>seed_best.pkl.",
    )
    parser.add_argument(
        "--score-col",
        default="score",
        help="Prediction score column to average. Default: score.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Optional label column to keep from the first file. Default: label.",
    )
    parser.add_argument(
        "--allow-different-labels",
        action="store_true",
        help="Keep the first label column even if later files have different labels.",
    )
    return parser.parse_args()


def _resolve_prediction_paths(args: argparse.Namespace) -> List[Path]:
    if args.pred:
        paths = [path.expanduser() for path in args.pred]
    else:
        run_dir = args.run_dir.expanduser()
        paths = [run_dir / f"{seed}_best.pkl" for seed in args.seeds]

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prediction files: " + ", ".join(missing))
    return paths


def _default_output_path(args: argparse.Namespace, paths: List[Path]) -> Path:
    if args.output is not None:
        return args.output.expanduser()
    output_dir = args.run_dir.expanduser() if args.run_dir else paths[0].parent
    return output_dir / f"ensemble_{len(paths)}seed_best.pkl"


def _load_prediction(path: Path, score_col: str) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=frame.name or score_col)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame or Series in {path}, got {type(frame)!r}.")
    if score_col not in frame.columns:
        raise KeyError(f"{path} does not contain score column {score_col!r}. Columns: {list(frame.columns)}")
    if frame[score_col].isna().any():
        missing = int(frame[score_col].isna().sum())
        raise ValueError(f"{path} has {missing} missing values in {score_col!r}.")
    return frame.sort_index()


def _validate_labels(
    frames: List[pd.DataFrame],
    paths: List[Path],
    label_col: str,
    allow_different_labels: bool,
) -> None:
    if label_col not in frames[0].columns:
        return
    for frame, path in zip(frames[1:], paths[1:]):
        if label_col not in frame.columns:
            if allow_different_labels:
                continue
            raise KeyError(f"{path} does not contain label column {label_col!r}.")
        if not frames[0][label_col].equals(frame[label_col]) and not allow_different_labels:
            raise ValueError(
                f"Label column {label_col!r} differs in {path}. "
                "Pass --allow-different-labels to keep labels from the first file."
            )


def build_ensemble(
    paths: List[Path],
    output_path: Path,
    score_col: str,
    label_col: str,
    allow_different_labels: bool,
) -> pd.DataFrame:
    frames = [_load_prediction(path, score_col) for path in paths]
    index = frames[0].index
    for frame, path in zip(frames[1:], paths[1:]):
        if not frame.index.equals(index):
            raise ValueError(
                f"Index mismatch in {path}. "
                "Align or regenerate predictions before ensembling."
            )

    _validate_labels(frames, paths, label_col, allow_different_labels)

    ensemble = frames[0].copy()
    ensemble[score_col] = sum(frame[score_col].astype(float) for frame in frames) / len(frames)
    if label_col in frames[0].columns:
        ensemble[label_col] = frames[0][label_col]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_pickle(output_path)
    return ensemble


def main() -> None:
    args = _parse_args()
    paths = _resolve_prediction_paths(args)
    output_path = _default_output_path(args, paths)
    ensemble = build_ensemble(
        paths=paths,
        output_path=output_path,
        score_col=args.score_col,
        label_col=args.label_col,
        allow_different_labels=args.allow_different_labels,
    )

    print(f"Ensembled {len(paths)} prediction files:")
    for path in paths:
        print(f"  {path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(ensemble):,}")
    if isinstance(ensemble.index, pd.MultiIndex) and "datetime" in ensemble.index.names:
        dates = ensemble.index.get_level_values("datetime")
        print(f"Date range: {dates.min()} -> {dates.max()}")
    if isinstance(ensemble.index, pd.MultiIndex) and "instrument" in ensemble.index.names:
        instruments = ensemble.index.get_level_values("instrument")
        print(f"Instruments: {instruments.nunique():,}")


if __name__ == "__main__":
    main()
