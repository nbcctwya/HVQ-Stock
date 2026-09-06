"""Read-only dataset/checkpoint compatibility checks; no training or backtest."""

import argparse
import gc
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.dataset import init_data_loader
from dataset.schema import MARKET_INDICES, dataset_basename, unpack_batch


def load_model(checkpoint):
    from trainer.train_ypred import GenerateReturn

    with (ROOT / "configs/config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config["predictor"]["k"] = config["predictor"]["n_expert"] // 2
    # The full Stage2 state includes Stage1 weights; avoid loading a second file.
    with patch.object(GenerateReturn, "load_pretrained_vqvae"):
        model = GenerateReturn(config, T_max=1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def check(universe, split, source, output, model):
    region = "CN" if universe == "csi300" else "US"
    with (source / region / f"{dataset_basename(universe, version=1)}_{split}.pkl").open("rb") as stream:
        old = pickle.load(stream)
    with (output / region / f"{dataset_basename(universe)}_{split}.pkl").open("rb") as stream:
        new = pickle.load(stream)
    assert new.market_indices == MARKET_INDICES[universe]
    assert new.schema.total_dim == 244 and new.step_len == 20
    pd.testing.assert_index_equal(old.get_index(), new.get_index(), exact=True)
    assert old.fillna_type == new.stock_sampler.fillna_type
    # Check every stored legacy row, in bounded chunks (including NaN padding).
    assert old.data_arr.shape == new.stock_sampler.data_arr.shape
    for start in range(0, len(old.data_arr), 10000):
        np.testing.assert_array_equal(old.data_arr[start:start + 10000],
                                      new.stock_sampler.data_arr[start:start + 10000])
    indices = np.unique(np.linspace(0, len(old) - 1, min(32, len(old)), dtype=int))
    a, b = unpack_batch(old[indices]), unpack_batch(new[indices])
    assert [x.shape[1:] for x in b] == [(20, 158), (13,), (20, 63), (10,)]
    for i in (0, 1, 3):
        np.testing.assert_array_equal(a[i], b[i])
    np.testing.assert_array_equal(a.target(5), b.target(5))
    dates = new.get_index().get_level_values("datetime")
    unique_dates = dates.unique().sort_values()
    for date in unique_dates[[0, len(unique_dates) // 2, -1]]:
        day_indices = np.flatnonzero(dates == date)
        market = unpack_batch(new[day_indices]).market_feature
        assert np.isfinite(market).all()
        np.testing.assert_array_equal(market, np.broadcast_to(market[0], market.shape))
    loader, _ = init_data_loader(new, shuffle=False)
    batch = next(iter(loader))
    assert unpack_batch(batch).market_feature.shape[1:] == (20, 63)
    if model is not None:
        before, after = torch.from_numpy(old[indices]).float(), torch.from_numpy(new[indices]).float()
        x1, p1, y1 = model._get_data(before, 0)
        x2, p2, y2 = model._get_data(after, 0)
        with torch.no_grad():
            pred_a = model(x1, p1)[0]
            pred_b = model(x2, p2)[0]
        assert torch.isfinite(pred_a).all() and torch.isfinite(pred_b).all()
        torch.testing.assert_close(pred_a, pred_b, rtol=0, atol=0)
        torch.testing.assert_close(y1, y2, rtol=0, atol=0, equal_nan=True)
        print(f"  checkpoint forward: exact A/B equality, max diff={(pred_a - pred_b).abs().max().item()}")
    print(f"PASS {universe}/{split}: {len(new)} samples; all legacy storage rows unchanged; "
          "244 dims, four groups, market date consistency, target_day=5, DataLoader")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "dataset/data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset/schema_v2_data")
    parser.add_argument("--universe", choices=tuple(MARKET_INDICES), nargs="+", default=list(MARKET_INDICES))
    parser.add_argument("--splits", choices=("train", "valid", "test"), nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--checkpoint", type=Path, help="Optional full Stage2 checkpoint matching main's config")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_model(args.checkpoint) if args.checkpoint else None
    for universe in args.universe:
        for split in args.splits:
            check(universe, split, args.source_dir, args.output_dir, model)
            gc.collect()


if __name__ == "__main__":
    main()
