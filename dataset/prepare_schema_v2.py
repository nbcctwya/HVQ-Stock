"""Generate CSI300/SP500 v2 pickles without overwriting any existing data."""

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import qlib
import yaml

from dataset.market import MarketWindowSampler, load_market_frame
from dataset.schema import MARKET_INDICES, RETURN_DIM, STEP_LEN, dataset_basename, unpack_batch

ROOT = Path(__file__).resolve().parent.parent


def prepare(universe, source_dir=None, output_dir=None, config_path=None):
    """Migrate existing v1 samplers; stock/prior/label are never recomputed."""
    source_dir = Path(source_dir or ROOT / "dataset/data").resolve()
    output_dir = Path(output_dir or ROOT / "dataset/schema_v2_data").resolve()
    if universe not in MARKET_INDICES:
        raise ValueError(f"Only CSI300 and SP500 are supported: {universe}")
    config_path = Path(config_path or ROOT / f"dataset/2025_{universe}.yaml")
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if config["market"] != universe:
        raise ValueError("Dataset config market does not match the requested universe")
    region = config["qlib_init"]["region"]
    step_len = config["task"]["dataset"]["kwargs"]["step_len"]
    if step_len != STEP_LEN:
        raise ValueError(f"Expected step_len={STEP_LEN}")
    old_base = dataset_basename(universe, step_len, RETURN_DIM, version=1)
    new_base = dataset_basename(universe, step_len, RETURN_DIM, version=2)
    splits = ("train", "valid", "test")
    sources = {s: source_dir / region.upper() / f"{old_base}_{s}.pkl" for s in splits}
    targets = {s: output_dir / region.upper() / f"{new_base}_{s}.pkl" for s in splits}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing v1 source sampler: {path}")
    # Fail before expensive computation or any writes if a target already exists.
    for path in targets.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite dataset: {path}")
    qlib.init(provider_uri=str(Path(config["qlib_init"]["provider_uri"]).expanduser()),
              region=region, kernels=2, exp_manager={"class": "MLflowExpManager",
              "module_path": "qlib.workflow.expm", "kwargs": {
                  "uri": str(output_dir / "qlib_runs"), "default_exp_name": "data_only"}})
    market = load_market_frame(config, universe)
    for split in splits:
        with sources[split].open("rb") as stream:
            old = pickle.load(stream)
        if old.step_len != step_len:
            raise ValueError(f"Unexpected source step_len: {old.step_len}")
        new = MarketWindowSampler(old, market, universe)
        positions = np.unique(np.linspace(0, len(old) - 1, min(32, len(old)), dtype=int))
        before = unpack_batch(old[positions], version=1)
        after = unpack_batch(new[positions], version=2)
        for a, b in ((before.stock_feature, after.stock_feature),
                     (before.prior_factor, after.prior_factor),
                     (before.future_returns, after.future_returns)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(before.target(5), after.target(5))
        targets[split].parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects both legacy data and existing v2 outputs.
        with targets[split].open("xb") as stream:
            pickle.dump(new, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{universe}/{split}: {len(new)} samples, {new[0].shape}; "
              f"v1/v2 exact compatibility PASS -> {targets[split]}", flush=True)
        del old, new
        gc.collect()
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=tuple(MARKET_INDICES), default="csi300")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "dataset/data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset/schema_v2_data")
    parser.add_argument("--data-handler-config", type=Path)
    args = parser.parse_args()
    prepare(args.universe, args.source_dir, args.output_dir, args.data_handler_config)


if __name__ == "__main__":
    main()
