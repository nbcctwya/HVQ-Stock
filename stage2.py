"""Stage 2: strict AlphaMaster checkpoint load and unified test inference."""

import pickle
import re
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from dataset.dataset import init_data_loader
from dataset.schema import dataset_basename
from trainer.train_alphamaster import (
    AlphaMasterModule,
    run_alphamaster_inference,
)
from utils import apply_artifact_root, get_root_dir, seed_everything


torch.set_float32_matmul_precision("high")

_BEST_CKPT_RE = re.compile(
    r"-epoch=(\d+)-val_loss=([0-9.]+?)(?:-v\d+)?\.ckpt$"
)


def _load_canonical_test_dataset(cfg: DictConfig, region_code: str, universe: str):
    data_path = cfg.data.get("data_path")
    if not data_path:
        raise ValueError("data.data_path is not configured")
    data_path = Path(to_absolute_path(str(data_path)))
    base = dataset_basename(
        universe, cfg.data.window_size, cfg.data.return_horizon
    )
    path = data_path / region_code / f"{base}_test.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Required pickle file not found: {path}")
    with path.open("rb") as stream:
        return pickle.load(stream)


def _resolve_checkpoint(cfg: DictConfig) -> Path:
    saved_model = str(cfg.predictor.saved_model)
    checkpoint_dir = Path(get_root_dir()) / cfg.train.save_dir
    if saved_model == "auto":
        candidates = []
        for path in checkpoint_dir.glob(
            f"alphamaster_{cfg.data.universe}_s42-*.ckpt"
        ):
            match = _BEST_CKPT_RE.search(path.name)
            if match and path.stat().st_size:
                candidates.append((float(match.group(2)), path))
        if not candidates:
            raise FileNotFoundError(
                f"No AlphaMaster best checkpoint found in {checkpoint_dir}"
            )
        return min(candidates, key=lambda item: item[0])[1]

    path = Path(saved_model)
    if not path.is_absolute():
        path = checkpoint_dir / path
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"AlphaMaster checkpoint not found: {path}")
    return path


def evaluate(cfg: DictConfig, config: dict, test_loader):
    checkpoint_path = _resolve_checkpoint(cfg)
    model = AlphaMasterModule.load_strict_checkpoint(checkpoint_path, config)
    print(f"Strict checkpoint load PASS: {checkpoint_path}")

    prediction, metrics = run_alphamaster_inference(model, test_loader)
    run_dir = (
        Path(get_root_dir())
        / cfg.train.save_res
        / f"alphamaster_{cfg.data.universe}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.train.seed
    pred_path = run_dir / f"{seed}_best.pkl"
    metric_path = run_dir / f"{seed}_metric.csv"
    prediction.to_pickle(pred_path)
    pd.DataFrame([metrics], index=["values"]).transpose().to_csv(metric_path)
    print(f"Results saved to {pred_path} and {metric_path}")
    print(metrics)
    return prediction, metrics


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    artifact_root = apply_artifact_root(cfg)
    if artifact_root:
        print(f"Artifact root: {artifact_root} (checkpoints/res redirected)")

    frozen_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_readonly(frozen_cfg, True)
    config = OmegaConf.to_container(frozen_cfg, resolve=True)

    seed_everything(frozen_cfg.train.seed)
    pl.seed_everything(frozen_cfg.train.seed, workers=True)

    mapping = {"csi300": "CN", "sp500": "US"}
    universe = str(frozen_cfg.data.universe)
    if universe not in mapping:
        raise ValueError(f"Invalid universe: {universe}")
    test_dataset = _load_canonical_test_dataset(
        frozen_cfg, mapping[universe], universe
    )
    test_loader, _ = init_data_loader(
        test_dataset, shuffle=False, num_workers=frozen_cfg.train.num_workers
    )
    return evaluate(frozen_cfg, config, test_loader)


if __name__ == "__main__":
    main()
