"""Stage 1: train AlphaMaster prior-factor head and save its best checkpoint."""

import pickle
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.dataset import init_data_loader
from dataset.schema import dataset_basename
from trainer.train_alphamaster import AlphaMasterModule
from utils import apply_artifact_root, get_root_dir, seed_everything


torch.set_float32_matmul_precision("high")

STAGE1_SEED = 42


def _build_run_name(cfg: DictConfig) -> str:
    if cfg.train.run_name != "auto":
        return str(cfg.train.run_name)
    return f"alphamaster_{cfg.data.universe}_s{STAGE1_SEED}"


def _build_callbacks(cfg: DictConfig, run_name: str):
    checkpoint_dir = Path(get_root_dir()) / cfg.train.save_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    early_cfg = cfg.train.early_stopping

    checkpoint = ModelCheckpoint(
        save_top_k=1,
        save_last=False,
        monitor=early_cfg.monitor,
        mode=early_cfg.mode,
        dirpath=str(checkpoint_dir),
        filename=f"{run_name}" + "-{epoch}-{val_loss:.4f}",
    )
    early_stop = instantiate({
        "_target_": "pytorch_lightning.callbacks.EarlyStopping",
        "monitor": early_cfg.monitor,
        "min_delta": early_cfg.min_delta,
        "patience": early_cfg.patience,
        "verbose": early_cfg.verbose,
        "mode": early_cfg.mode,
    })
    return [checkpoint, early_stop], checkpoint


def _load_canonical_datasets(cfg: DictConfig, region_code: str, universe: str):
    data_path = cfg.data.get("data_path")
    if not data_path:
        raise ValueError("data.data_path is not configured")
    data_path = Path(to_absolute_path(str(data_path)))
    base = dataset_basename(
        universe, cfg.data.window_size, cfg.data.return_horizon
    )
    paths = {
        split: data_path / region_code / f"{base}_{split}.pkl"
        for split in ("train", "valid")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required pickle files not found: " + ", ".join(missing))
    with paths["train"].open("rb") as stream:
        train_dataset = pickle.load(stream)
    with paths["valid"].open("rb") as stream:
        valid_dataset = pickle.load(stream)
    return train_dataset, valid_dataset


def train(cfg: DictConfig, config: dict, train_loader, valid_loader):
    run_name = _build_run_name(cfg)
    model = AlphaMasterModule(config)
    callbacks, checkpoint = _build_callbacks(cfg, run_name)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        callbacks=callbacks,
        max_epochs=cfg.train.num_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.gpu_counts,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        gradient_clip_algorithm="value",
        deterministic=True,
        log_every_n_steps=cfg.train.log_every_n_steps,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
    )
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
    if not checkpoint.best_model_path:
        raise RuntimeError("AlphaMaster Stage 1 did not produce a best checkpoint")
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    print(f"Best validation loss: {checkpoint.best_model_score}")
    return checkpoint.best_model_path


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    artifact_root = apply_artifact_root(cfg)
    if artifact_root:
        print(f"Artifact root: {artifact_root} (checkpoints/res redirected)")

    frozen_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_readonly(frozen_cfg, True)
    config = OmegaConf.to_container(frozen_cfg, resolve=True)

    seed_everything(STAGE1_SEED)
    pl.seed_everything(STAGE1_SEED, workers=True)

    mapping = {"csi300": "CN", "sp500": "US"}
    universe = str(frozen_cfg.data.universe)
    if universe not in mapping:
        raise ValueError(f"Invalid universe: {universe}")

    train_dataset, valid_dataset = _load_canonical_datasets(
        frozen_cfg, mapping[universe], universe
    )
    train_loader, _ = init_data_loader(
        train_dataset, shuffle=True, num_workers=frozen_cfg.train.num_workers
    )
    valid_loader, _ = init_data_loader(
        valid_dataset, shuffle=False, num_workers=frozen_cfg.train.num_workers
    )

    print(f"Seed value: {STAGE1_SEED}")
    print(f"Universe: {universe}")
    train(frozen_cfg, config, train_loader, valid_loader)


if __name__ == "__main__":
    main()
