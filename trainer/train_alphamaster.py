"""Lightning and inference adapters for AlphaMaster on canonical HVQ batches."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from dataset.schema import MARKET_DIM, STOCK_DIM, unpack_batch
from module.alphamaster import MASTER
from utils.test import Cal_IC_IR


class AlphaMasterModule(pl.LightningModule):
    """AlphaMaster with current-gate and discrete historical-adapter paths."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config["alphamaster"]
        universe = config["data"]["universe"]
        beta = model_cfg["beta"][universe]
        market_encoder_cfg = model_cfg["market_encoder"]
        market_quantizer_cfg = model_cfg["market_quantizer"]
        market_adapter_cfg = model_cfg["market_adapter"]
        if market_encoder_cfg["type"] != "gru":
            raise ValueError("Temporal market encoder must be 'gru'")
        if not market_encoder_cfg["batch_first"]:
            raise ValueError("Temporal market encoder must be batch-first")
        if market_encoder_cfg["bidirectional"]:
            raise ValueError("Temporal market encoder must be unidirectional")
        if market_adapter_cfg["type"] != "linear":
            raise ValueError("Market adapter must be linear")
        if market_quantizer_cfg["type"] != "standard_vq":
            raise ValueError("Market quantizer must be standard VQ")
        if market_quantizer_cfg["distance"] != "l2":
            raise ValueError("Market quantizer must use L2 nearest neighbors")
        if not market_quantizer_cfg["straight_through"]:
            raise ValueError("Market quantizer must use straight-through estimation")
        if market_quantizer_cfg["embedding_dim"] != market_encoder_cfg["hidden_size"]:
            raise ValueError("VQ embedding dimension must match market state width")
        if market_adapter_cfg["input_size"] != market_quantizer_cfg["embedding_dim"]:
            raise ValueError("Market adapter input must match quantized state width")
        if market_adapter_cfg["bias"]:
            raise ValueError("Market adapter must not use bias")
        if not market_adapter_cfg["zero_init"]:
            raise ValueError("Market adapter must be zero-initialized")

        self.stock_dim = model_cfg["d_feat"]
        self.market_dim = model_cfg["d_market"]
        self.target_day = config["predictor"]["target_day"]
        if self.stock_dim != STOCK_DIM or self.market_dim != MARKET_DIM:
            raise ValueError(
                f"AlphaMaster requires canonical stock{STOCK_DIM}/market{MARKET_DIM}, "
                f"got stock{self.stock_dim}/market{self.market_dim}"
            )

        self.master = MASTER(
            d_feat=self.stock_dim,
            d_model=model_cfg["d_model"],
            t_nhead=model_cfg["t_nhead"],
            s_nhead=model_cfg["s_nhead"],
            T_dropout_rate=model_cfg["T_dropout_rate"],
            S_dropout_rate=model_cfg["S_dropout_rate"],
            gate_input_start_index=self.stock_dim,
            gate_input_end_index=self.stock_dim + self.market_dim,
            beta=beta,
            market_encoder_input_size=market_encoder_cfg["input_size"],
            market_encoder_hidden_size=market_encoder_cfg["hidden_size"],
            market_encoder_num_layers=market_encoder_cfg["num_layers"],
            market_encoder_dropout=market_encoder_cfg["dropout"],
            market_vq_codebook_size=market_quantizer_cfg["codebook_size"],
            market_vq_embedding_dim=market_quantizer_cfg["embedding_dim"],
            market_vq_commitment_weight=market_quantizer_cfg["commitment_weight"],
            market_adapter_output_size=market_adapter_cfg["output_size"],
        )

    def forward(self, stock_feature, market_feature, return_vq_output=False):
        if stock_feature.shape[:-1] != market_feature.shape[:-1]:
            raise ValueError("stock and market tensors must share [N,T]")
        if stock_feature.shape[-1] != self.stock_dim:
            raise ValueError(f"Expected {self.stock_dim} stock features")
        if market_feature.shape[-1] != self.market_dim:
            raise ValueError(f"Expected {self.market_dim} market features")
        return self.master(
            torch.cat([stock_feature, market_feature], dim=-1),
            return_vq_output=return_vq_output,
        )

    def _get_data(self, batch, batch_idx=0):
        parts = unpack_batch(batch.float())
        # prior_factor is intentionally not returned: pure AlphaMaster has no
        # prior input, parameter, fusion path, or auxiliary loss.
        return (
            parts.stock_feature,
            parts.market_feature,
            parts.target(self.target_day),
        )

    @staticmethod
    def loss_fn(prediction, target):
        mask = torch.isfinite(target)
        if not torch.any(mask):
            raise ValueError("AlphaMaster batch contains no finite targets")
        return torch.mean((prediction[mask] - target[mask]) ** 2)

    def training_step(self, batch, batch_idx):
        stock, market, target = self._get_data(batch, batch_idx)
        prediction, vq_output = self(stock, market, return_vq_output=True)
        prediction_loss = self.loss_fn(prediction, target)
        loss = prediction_loss + vq_output.loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "train_prediction_loss", prediction_loss, on_step=True, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "train_vq_loss", vq_output.loss, on_step=True, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        stock, market, target = self._get_data(batch, batch_idx)
        prediction, vq_output = self(stock, market, return_vq_output=True)
        prediction_loss = self.loss_fn(prediction, target)
        loss = prediction_loss + vq_output.loss
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "val_prediction_loss", prediction_loss, on_step=False, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "val_vq_loss", vq_output.loss, on_step=False, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.config["train"]["learning_rate"]
        )

    @classmethod
    def load_strict_checkpoint(cls, checkpoint_path, config):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = cls(config)
        model.load_state_dict(state_dict, strict=True)
        return model


@torch.no_grad()
def run_alphamaster_inference(model, data_loader, device=None):
    """Return the standard score/label frame and unified IC metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    predictions = []
    targets = []
    for batch in data_loader:
        batch = batch.float().to(device)
        stock, market, target = model._get_data(batch)
        predictions.append(model(stock, market).detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

    positions = data_loader.batch_sampler.ordered_indices()
    index = data_loader.dataset.get_index()[positions]
    prediction = np.concatenate(predictions).reshape(-1)
    target = np.concatenate(targets).reshape(-1)
    if len(index) != len(prediction):
        raise RuntimeError("Prediction/index alignment failed")

    frame = pd.DataFrame(
        {"score": prediction, "label": target}, index=index
    ).sort_index()
    metrics = Cal_IC_IR(frame, column1="label", column2="score")
    return frame, metrics
