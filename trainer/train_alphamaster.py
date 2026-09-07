"""Lightning and inference adapters for AlphaMaster on canonical HVQ batches."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from dataset.schema import MARKET_DIM, STOCK_DIM, unpack_batch
from module.alphamaster import MASTER
from utils.test import Cal_IC_IR


class AlphaMasterModule(pl.LightningModule):
    """AlphaMaster with continuous warm-up then the standard VQ path."""

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
        self.warmup_epochs = config["train"]["warmup_epochs"]
        if isinstance(self.warmup_epochs, bool) or not isinstance(
            self.warmup_epochs, int
        ):
            raise ValueError("train.warmup_epochs must be an integer")
        if self.warmup_epochs < 0:
            raise ValueError("train.warmup_epochs must be non-negative")
        self.vq_switch_diagnostics = None
        self._reset_vq_switch_accumulators()
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

    def forward(
        self,
        stock_feature,
        market_feature,
        use_vq=True,
        return_vq_output=False,
        return_switch_output=False,
    ):
        if stock_feature.shape[:-1] != market_feature.shape[:-1]:
            raise ValueError("stock and market tensors must share [N,T]")
        if stock_feature.shape[-1] != self.stock_dim:
            raise ValueError(f"Expected {self.stock_dim} stock features")
        if market_feature.shape[-1] != self.market_dim:
            raise ValueError(f"Expected {self.market_dim} market features")
        return self.master(
            torch.cat([stock_feature, market_feature], dim=-1),
            use_vq=use_vq,
            return_vq_output=return_vq_output,
            return_switch_output=return_switch_output,
        )

    def use_vq_at_epoch(self, epoch):
        """Return the single protocol switch: ``epoch >= warmup_epochs``."""
        return int(epoch) >= self.warmup_epochs

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

    def _objective(self, stock, market, target, epoch, collect_switch=False):
        use_vq = self.use_vq_at_epoch(epoch)
        switch_output = None
        if use_vq:
            if collect_switch:
                prediction, vq_output, switch_output = self(
                    stock,
                    market,
                    use_vq=True,
                    return_vq_output=True,
                    return_switch_output=True,
                )
            else:
                prediction, vq_output = self(
                    stock, market, use_vq=True, return_vq_output=True
                )
            vq_loss = vq_output.loss
        else:
            # The VQ module remains constructed but is not called or optimized
            # during warm-up; the adapter receives the continuous GRU state.
            prediction = self(stock, market, use_vq=False)
            vq_output = None
            vq_loss = prediction.new_zeros(())
        prediction_loss = self.loss_fn(prediction, target)
        loss = prediction_loss + vq_loss
        return loss, prediction_loss, vq_loss, vq_output, switch_output

    def training_step(self, batch, batch_idx):
        stock, market, target = self._get_data(batch, batch_idx)
        loss, prediction_loss, vq_loss, _, _ = self._objective(
            stock, market, target, self.current_epoch
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "train_prediction_loss", prediction_loss, on_step=True, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "train_vq_loss", vq_loss, on_step=True, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "train_vq_enabled", float(self.use_vq_at_epoch(self.current_epoch)),
            on_step=False, on_epoch=True, logger=True, sync_dist=True,
            batch_size=target.numel(),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        stock, market, target = self._get_data(batch, batch_idx)
        collect_switch = self.current_epoch == self.warmup_epochs
        loss, prediction_loss, vq_loss, vq_output, switch_output = self._objective(
            stock,
            market,
            target,
            self.current_epoch,
            collect_switch=collect_switch,
        )
        if collect_switch:
            self._accumulate_vq_switch(vq_output, switch_output)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "val_prediction_loss", prediction_loss, on_step=False, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "val_vq_loss", vq_loss, on_step=False, on_epoch=True,
            logger=True, sync_dist=True, batch_size=target.numel(),
        )
        self.log(
            "val_vq_enabled", float(self.use_vq_at_epoch(self.current_epoch)),
            on_step=False, on_epoch=True, logger=True, sync_dist=True,
            batch_size=target.numel(),
        )
        return loss

    def _reset_vq_switch_accumulators(self):
        self._vq_switch_code_counts = None
        self._vq_switch_day_assignments = 0
        self._vq_switch_distortion_sum = 0.0
        self._vq_switch_distortion_count = 0
        self._vq_switch_prediction_abs_sum = 0.0
        self._vq_switch_prediction_sq_sum = 0.0
        self._vq_switch_prediction_count = 0
        self._vq_switch_prediction_max_abs = 0.0

    def on_validation_epoch_start(self):
        if self.current_epoch == self.warmup_epochs:
            self._reset_vq_switch_accumulators()

    def _accumulate_vq_switch(self, vq_output, switch_output):
        indices = vq_output.indices.detach().reshape(-1).cpu()
        if indices.numel() == 0:
            return
        if not torch.equal(indices, indices[:1].expand_as(indices)):
            raise RuntimeError(
                "VQ switch diagnostics require one shared regime per trading day"
            )
        if self._vq_switch_code_counts is None:
            self._vq_switch_code_counts = torch.zeros(
                self.master.market_quantizer.codebook_size, dtype=torch.long
            )
        self._vq_switch_code_counts[int(indices[0].item())] += 1
        self._vq_switch_day_assignments += 1

        quantized = vq_output.quantized.detach()
        continuous = switch_output.market_state.detach()
        error = quantized - continuous
        self._vq_switch_distortion_sum += float(error.square().sum().cpu())
        self._vq_switch_distortion_count += error.numel()

        prediction_error = (
            switch_output.quantized_prediction.detach()
            - switch_output.continuous_prediction.detach()
        ).reshape(-1)
        absolute = prediction_error.abs()
        self._vq_switch_prediction_abs_sum += float(absolute.sum().cpu())
        self._vq_switch_prediction_sq_sum += float(
            prediction_error.square().sum().cpu()
        )
        self._vq_switch_prediction_count += prediction_error.numel()
        self._vq_switch_prediction_max_abs = max(
            self._vq_switch_prediction_max_abs,
            float(absolute.max().cpu()),
        )

    def on_validation_epoch_end(self):
        if (
            self.current_epoch != self.warmup_epochs
            or not self._vq_switch_day_assignments
        ):
            return
        counts = self._vq_switch_code_counts
        probabilities = counts.float() / counts.sum()
        nonzero = probabilities > 0
        perplexity = float(
            torch.exp(
                -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
            )
        )
        prediction_count = self._vq_switch_prediction_count
        diagnostics = {
            "epoch": int(self.current_epoch),
            "quantization_distortion": (
                self._vq_switch_distortion_sum
                / self._vq_switch_distortion_count
            ),
            "code_usage": counts.tolist(),
            "active_codes": int(torch.count_nonzero(counts)),
            "perplexity": perplexity,
            "day_assignments": self._vq_switch_day_assignments,
            "continuous_quantized_prediction_mae": (
                self._vq_switch_prediction_abs_sum / prediction_count
            ),
            "continuous_quantized_prediction_rmse": math.sqrt(
                self._vq_switch_prediction_sq_sum / prediction_count
            ),
            "continuous_quantized_prediction_max_abs": (
                self._vq_switch_prediction_max_abs
            ),
        }
        self.vq_switch_diagnostics = diagnostics
        for name in (
            "quantization_distortion",
            "active_codes",
            "perplexity",
            "continuous_quantized_prediction_mae",
            "continuous_quantized_prediction_rmse",
            "continuous_quantized_prediction_max_abs",
        ):
            self.log(
                f"vq_switch_{name}",
                float(diagnostics[name]),
                on_step=False,
                on_epoch=True,
                logger=False,
                sync_dist=False,
            )
        print("VQ switch diagnostics: " + json.dumps(diagnostics, sort_keys=True))

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
