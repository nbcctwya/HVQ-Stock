import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from module.bidirectional import LoadingGenerator
from utils import get_root_dir, calc_ic
from utils.rankloss import RankLoss
from module.quantise import VectorQuantiser
from module.quantise_hvq import ResidualVectorQuantiser, create_quantizer
from module.layers.encoder import SpatialEncoder
from module.layers.decoder import ReconstructionDecoder
from module.layers.src import RevIN
from utils import corr_cluster_order
from torch.optim.lr_scheduler import LambdaLR
import math
from module.layers.src import ListNetLoss

def softcap_log1p(x, c):
    x = F.relu(x)
    return c * torch.log1p(x / c)

class LatentValueHead(nn.Module):
    def __init__(self, d_latent, K):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_latent, K),
            nn.LayerNorm(K),
            nn.GELU(),
            nn.Linear(K, K)
        )
    def forward(self, z_q):
        return self.head(z_q)    # (B, K)

def check_vq_idx(idx, K):
    if torch.any(idx >= K) or torch.any(idx < 0):
        bad = idx[(idx >= K) | (idx < 0)]
        raise ValueError(f"[VQ-IDX] out-of-range values: {bad.tolist()} (K_latent={K})")
    
class GenerateReturn(pl.LightningModule):
    def __init__(self,
                 config,
                 T_max,
                 ):
        super().__init__()
        self.config = config

        vqvae_cfg = config['vqvae']

        # VQVAE
        self.num_prior_factors = vqvae_cfg['num_prior_factors']  # P
        self.num_embed     = vqvae_cfg['num_embed']      # K (codebook size)
        self.num_features  = vqvae_cfg['num_features']   # C
        self.hidden_size   = vqvae_cfg['hidden_size']    # H (GRU output dim)
        self.vq_embed_dim  = vqvae_cfg['vq_embed_dim']   # d (VQ / encoder output dim)
        self.seq_len       = vqvae_cfg['seq_len']        # T_window for reconstruction
        self.aux_imp       = config['predictor']['aux_imp']

        # Quantizer
        self.decay         = vqvae_cfg['quantizer']['decay']
        self.commit_weight = vqvae_cfg['quantizer']['commit_weight']  # beta

        # Encoder
        self.transformer_heads = vqvae_cfg['encoder']['num_heads']
        self.transformer_layers = vqvae_cfg['encoder']['num_layers']

        # Decoder
        self.initial_T = vqvae_cfg['decoder']['initial_T']
        self.hidden_channels = vqvae_cfg['decoder']['hidden_channels']

        # 1. Encoder
        self.encoder = SpatialEncoder(
            input_features_C = self.num_features,
            T_window=self.seq_len,
            gru_hidden_size=self.hidden_size,
            num_transformer_heads=self.transformer_heads,
            num_transformer_layers=self.transformer_layers,
            final_embed_dim_d=self.vq_embed_dim
        )

        # 2. Vector Quantizer（按 quantizer.type 选择单层或残差式 HVQ）
        self.quantizer = create_quantizer(vqvae_cfg)

        # 003 消融开关：Stage 2 只使用 Residual HVQ 第一级量化输出 z0（而非 z0+z1）。
        # 默认 False 保持 001 的 z0+z1 行为。
        self.z0_only = config['predictor'].get('z0_only', False)
        if self.z0_only and not isinstance(self.quantizer, ResidualVectorQuantiser):
            raise ValueError(
                "predictor.z0_only=True 需要 hvq 量化器 (ResidualVectorQuantiser)，"
                "当前 quantizer.type 不是 'hvq'"
            )

        # 006 消融开关：在 003 的 z0-only 主预测路径（ŷ0 = F(z0)）之外，增加一个
        # 由 z1 驱动的 prediction-residual correction branch：
        #   Δŷ = G(z1) 专门学习残差目标 r = y - stopgrad(ŷ0)，最终 ŷ = ŷ0 + Δŷ。
        # z0 与 z1 在 Stage 2 中职责分离，绝不做 representation-level 的 z0+z1 融合。
        # 默认 False 保持 003 的纯 z0-only 行为。
        self.z1_residual_branch = config['predictor'].get('z1_residual_branch', False)
        if self.z1_residual_branch:
            if not isinstance(self.quantizer, ResidualVectorQuantiser):
                raise ValueError(
                    "predictor.z1_residual_branch=True 需要 hvq 量化器 "
                    "(ResidualVectorQuantiser)，当前 quantizer.type 不是 'hvq'"
                )
            if not self.z0_only:
                raise ValueError(
                    "predictor.z1_residual_branch=True 要求 predictor.z0_only=True："
                    "006 的 z0 主预测路径必须与 003 完全一致"
                )
            # 最小 residual prediction head：单层线性 z1(d) -> Δŷ(1)，
            # 不引入 MLP / attention / gate / market feature，只检验 z1 能否预测
            # z0 主路径尚未解释的 prediction residual。
            self.residual_head = nn.Linear(self.vq_embed_dim, 1)
        
        # 3. RevIN
        self.revin = RevIN(self.num_features)

        # # 4. Decoder
        # self.decoder = ReconstructionDecoder(
        #     latent_dim=self.vq_embed_dim,      # d
        #     prior_factor_dim=self.num_prior_factors, # P
        #     output_T=self.seq_len,             # T_window
        #     output_C=self.num_features,        # C
        #     initial_T=self.initial_T,
        #     hidden_channels=self.hidden_channels,
        #     norm_type=vqvae_cfg['decoder'].get('norm_type', 'none'),
        #     num_groups=vqvae_cfg['decoder'].get('num_groups', 8)
        # )

        self.saved_model = config['predictor']['saved_model']
        self.load_pretrained_vqvae(checkpoint_path=os.path.join('checkpoints', f"{self.saved_model}"))
        self.freeze_vqvae()

        # 4. Factor Loading
        self.z_prior_norm = nn.LayerNorm(self.num_prior_factors)
        self.loadings = LoadingGenerator(config)
        # LatentValueHead或许可魔改@@@   ***
        self.latent_value_head = LatentValueHead(
            d_latent=self.vq_embed_dim,
            K = self.vq_embed_dim
        )

        self.use_prior = config['predictor']['use_prior']
        # 5. Return Predictor
        self.return_predictor = ReturnPredictor(num_prior  = self.num_prior_factors, 
                                                num_latent = self.vq_embed_dim,
                                                use_prior = self.use_prior)

        self.n_features = config['vqvae']['num_features']
        self.n_prior_factors = config['vqvae']['num_prior_factors']

        self.T_max = T_max
        self.target_index = config['predictor']['target_day'] - 1 # ex. 5 -> 4 (start from 0)
        
        self.aux_weight = config['predictor']['aux_weight']

        self.ic = []
        self.ric = []
        # 006：z0 主路径 ŷ0 的验证 IC / RankIC（仅 z1_residual_branch 启用时使用）
        self.ic_main = []
        self.ric_main = []
        self.best_val_loss = float('inf')
        self.best_metrics_at_min_loss = {}
        self.rank = config['predictor']['rank']
        self.rank_loss = RankLoss(alpha=self.rank)
        self.listNet_loss = ListNetLoss(temperature=1.0)

    def configure_optimizers(self):
        optimizer  = torch.optim.AdamW(self.parameters(), lr=self.config['train']['learning_rate'], weight_decay=1e-5)
        # Linear warm-up over the first 5% of total steps, then cosine decay.
        total_steps = self.T_max
        warmup_steps = int(0.05 * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # scheduler  = CosineAnnealingLR(optimizer, T_max=self.T_max)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        sch_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [sch_config]
    
    def _get_data(self, batch, batch_idx):
        batch   = batch.squeeze(0)
        batch   = batch.float()
        feature = batch[:, :, 0:self.n_features] # (300, 20, 158)
        prior_factor = batch[:, -1, self.n_features : self.n_features+self.n_prior_factors] # (300, 13)
        future_returns = batch[:, -1, self.n_features+self.n_prior_factors  : ] # (300, 1, 10)
        future_returns = future_returns.squeeze(-1) # (300, 10)
        
        label = future_returns[:, self.target_index] # (300, 1)

        return feature, prior_factor, label
    
    def forward(self, feature, prior_factor, return_components=False):

        ####### STAGE 1: VQVAE #######
        feature_normalized = self.revin(feature, mode="norm")
        h_batch = self.encoder(feature_normalized)  # (B, H)
        z_q1 = None
        if self.z1_residual_branch:
            # 006：分别取两级量化输出。z0 走与 003 完全相同的主预测路径；
            # z1 只进入 residual correction branch，绝不与 z0 做 z0+z1 融合。
            z_q, z_q1, _, (_, min_encodings, vq_idx) = self.quantizer.forward_two_levels(h_batch)
        elif self.z0_only:
            # z0-only 消融：只取 Residual HVQ 第一级量化输出 z0（而非 z0+z1）
            z_q, _, (_, min_encodings, vq_idx) = self.quantizer.forward_level0(h_batch)
        else:
            z_q, _, (_, min_encodings, vq_idx) = self.quantizer(h_batch)
        z_q = z_q.detach()
        if z_q1 is not None:
            z_q1 = z_q1.detach()

        ####### STAGE 2: Loading Generator #######    --此处可改
        alpha, beta_p, beta_l, loss_imp = self.loadings(feature, z_q)
        prior_factor_normed = self.z_prior_norm(prior_factor)

        f_latent = self.latent_value_head(z_q)

        y_pred = self.return_predictor(
            alpha    = alpha,
            beta_p   = beta_p,
            beta_l   = beta_l,              # soft weights?
            f_prior  = prior_factor_normed, # (B,P)
            f_latent = f_latent,            # (B,K)
        )
        loss_imp = softcap_log1p(loss_imp, self.aux_imp)

        if self.z1_residual_branch:
            # y_pred 到这里为止就是 ŷ0 = F(z0)（与 003 的主路径输出完全一致）。
            y0 = y_pred
            delta_y = self.residual_head(z_q1).squeeze(-1)  # (B,)
            y_pred = y0 + delta_y                           # ŷ = ŷ0 + Δŷ
            if return_components:
                return {
                    'y_pred': y_pred, 'y0': y0, 'delta_y': delta_y,
                    'beta_p': beta_p, 'beta_l': beta_l,
                    'z_q': z_q, 'z_q1': z_q1, 'loss_imp': loss_imp,
                }
        return y_pred, beta_p, beta_l, z_q, loss_imp

    def _residual_branch_losses(self, y0, delta_y, label):
        """006 固定损失定义（无可调权重）：

        main_loss = L(ŷ0, y)               —— 与 003 的主路径损失完全相同；
        res_target = y - stopgrad(ŷ0)      —— stop-gradient 的 prediction residual；
        res_loss  = L(Δŷ, res_target)      —— 同一 RankLoss 应用于 residual 任务。

        res_target 已 detach，residual branch 的梯度不会经 target 回传到 ŷ0。
        """
        main_loss = self.rank_loss(y0, label)
        res_target = label - y0.detach()
        res_loss = self.rank_loss(delta_y, res_target)
        return main_loss, res_loss, res_target


    def training_step(self, batch, batch_idx):
        feature, prior_factor, label = self._get_data(batch, batch_idx)

        if self.z1_residual_branch:
            out = self.forward(feature, prior_factor, return_components=True)
            aux_loss = out['loss_imp']
            main_loss, res_loss, _ = self._residual_branch_losses(
                out['y0'], out['delta_y'], label)
            # 固定组合：主预测损失 + residual 损失 + aux，均无可调权重
            loss = main_loss + res_loss + self.aux_weight * aux_loss

            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_mse_loss', main_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_res_loss', res_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            return {"loss": loss}

        y_pred, beta_p, beta_l, z_q, aux_loss = self.forward(feature, prior_factor)

        mse_loss = self.rank_loss(y_pred, label)
        main_loss = mse_loss

        loss = main_loss + self.aux_weight * aux_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # self.log('train_rank_loss', rank_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        feature, prior_factor, label = self._get_data(batch, batch_idx)

        if self.z1_residual_branch:
            out = self.forward(feature, prior_factor, return_components=True)
            y_pred = out['y_pred']      # 正式预测 ŷ = ŷ0 + Δŷ
            aux_loss = out['loss_imp']
            main_loss, res_loss, _ = self._residual_branch_losses(
                out['y0'], out['delta_y'], label)
            loss = main_loss + res_loss + self.aux_weight * aux_loss

            self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_mse_loss', main_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_res_loss', res_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

            daily_ic, daily_ric = calc_ic(y_pred.cpu().numpy(), label.cpu().numpy())
            self.ic.append(daily_ic)
            self.ric.append(daily_ric)
            # 同步记录 z0 主路径 ŷ0 的 IC / RankIC，用于与最终 ŷ 对照
            daily_ic0, daily_ric0 = calc_ic(out['y0'].detach().cpu().numpy(), label.cpu().numpy())
            self.ic_main.append(daily_ic0)
            self.ric_main.append(daily_ric0)
            return {"loss": loss}

        feature, prior_factor, label = self._get_data(batch, batch_idx)
        y_pred, beta_p, beta_l, z_q, aux_loss = self.forward(feature, prior_factor)

        mse_loss = self.rank_loss(y_pred, label)
        main_loss = mse_loss

        loss = main_loss + self.aux_weight * aux_loss

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mse_loss', mse_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # self.log('val_rank_loss', rank_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        daily_ic, daily_ric = calc_ic(y_pred.cpu().numpy(), label.cpu().numpy())
        self.ic.append(daily_ic)
        self.ric.append(daily_ric)
        return {"loss": loss}
            
    def on_validation_epoch_end(self):
        current_ic = np.mean(self.ic)
        current_ric = np.mean(self.ric)
        current_icir = np.mean(self.ic) / np.std(self.ic) if np.std(self.ic) != 0 else 0
        current_ricir = np.mean(self.ric) / np.std(self.ric) if np.std(self.ric) != 0 else 0

        val_loss_epoch = self.trainer.callback_metrics.get('val_loss')
        
        self.log('Val_RIC', current_ric, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        other_metrics = {
            'Val_IC': current_ic,
            'Val_ICIR': current_icir,
            'Val_RICIR': current_ricir,
        }
        self.log_dict(other_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

        if self.z1_residual_branch and self.ic_main:
            # 006：z0 主路径 ŷ0 单独的验证指标，用于判断 z1 是否提供增量价值
            main_metrics = {
                'Val_IC_main': float(np.mean(self.ic_main)),
                'Val_RIC_main': float(np.mean(self.ric_main)),
            }
            self.log_dict(main_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.ic_main = []
        self.ric_main = []

        # Reset the IC and RIC lists
        self.ic = []
        self.ric = []

        if val_loss_epoch is not None and val_loss_epoch < self.best_val_loss:
            self.best_val_loss = val_loss_epoch
            self.best_metrics_at_min_loss = {
                'Best_Val_Loss': float(val_loss_epoch),
                'Best_Val_IC': current_ic,
                'Best_Val_ICIR': current_icir,
                'Best_Val_RIC': current_ric,
                'Best_Val_RICIR': current_ricir,
            }
            self.log_dict(self.best_metrics_at_min_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        if val_loss_epoch is not None:
            self.log('val_loss_epoch', val_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)


    def init_from_ckpt(self, path, ignore_keys=list()):
        """Load pretrained model weights."""
        sd = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if "state_dict" in sd:
            sd = sd["state_dict"]
        
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        
        self.load_state_dict(sd, strict=False)
        print(f"Model restored from {path}.")

    def load_pretrained_vqvae(self, checkpoint_path=None):
        """Load Encoder, Quantizer, and RevIN weights from a pretrained VQ-VAE checkpoint."""
        print(f"Loading pretrained VQ-VAE from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        revin_state_dict = {}
        encoder_state_dict = {}
        quantizer_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith('vqvae.spatial_encoder.'):
                encoder_state_dict[k.replace('vqvae.spatial_encoder.', '')] = v
            elif k.startswith('vqvae.quantizer.'):
                quantizer_state_dict[k.replace('vqvae.quantizer.', '')] = v
            elif k.startswith('vqvae.revin.'):
                revin_state_dict[k.replace('vqvae.revin.', '')] = v

        missing_encoder, unexpected_encoder = self.encoder.load_state_dict(encoder_state_dict, strict=True)
        missing_quantizer, unexpected_quantizer = self.quantizer.load_state_dict(quantizer_state_dict, strict=True)
        missing_revin, unexpected_revin = self.revin.load_state_dict(revin_state_dict, strict=True)
        print(f"--- Encoder loaded: missing={len(missing_encoder)}, unexpected={len(unexpected_encoder)}")
        print(f"--- Quantizer loaded: missing={len(missing_quantizer)}, unexpected={len(unexpected_quantizer)}")
        print(f"--- RevIN loaded: missing={len(missing_revin)}, unexpected={len(unexpected_revin)}")

    def train(self, mode=True):
        # Lightning 训练时会对整个模块递归调用 .train()，把冻结的 Stage 1
        # 子模块重新切回 training mode：VectorQuantiser 在 training mode 下会
        # 更新 embed_prob 并通过 .data 改写 codebook，encoder 中 Transformer
        # 的 dropout 也会让 frozen representation 在 Stage 2 训练期间保持
        # 随机。requires_grad=False 无法阻止这些 mode 副作用，因此这里强制
        # 预训练的 encoder / quantizer / revin 始终保持 eval。
        super().train(mode)
        self.encoder.eval()
        self.quantizer.eval()
        self.revin.eval()
        return self

    def freeze_vqvae(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.quantizer.parameters():
            param.requires_grad = False
        for param in self.revin.parameters():
            param.requires_grad = False
        print("== VQ-VAE weights frozen ==")

        self.encoder.eval()
        self.quantizer.eval()
        self.revin.eval()

class ReturnPredictor(nn.Module):
    def __init__(self, num_prior, num_latent, use_prior=True):
        super().__init__()
        self.num_prior = num_prior
        self.num_latent = num_latent
        self.use_prior = use_prior
        
    def forward(self, alpha, beta_p, beta_l, f_prior, f_latent):
        prior_term = (beta_p * f_prior).sum(dim=1)
        latent_term = (beta_l * f_latent).sum(dim=1)  # elementwise (B, K)

        combined = torch.cat([prior_term.unsqueeze(1), latent_term.unsqueeze(1)], dim=1)
        # intercept_term = self.final_layer(combined).squeeze(-1)
        
        if self.use_prior:  
            output = alpha + prior_term + latent_term # + intercept_term
        else:
            output = alpha + latent_term

        return output