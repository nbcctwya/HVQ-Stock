"""Stage 1 encoder layers for experiment 030.

The four attention components below are copied from
``AlphaMaster/src/alphamaster/model.py``. Their computations are deliberately
kept unchanged; only the surrounding VQ-VAE interface is adapted here.
"""

import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class FeatureTransform(nn.Module):
    """The original pre-GRU ``Linear -> LayerNorm -> LeakyReLU`` block."""

    def __init__(self, n_feature):
        super().__init__()
        self.n_feature = n_feature
        self.normalize = nn.LayerNorm(n_feature)
        self.linear = nn.Linear(n_feature, n_feature)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.normalize(x)
        return self.leakyrelu(x)


# Copied from AlphaMaster/src/alphamaster/model.py; no algorithmic changes.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


# Copied from AlphaMaster/src/alphamaster/model.py; no algorithmic changes.
class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


# Copied from AlphaMaster/src/alphamaster/model.py; no algorithmic changes.
class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


# Copied from AlphaMaster/src/alphamaster/model.py; no algorithmic changes.
class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTERStyleEncoder(nn.Module):
    """MASTER attention stack with parameter-free temporal mean pooling."""

    def __init__(self, d_feat, d_model, t_nhead, s_nhead,
                 t_dropout_rate, s_dropout_rate):
        super().__init__()
        self.x2y = nn.Linear(d_feat, d_model)
        self.pe = PositionalEncoding(d_model)
        self.tatten = TAttention(
            d_model=d_model, nhead=t_nhead, dropout=t_dropout_rate
        )
        self.satten = SAttention(
            d_model=d_model, nhead=s_nhead, dropout=s_dropout_rate
        )

    def forward(self, x):
        x = self.x2y(x)
        x = self.pe(x)
        x = self.tatten(x)
        x = self.satten(x)
        return x.mean(dim=1)


class SpatialEncoder(nn.Module):
    """Stage 1: preserved feature transform + MASTER-style encoder + MLP."""

    def __init__(self,
                 input_features_C,
                 T_window,
                 gru_hidden_size,
                 num_transformer_heads,
                 num_transformer_layers,
                 final_embed_dim_d,
                 encoder_type="master",
                 temporal_num_heads=None,
                 spatial_num_heads=None,
                 temporal_dropout=0.1,
                 spatial_dropout=0.1):
        super().__init__()
        if encoder_type != "master":
            raise ValueError(
                "Experiment 027 requires vqvae.encoder.type='master'"
            )
        if num_transformer_layers != 1:
            raise ValueError(
                "MASTER-style Stage 1 uses exactly one TAttention and one "
                "SAttention block"
            )

        temporal_num_heads = temporal_num_heads or num_transformer_heads
        spatial_num_heads = spatial_num_heads or num_transformer_heads
        self.T_window = T_window
        self.C = input_features_C
        self.hidden_size = gru_hidden_size

        # Preserved verbatim from the original pre-GRU feature path.
        self.feature_transform = FeatureTransform(input_features_C)

        # Replaces only GRU summarization + CrossAssetTransformer.
        self.master_encoder = MASTERStyleEncoder(
            d_feat=input_features_C,
            d_model=gru_hidden_size,
            t_nhead=temporal_num_heads,
            s_nhead=spatial_num_heads,
            t_dropout_rate=temporal_dropout,
            s_dropout_rate=spatial_dropout,
        )

        # Preserved from CrossAssetTransformerEncoder.out_layer.
        self.out_layer = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size * 4),
            nn.GELU(),
            nn.Linear(gru_hidden_size * 4, final_embed_dim_d)
        )

    def forward(self, x_batch):
        # x_batch: (N_t, T_window, C), one day's cross-section.
        transformed = self.feature_transform(x_batch)
        encoded = self.master_encoder(transformed)
        return self.out_layer(encoded)
