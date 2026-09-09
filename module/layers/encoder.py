import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class FeatureTransform(nn.Module):
    """Original pre-GRU ``Linear -> LayerNorm -> LeakyReLU`` transform."""

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


class TemporalAttentionEncoder(nn.Module):
    """Input projection and AlphaMaster temporal-attention stack."""

    def __init__(self, d_feat, d_model, nhead, dropout):
        super().__init__()
        self.input_projection = nn.Linear(d_feat, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.temporal_attention = TAttention(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.temporal_aggregation = TemporalAttention(d_model=d_model)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.temporal_attention(x)
        return self.temporal_aggregation(x)

class CrossAssetTransformerEncoder(nn.Module):
    """Transformer encoder that models cross-asset relationships."""
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_layers,
                 d_out,
                 dim_feedforward=None):

        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * embed_dim

        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim))  # kept for state-dict compatibility

            def forward(self, x):
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
                x = x / rms * self.weight
                return x

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )
        
        # Swap default LayerNorm for RMSNorm.
        encoder_layer.norm1 = RMSNorm(embed_dim)
        encoder_layer.norm2 = RMSNorm(embed_dim)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, d_out)
        )


    def forward(self, temporal_summaries):
        # Expect (N_t, embed_dim) or (B, N_t, embed_dim). Treat (N_t, embed_dim)
        # as Batch=1, Seq=N_t. Variable N_t would require padding/masking.
        if temporal_summaries.dim() == 2:
            temporal_summaries = temporal_summaries.unsqueeze(0)  # (1, N_t, embed_dim)

        refined_representation = self.transformer_encoder(temporal_summaries)

        if refined_representation.shape[0] == 1:
            refined_representation = refined_representation.squeeze(0)
        out = self.out_layer(refined_representation)  # project to vq_dim (factor dim)
        return out


class SpatialEncoder(nn.Module):
    """Stage 1 temporal attention followed by the original asset Transformer."""
    def __init__(self,
                 input_features_C,
                 T_window,
                 gru_hidden_size,
                 num_transformer_heads,
                 num_transformer_layers,
                 final_embed_dim_d,
                 encoder_type="temporal-attention",
                 temporal_dropout=0.1
                 ):
        super().__init__()
        if encoder_type != "temporal-attention":
            raise ValueError(
                "Experiment 028 requires vqvae.encoder.type='temporal-attention'"
            )
        self.T_window = T_window
        self.C = input_features_C
        self.gru_hidden_size = gru_hidden_size

        # Original pre-GRU feature transform is unchanged.
        self.feature_transform = FeatureTransform(input_features_C)

        # Replaces only the original GRU temporal summarization.
        self.temporal_encoder = TemporalAttentionEncoder(
            d_feat=input_features_C,
            d_model=gru_hidden_size,
            nhead=num_transformer_heads,
            dropout=temporal_dropout,
        )

        # Original cross-asset Transformer remains unchanged.
        self.cross_asset_transformer = CrossAssetTransformerEncoder(embed_dim=gru_hidden_size,
                                                                    num_heads=num_transformer_heads,
                                                                    num_layers=num_transformer_layers,
                                                                    d_out=final_embed_dim_d)


    def forward(self, x_batch):
        # x_batch: (N_t, T_window, C) — N_t stocks at timestep t.
        transformed = self.feature_transform(x_batch)

        # Per-stock attention summary -> (N_t, gru_hidden_size).
        temporal_summaries = self.temporal_encoder(transformed)

        # Step 2: cross-asset Transformer -> (N_t, final_embed_dim_d).
        refined_representation = self.cross_asset_transformer(temporal_summaries)

        return refined_representation
