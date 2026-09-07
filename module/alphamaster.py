"""AlphaMaster with a discrete historical-market decoder adapter.

The original current-market feature gate and complete stock backbone are kept
intact. A lightweight GRU encodes the previous 19 market observations, a
EMA vector quantizer maps the continuous state to a reusable regime, and
a zero-initialized linear hypernetwork adds a decoder-weight residual.
"""

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList(
            [Dropout(p=dropout) for _ in range(nhead)]
        )

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            attention = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)) / self.temperature,
                dim=-1,
            )
            if self.attn_dropout:
                attention = self.attn_dropout[i](attention)
            att_output.append(torch.matmul(attention, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


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
            self.attn_dropout = nn.ModuleList(
                [Dropout(p=dropout) for _ in range(nhead)]
            )

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            attention = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)), dim=-1
            )
            if self.attn_dropout:
                attention = self.attn_dropout[i](attention)
            att_output.append(torch.matmul(attention, vh))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output


class TemporalMarketEncoder(nn.Module):
    """Encode a ``[N,19,63]`` market history as one continuous state."""

    def __init__(
        self,
        input_size=63,
        hidden_size=63,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )

    def forward(self, market_history):
        _, last_hidden = self.gru(market_history)
        return last_hidden[-1]


class VectorQuantizerOutput(NamedTuple):
    quantized: torch.Tensor
    loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    indices: torch.Tensor


class EMAVectorQuantizer(nn.Module):
    """L2 VQ whose EMA statistics count one shared state per trading day."""

    def __init__(
        self,
        codebook_size=8,
        embedding_dim=63,
        commitment_weight=0.25,
        decay=0.99,
        statistics_level="trading_day",
    ):
        super().__init__()
        if codebook_size <= 0 or embedding_dim <= 0:
            raise ValueError("VQ codebook size and embedding dimension must be positive")
        if commitment_weight < 0:
            raise ValueError("VQ commitment weight must be non-negative")
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        if statistics_level != "trading_day":
            raise ValueError("EMA statistics must be updated at trading-day level")
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.statistics_level = statistics_level
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )
        self.embedding.weight.requires_grad_(False)
        # One initial pseudo-observation per code keeps unused prototypes fixed
        # and avoids unstable normalization during the first EMA updates.
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer(
            "ema_embedding_sum", self.embedding.weight.detach().clone()
        )

    @torch.no_grad()
    def _update_codebook(self, flat_inputs, flat_indices):
        assignments = F.one_hot(
            flat_indices, num_classes=self.codebook_size
        ).to(dtype=flat_inputs.dtype)
        batch_cluster_size = assignments.sum(dim=0)
        batch_embedding_sum = assignments.t() @ flat_inputs.detach()
        self.ema_cluster_size.mul_(self.decay).add_(
            batch_cluster_size, alpha=1.0 - self.decay
        )
        self.ema_embedding_sum.mul_(self.decay).add_(
            batch_embedding_sum, alpha=1.0 - self.decay
        )
        self.embedding.weight.copy_(
            self.ema_embedding_sum / self.ema_cluster_size.unsqueeze(1)
        )

    def forward(self, inputs):
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected VQ inputs with width {self.embedding_dim}, "
                f"got {inputs.shape[-1]}"
            )
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        codebook = self.embedding.weight
        distances = (
            flat_inputs.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat_inputs @ codebook.t()
        )
        flat_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(flat_indices).view_as(inputs)

        # Keep the 012 VQ-loss scalar unchanged for logging, validation, and
        # checkpoint selection. Detaching this term ensures the codebook is
        # updated exclusively by EMA rather than ordinary gradients.
        codebook_loss = F.mse_loss(quantized.detach(), inputs.detach())
        commitment_loss = F.mse_loss(inputs, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss
        # Forward values are exactly the selected embeddings, while gradients
        # to the encoder are the identity. This equivalent STE form also makes
        # every member of one cross-section bitwise-share its regime vector.
        quantized_st = quantized.detach() + (inputs - inputs.detach())
        indices = flat_indices.view(inputs.shape[:-1])
        if self.training:
            # DailyBatchSamplerRandom supplies exactly one complete trading-day
            # cross-section per forward. Its rows share the same historical
            # market state, so the first row is the day's single EMA
            # observation; cross-section size must not weight the prototype.
            self._update_codebook(flat_inputs[:1], flat_indices[:1])
        return VectorQuantizerOutput(
            quantized_st,
            loss,
            codebook_loss,
            commitment_loss,
            indices,
        )


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        return torch.matmul(lam, z).squeeze(1)


class MASTER(nn.Module):
    def __init__(
        self,
        d_feat=158,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        T_dropout_rate=0.5,
        S_dropout_rate=0.5,
        gate_input_start_index=158,
        gate_input_end_index=221,
        beta=None,
        market_encoder_input_size=63,
        market_encoder_hidden_size=63,
        market_encoder_num_layers=1,
        market_encoder_dropout=0.0,
        market_vq_codebook_size=8,
        market_vq_embedding_dim=63,
        market_vq_commitment_weight=0.25,
        market_vq_decay=0.99,
        market_vq_statistics_level="trading_day",
        market_adapter_output_size=256,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.x2y = nn.Linear(d_feat, d_model)
        self.pe = PositionalEncoding(d_model)
        self.tatten = TAttention(
            d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate
        )
        self.satten = SAttention(
            d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate
        )
        self.temporalatten = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

        if market_encoder_input_size != self.d_gate_input:
            raise ValueError("Market encoder input must match market width")
        if market_encoder_hidden_size != self.d_gate_input:
            raise ValueError("Market encoder state must be 63-dimensional")
        if market_adapter_output_size != d_model:
            raise ValueError("Market adapter output must match decoder width")
        self.market_encoder = TemporalMarketEncoder(
            input_size=market_encoder_input_size,
            hidden_size=market_encoder_hidden_size,
            num_layers=market_encoder_num_layers,
            dropout=market_encoder_dropout,
        )
        self.market_adapter = nn.Linear(
            market_encoder_hidden_size,
            market_adapter_output_size,
            bias=False,
        )
        nn.init.zeros_(self.market_adapter.weight)
        # Construct VQ after all 011 modules so their seeded initialization is
        # unchanged; only the formal forward graph gains this new module.
        if market_vq_embedding_dim != market_encoder_hidden_size:
            raise ValueError("VQ embedding dimension must match market state width")
        self.market_quantizer = EMAVectorQuantizer(
            codebook_size=market_vq_codebook_size,
            embedding_dim=market_vq_embedding_dim,
            commitment_weight=market_vq_commitment_weight,
            decay=market_vq_decay,
            statistics_level=market_vq_statistics_level,
        )

    def forward(self, x, return_vq_output=False):
        src = x[:, :, :self.gate_input_start_index]
        gate_input = x[
            :, -1, self.gate_input_start_index:self.gate_input_end_index
        ]
        market_history = x[
            :, :-1, self.gate_input_start_index:self.gate_input_end_index
        ]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        x = self.x2y(src)
        x = self.pe(x)
        x = self.tatten(x)
        x = self.satten(x)
        h = self.temporalatten(x)
        market_state = self.market_encoder(market_history)
        vq_output = self.market_quantizer(market_state)
        delta_weight = self.market_adapter(vq_output.quantized)
        y_base = self.decoder(h).squeeze(-1)
        y_market = torch.sum(delta_weight * h, dim=-1)
        prediction = y_base + y_market
        if return_vq_output:
            return prediction, vq_output
        return prediction
