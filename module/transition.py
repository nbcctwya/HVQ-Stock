"""VQ transition-aware routing branch for experiment 024.

Encodes the recent discrete latent-state *transition* trajectory of the
current instrument and maps it to an additive bias on the MoE clean routing
logits:

    e_k        = frozen_stage1_codebook[k].detach()
    delta_z_j  = e_{k_j} - e_{k_{j-1}}          (consecutive prototype diff)
    state      = GRU(delta_z sequence)           (single layer, no attention)
    bias       = W_t(state)                      (zero-initialized Linear)

Only adjacent prototype *differences* are encoded — the absolute code
identity of the current state is already carried by ``z_q`` on the original
routing path, so this branch never re-encodes absolute prototypes or code
ids as a separate prediction signal.  Padding is excluded via packed
sequences; a sample with no valid history yields a strictly zero state.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class VQTransitionEncoder(nn.Module):
    """GRU over consecutive frozen-codebook prototype differences.

    Args:
        codebook_weight: the frozen Stage 1 ``quantizer.embedding.weight``
            (kept by reference; always detached on read, never updated here).
        num_experts: MoE expert count; the output bias dimension.
        history_len: maximum number of history codes (L - 1 = 4), hence the
            maximum number of encoded transitions.
        gru_hidden: GRU hidden size (64).
    """

    def __init__(self, codebook_weight, num_experts, history_len=4,
                 gru_hidden=64):
        super().__init__()
        if codebook_weight.ndim != 2:
            raise ValueError(
                f"codebook_weight must be [num_embed, embed_dim], got "
                f"{tuple(codebook_weight.shape)}"
            )
        if not isinstance(history_len, int) or history_len < 1:
            raise ValueError(f"history_len must be a positive int, got {history_len}")
        self.num_codes, self.embed_dim = codebook_weight.shape
        self.history_len = history_len
        self.gru_hidden = gru_hidden
        self.num_experts = num_experts
        # Plain reference via __dict__ on purpose: assigning an nn.Parameter
        # through nn.Module.__setattr__ would register it as a parameter of
        # this branch (trainable + serialized); the codebook must remain
        # owned solely by the frozen Stage 1 quantizer.  The Parameter is
        # moved in-place with the quantizer on .to(device) and is always
        # detached before lookup.
        self.__dict__["codebook_weight"] = codebook_weight

        # RNG isolation: constructing this branch must not advance the RNG
        # stream seen by any baseline module initialized afterwards, so that
        # under the same seed all pre-existing parameters stay bitwise
        # identical to the base experiment.
        with torch.random.fork_rng(devices=[]):
            self.gru = nn.GRU(
                input_size=self.embed_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
            )
            self.proj = nn.Linear(gru_hidden, num_experts)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def transition_state(self, hist_codes, hist_len, current_code):
        """Encode the transition trajectory into a [B, gru_hidden] state.

        ``hist_codes`` [B, history_len] holds the up to ``hist_len`` most
        recent strictly-earlier codes (oldest first); ``current_code`` [B]
        is the live quantizer output of the current forward.  Rows with
        ``hist_len == 0`` return a strictly zero state.
        """
        if hist_codes.ndim != 2 or hist_codes.shape[1] != self.history_len:
            raise ValueError(
                f"hist_codes must be [B, {self.history_len}], got "
                f"{tuple(hist_codes.shape)}"
            )
        batch = current_code.shape[0]
        if hist_codes.shape[0] != batch or hist_len.shape != (batch,):
            raise ValueError(
                "hist_codes/hist_len/current_code batch sizes must match"
            )
        codes = torch.cat(
            [hist_codes.long(), current_code.long().unsqueeze(1)], dim=1
        )
        if torch.any(codes < 0) or torch.any(codes >= self.num_codes):
            raise ValueError("code id out of range for the frozen codebook")
        hist_len = hist_len.long().clamp(min=0, max=self.history_len)

        codebook = self.codebook_weight.detach()
        protos = codebook[codes]                    # (B, 1 + history_len, D)
        deltas = protos[:, 1:] - protos[:, :-1]     # (B, history_len, D)
        # Slot j encodes the transition k_j -> k_{j+1}.  When the history is
        # shorter than history_len, the last valid transition is
        # (last history code) -> (current code), which is not the adjacent
        # pair in the padded chain; compute it explicitly.
        current_delta = protos[:, -1] - protos.gather(
            1,
            (hist_len - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.embed_dim),
        ).squeeze(1)
        slots = torch.arange(self.history_len, device=codes.device)
        valid = slots.unsqueeze(0) < hist_len.unsqueeze(1)
        is_last = slots.unsqueeze(0) == (hist_len - 1).clamp(min=0).unsqueeze(1)
        replace = (valid & is_last).unsqueeze(-1)
        deltas = torch.where(replace, current_delta.unsqueeze(1), deltas)
        deltas = deltas * valid.unsqueeze(-1).to(deltas.dtype)

        # Packed sequences keep padding out of the GRU entirely; length-0
        # rows are clamped to a dummy zero step and zeroed afterwards.
        lengths = hist_len.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            deltas, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        state = hidden.squeeze(0)                   # (B, gru_hidden)
        has_history = (hist_len > 0).unsqueeze(1).to(state.dtype)
        return state * has_history

    def forward(self, hist_codes, hist_len, current_code):
        """Return the additive routing bias ``W_t(state)``, shape [B, E]."""
        return self.proj(
            self.transition_state(hist_codes, hist_len, current_code)
        )
