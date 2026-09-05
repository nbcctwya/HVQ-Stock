import torch
import torch.nn as nn

from module.quantise import VectorQuantiser


class ResidualVectorQuantiser(nn.Module):
    """
    残差式层次化向量量化器（Hierarchical VQ / Residual VQ）。

    第 0 级量化输入 h，第 l 级量化前级残差 r = h - sum(z_q_j) (j < l)，
    最终 z_q = sum(z_q_l)，整体通过 STE 把梯度回传给 h。
    每一级复用现有的 VectorQuantiser（含各自的 dead-code 重初始化与 FeaturePool），
    总 vq_loss 为各级 loss 之和（各级内部已含 beta commitment + codebook loss，
    contras_loss 逐层按各自残差计算）。

    num_levels: 量化级数
    level_num_embed: 各级 codebook 大小的 list，长度须等于 num_levels
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_levels=2, level_num_embed=None, embed_dim=128, beta=0.25,
                 distance='cos', anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()

        if level_num_embed is None:
            level_num_embed = [256] * num_levels
        assert len(level_num_embed) == num_levels, \
            f"level_num_embed 长度 ({len(level_num_embed)}) 须等于 num_levels ({num_levels})"

        self.num_levels = num_levels
        self.level_num_embed = list(level_num_embed)
        self.embed_dim = embed_dim
        self.beta = beta

        self.levels = nn.ModuleList([
            VectorQuantiser(
                num_embed=num_embed,
                embed_dim=embed_dim,
                beta=beta,
                distance=distance,
                anchor=anchor,
                first_batch=first_batch,
                contras_loss=contras_loss
            )
            for num_embed in self.level_num_embed
        ])

    def forward(self, h_batch):
        # h_batch: (B, D) == (N_t, d)
        residual = h_batch
        z_q_sum = torch.zeros_like(h_batch)
        total_loss = 0.0
        perplexities, min_encodings_list, indices_list = [], [], []

        for level in self.levels:
            # 单层 VectorQuantiser 内部已对输入做 STE，返回值在数值上等于 z_q_vectors
            z_q_l, loss_l, (perplexity_l, min_encodings_l, encoding_indices_l) = level(residual)
            # 残差只按数值更新，梯度由各级自身的 STE 处理
            residual = residual - z_q_l.detach()
            z_q_sum = z_q_sum + z_q_l.detach()
            total_loss = total_loss + loss_l
            perplexities.append(perplexity_l)
            min_encodings_list.append(min_encodings_l)
            indices_list.append(encoding_indices_l)

        # 整体 STE：数值取 sum(z_q_l)，梯度回传到 h_batch
        z_q_output = h_batch + (z_q_sum - h_batch).detach()

        return z_q_output, total_loss, (perplexities, min_encodings_list, indices_list)

    def forward_per_level(self, h_batch):
        """按与 forward 完全相同的残差链运行各级量化，返回按级的量化输出。

        返回 (z_q_levels, total_loss, (perplexities, min_encodings_list, indices_list))，
        其中 z_q_levels 是长度为 num_levels 的 list，z_q_levels[l] 为第 l 级的
        量化输出（各级 VectorQuantiser 内部 STE 保证梯度可回传到 h_batch）。
        z_q_levels 满足 sum(z_q_levels) 在数值上等于 forward(h_batch) 的 z_q。
        用于 learnable-z1-scale 实验：Stage 2 自行按 z0 + alpha * z1 融合，
        不改变 forward 的默认 z0 + z1 行为。
        """
        residual = h_batch
        z_q_levels = []
        total_loss = 0.0
        perplexities, min_encodings_list, indices_list = [], [], []

        for level in self.levels:
            z_q_l, loss_l, (perplexity_l, min_encodings_l, encoding_indices_l) = level(residual)
            residual = residual - z_q_l.detach()
            z_q_levels.append(z_q_l)
            total_loss = total_loss + loss_l
            perplexities.append(perplexity_l)
            min_encodings_list.append(min_encodings_l)
            indices_list.append(encoding_indices_l)

        return z_q_levels, total_loss, (perplexities, min_encodings_list, indices_list)


def create_quantizer(vqvae_cfg):
    """
    按 vqvae.quantizer.type 构建量化器：
    'hvq' -> ResidualVectorQuantiser，其余（默认 'single'）-> 原 VectorQuantiser。
    Stage 1 (VQVAE) 与 Stage 2 (GenerateReturn) 共用，保证参数命名结构一致。
    """
    quantizer_cfg = vqvae_cfg['quantizer']
    common_kwargs = dict(
        beta=quantizer_cfg['commit_weight'],
        distance=quantizer_cfg['distance'],
        anchor=quantizer_cfg['anchor'],
        first_batch=quantizer_cfg['first_batch'],
        contras_loss=quantizer_cfg['contras_loss']
    )

    if quantizer_cfg.get('type', 'single') == 'hvq':
        return ResidualVectorQuantiser(
            num_levels=quantizer_cfg.get('num_levels', 2),
            level_num_embed=quantizer_cfg.get('level_num_embed'),
            embed_dim=vqvae_cfg['vq_embed_dim'],
            **common_kwargs
        )

    return VectorQuantiser(
        num_embed=vqvae_cfg['num_embed'],
        embed_dim=vqvae_cfg['vq_embed_dim'],
        **common_kwargs
    )
