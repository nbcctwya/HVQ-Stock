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

    def forward_level0(self, h_batch):
        """只运行第 0 级量化，返回 (z_q0, loss_0, ([ppl_0], [min_enc_0], [idx_0]))。

        z_q0 数值上等于第 0 级 codebook 向量（该级 VectorQuantiser 内部 STE 保证
        梯度可回传到 h_batch），返回结构与 forward 同构，只是各级统计量为
        长度 1 的 list。用于 z0-only 消融：Stage 2 只使用第一级量化表示。
        不改变 forward 的默认 z0 + z1 行为。
        """
        z_q0, loss_0, (perplexity_0, min_encodings_0, indices_0) = self.levels[0](h_batch)
        return z_q0, loss_0, ([perplexity_0], [min_encodings_0], [indices_0])

    def forward_two_levels(self, h_batch):
        """分别返回两级量化输出 (z_q0, z_q1)，供 prediction-residual 消融使用。

        z_q0 为第 0 级量化输出（与 forward_level0 一致）；z_q1 为第 1 级对残差
        h - z_q0 的量化输出。逐级 STE 语义与 forward 完全相同，且
        forward(h_batch) 的 z_q 数值上等于 z_q0 + z_q1。
        不改变 forward 的默认 z0 + z1 行为；仅用于 num_levels == 2。
        """
        assert self.num_levels == 2, "forward_two_levels 仅支持 num_levels == 2"
        z_q0, loss_0, (perplexity_0, min_encodings_0, indices_0) = self.levels[0](h_batch)
        residual = h_batch - z_q0.detach()
        z_q1, loss_1, (perplexity_1, min_encodings_1, indices_1) = self.levels[1](residual)
        return (z_q0, z_q1, loss_0 + loss_1,
                ([perplexity_0, perplexity_1],
                 [min_encodings_0, min_encodings_1],
                 [indices_0, indices_1]))


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
