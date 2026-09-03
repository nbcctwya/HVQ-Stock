import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from module.quantise import VectorQuantiser
from module.quantise_hvq import ResidualVectorQuantiser, create_quantizer


def _make_hvq(num_levels=2, level_num_embed=(64, 32), embed_dim=128, **overrides):
    kwargs = dict(
        num_levels=num_levels,
        level_num_embed=list(level_num_embed),
        embed_dim=embed_dim,
        beta=0.25,
        distance='l2',
        anchor='probrandom',
        first_batch=False,
        contras_loss=False,
    )
    kwargs.update(overrides)
    return ResidualVectorQuantiser(**kwargs)


def _check_hvq_forward(test_case, hvq, h):
    z_q, loss, (perplexities, min_encodings_list, indices_list) = hvq(h)

    test_case.assertEqual(z_q.shape, h.shape)
    test_case.assertTrue(torch.is_tensor(loss))
    test_case.assertEqual(loss.dim(), 0)
    test_case.assertTrue(torch.isfinite(loss).item())

    test_case.assertEqual(len(perplexities), hvq.num_levels)
    test_case.assertEqual(len(min_encodings_list), hvq.num_levels)
    test_case.assertEqual(len(indices_list), hvq.num_levels)
    for level_idx, indices in enumerate(indices_list):
        test_case.assertEqual(indices.shape, (h.shape[0],))
        test_case.assertGreaterEqual(int(indices.min()), 0)
        test_case.assertLess(int(indices.max()), hvq.level_num_embed[level_idx])

    return z_q, loss


class ResidualVectorQuantiserTests(unittest.TestCase):
    def test_forward_train_mode(self):
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.train()
        h = torch.randn(32, 128, requires_grad=True)

        z_q, loss = _check_hvq_forward(self, hvq, h)

        # STE: z_q 对输入的梯度应能回传
        z_q.sum().backward()
        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(h.grad).all().item())
        self.assertTrue(torch.allclose(h.grad, torch.ones_like(h)))

    def test_forward_eval_mode(self):
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128, requires_grad=True)

        z_q, loss = _check_hvq_forward(self, hvq, h)

        (z_q.sum() + loss).backward()
        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(h.grad).all().item())

    def test_residual_reconstruction(self):
        # z_q 数值上应等于各级量化值之和（整体 STE 只改梯度不改数值）
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q, _, _ = hvq(h)
        # 直接逐级重算：第 l 级输入为 h 减去前级量化值
        with torch.no_grad():
            residual = h.clone()
            expected = torch.zeros_like(h)
            for level in hvq.levels:
                _, _, (_, _, idx_l) = level(residual)
                expected = expected + level.embedding.weight[idx_l]
                residual = residual - level.embedding.weight[idx_l]
        self.assertTrue(torch.allclose(z_q, expected, atol=1e-6))

    def test_three_levels_with_distinct_codebook_sizes(self):
        torch.manual_seed(0)
        hvq = _make_hvq(num_levels=3, level_num_embed=(64, 32, 16))
        hvq.train()
        h = torch.randn(32, 128)
        _, _, (_, _, indices_list) = hvq(h)
        self.assertEqual(len(indices_list), 3)
        for level_idx, limit in enumerate((64, 32, 16)):
            self.assertLess(int(indices_list[level_idx].max()), limit)

    def test_each_level_has_own_pool_and_codebook(self):
        hvq = _make_hvq()
        self.assertEqual(len(hvq.levels), hvq.num_levels)
        pools = [level.pool for level in hvq.levels]
        for level, num_embed in zip(hvq.levels, hvq.level_num_embed):
            self.assertEqual(level.embedding.weight.shape, (num_embed, hvq.embed_dim))
            # 每一级拥有独立的 FeaturePool 实例
            self.assertEqual(pools.count(level.pool), 1)


class SingleVectorQuantiserRegressionTests(unittest.TestCase):
    """type='single' 路径回归：原 VectorQuantiser 接口不变。"""

    def test_single_quantiser_interface(self):
        torch.manual_seed(0)
        vq = VectorQuantiser(num_embed=64, embed_dim=128, beta=0.25, distance='l2',
                             anchor='probrandom', first_batch=False, contras_loss=False)
        vq.train()
        h = torch.randn(32, 128, requires_grad=True)
        z_q, loss, (perplexity, min_encodings, encoding_indices) = vq(h)

        self.assertEqual(z_q.shape, h.shape)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.is_tensor(perplexity))
        self.assertEqual(encoding_indices.shape, (32,))
        self.assertLess(int(encoding_indices.max()), 64)

        z_q.sum().backward()
        self.assertIsNotNone(h.grad)


class QuantizerFactoryTests(unittest.TestCase):
    """create_quantizer 按 vqvae.quantizer.type 切换量化器实现。"""

    @staticmethod
    def _make_cfg(quantizer_overrides=None):
        quantizer_cfg = {
            'decay': 0.95,
            'commit_weight': 0.25,
            'distance': 'l2',
            'anchor': 'probrandom',
            'first_batch': False,
            'contras_loss': True,
        }
        if quantizer_overrides:
            quantizer_cfg.update(quantizer_overrides)
        return {
            'vqvae': {
                'num_embed': 512,
                'vq_embed_dim': 128,
                'quantizer': quantizer_cfg,
            }
        }

    def test_default_type_is_single(self):
        cfg = self._make_cfg()
        quantizer = create_quantizer(cfg['vqvae'])
        self.assertIsInstance(quantizer, VectorQuantiser)
        self.assertNotIsInstance(quantizer, ResidualVectorQuantiser)
        self.assertEqual(quantizer.num_embed, 512)

    def test_explicit_single_type(self):
        cfg = self._make_cfg({'type': 'single'})
        quantizer = create_quantizer(cfg['vqvae'])
        self.assertIsInstance(quantizer, VectorQuantiser)
        self.assertNotIsInstance(quantizer, ResidualVectorQuantiser)

    def test_hvq_type(self):
        cfg = self._make_cfg({'type': 'hvq', 'num_levels': 2, 'level_num_embed': [256, 256]})
        quantizer = create_quantizer(cfg['vqvae'])
        self.assertIsInstance(quantizer, ResidualVectorQuantiser)
        self.assertEqual(quantizer.num_levels, 2)
        self.assertEqual(quantizer.level_num_embed, [256, 256])
        # 各级 codebook 大小与 embed_dim 生效
        for level in quantizer.levels:
            self.assertEqual(level.embedding.weight.shape, (256, 128))

    def test_hvq_forward_smoke(self):
        cfg = self._make_cfg({'type': 'hvq', 'num_levels': 2, 'level_num_embed': [64, 32]})
        quantizer = create_quantizer(cfg['vqvae'])
        quantizer.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q, loss, aux = quantizer(h)
        self.assertEqual(z_q.shape, h.shape)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(len(aux[2]), 2)


if __name__ == '__main__':
    unittest.main()
