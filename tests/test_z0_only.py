import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from module.quantise_hvq import ResidualVectorQuantiser


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


class ForwardLevel0Tests(unittest.TestCase):
    """z0-only 消融：ResidualVectorQuantiser.forward_level0 的行为契约。"""

    def test_z0_matches_level0_codebook_lookup(self):
        # forward_level0 的 z_q0 在数值上必须等于第 0 级 codebook 中
        # 按 encoding_indices 查到的向量
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q0, loss_0, (perplexities, min_encodings_list, indices_list) = hvq.forward_level0(h)
        self.assertEqual(z_q0.shape, h.shape)
        self.assertTrue(torch.isfinite(loss_0).item())
        self.assertEqual(len(indices_list), 1)
        expected = hvq.levels[0].embedding.weight[indices_list[0]]
        self.assertTrue(torch.allclose(z_q0, expected, atol=1e-6))

    def test_z0_excludes_second_level(self):
        # z_q0 必须严格等于第 0 级输出，且（一般情况下）不等于完整 forward 的 z0+z1
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q0, _, _ = hvq.forward_level0(h)
            z_q_full, _, (_, _, indices_full) = hvq(h)
        # 完整 forward 的第 0 级索引与 forward_level0 一致（同一输入、同一 codebook）
        with torch.no_grad():
            _, _, (_, _, indices_0) = hvq.forward_level0(h)
        self.assertTrue(torch.equal(indices_full[0], indices_0[0]))
        # z0+z1 一般不等于 z0（第二级残差量化确实引入了额外信息）
        self.assertFalse(torch.allclose(z_q0, z_q_full, atol=1e-6))
        # z_q_full - z_q0 必须等于第 1 级 codebook 向量
        z_q1 = hvq.levels[1].embedding.weight[indices_full[1]]
        self.assertTrue(torch.allclose(z_q_full, z_q0 + z_q1, atol=1e-6))

    def test_interface_matches_forward(self):
        # 返回结构与 forward 同构：(z_q, loss, (perplexities, min_encodings, indices))
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(16, 128)
        with torch.no_grad():
            z_q0, loss_0, (perplexities, min_encodings_list, indices_list) = hvq.forward_level0(h)
        self.assertEqual(z_q0.shape, h.shape)
        self.assertTrue(torch.is_tensor(loss_0))
        self.assertEqual(loss_0.dim(), 0)
        self.assertEqual(len(perplexities), 1)
        self.assertEqual(len(min_encodings_list), 1)
        self.assertEqual(len(indices_list), 1)
        self.assertEqual(indices_list[0].shape, (16,))
        self.assertGreaterEqual(int(indices_list[0].min()), 0)
        self.assertLess(int(indices_list[0].max()), hvq.level_num_embed[0])

    def test_ste_gradient_flows(self):
        # STE：z_q0 对输入 h 的梯度应能回传（数值为恒等梯度）
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128, requires_grad=True)
        z_q0, _, _ = hvq.forward_level0(h)
        z_q0.sum().backward()
        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(h.grad).all().item())
        self.assertTrue(torch.allclose(h.grad, torch.ones_like(h)))

    def test_default_forward_unchanged(self):
        # 增加 forward_level0 后，默认 forward 仍为完整的 z0+z1 残差量化
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q, loss, (perplexities, _, indices_list) = hvq(h)
        self.assertEqual(len(perplexities), hvq.num_levels)
        self.assertEqual(len(indices_list), hvq.num_levels)
        # 逐级手工重算 sum(z_q_l)
        with torch.no_grad():
            residual = h.clone()
            expected = torch.zeros_like(h)
            for level in hvq.levels:
                _, _, (_, _, idx_l) = level(residual)
                expected = expected + level.embedding.weight[idx_l]
                residual = residual - level.embedding.weight[idx_l]
        self.assertTrue(torch.allclose(z_q, expected, atol=1e-6))


class Z0OnlyConfigTests(unittest.TestCase):
    """默认 configs/config.yaml 必须直接代表 003（z0-only），无需 CLI override。"""

    def test_default_config_enables_z0_only(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("omegaconf 不可用")
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(os.path.dirname(__file__))),
            'configs', 'config.yaml',
        )
        cfg = OmegaConf.load(config_path)
        self.assertTrue(bool(cfg.predictor.z0_only))
        self.assertEqual(int(cfg.train.seed), 0)
        self.assertEqual(str(cfg.vqvae.quantizer.type), 'hvq')


if __name__ == '__main__':
    unittest.main()
