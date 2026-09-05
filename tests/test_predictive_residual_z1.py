"""006 — hvq-predictive-residual-z1 单元测试。

覆盖：
- ResidualVectorQuantiser.forward_two_levels 的行为契约（z_q0 + z_q1 == forward）；
- 003 原有 z0 主预测路径仍可独立生成 ŷ0（与 z0-only 模型逐位一致）；
- z1 不与 z0 在 representation level 做 z0+z1 融合；
- residual target 严格等于 y - stopgrad(ŷ0)，且不对 ŷ0 传播梯度；
- Δŷ 由 z1 residual head 生成，最终预测严格满足 ŷ = ŷ0 + Δŷ；
- residual head 参数能获得梯度并更新；
- Stage 1 encoder / quantizer / RevIN 继续冻结；
- Stage 1 strict checkpoint load PASS（复用 001 的 exact Stage 1 checkpoint）；
- checkpoint 保存/恢复后 residual head 状态正确；
- 默认 config 无需实验特有 CLI override 即启用 006。
"""
import os
import sys
import tempfile
import unittest

import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from module.quantise_hvq import ResidualVectorQuantiser

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
STAGE1_CKPT = os.path.join(
    REPO_ROOT, 'artifacts', '001', 'run', 'checkpoints',
    'hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt',
)


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


def _load_config(z1_residual_branch=True):
    from omegaconf import OmegaConf
    try:
        OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)
    except ValueError:
        pass  # resolver 已注册
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, 'configs', 'config.yaml'))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['predictor']['saved_model'] = STAGE1_CKPT
    cfg_dict['predictor']['z1_residual_branch'] = z1_residual_branch
    return cfg_dict


def _build_model(z1_residual_branch=True):
    from trainer.train_ypred import GenerateReturn
    model = GenerateReturn(_load_config(z1_residual_branch), T_max=10)
    model.eval()
    return model


def _make_batch(seed=0, n=8):
    g = torch.Generator().manual_seed(seed)
    feature = torch.randn(n, 20, 158, generator=g)
    prior = torch.randn(n, 13, generator=g)
    label = torch.randn(n, generator=g)
    return feature, prior, label


class ForwardTwoLevelsTests(unittest.TestCase):
    """ResidualVectorQuantiser.forward_two_levels 的行为契约。"""

    def test_two_levels_sum_matches_forward(self):
        # z_q0 + z_q1 必须在数值上等于默认 forward 的 z0+z1
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q0, z_q1, loss, (_, _, indices) = hvq.forward_two_levels(h)
            z_q_full, _, _ = hvq(h)
            z_q0_ref, _, _ = hvq.forward_level0(h)
        self.assertTrue(torch.allclose(z_q0, z_q0_ref, atol=1e-6))
        self.assertTrue(torch.allclose(z_q0 + z_q1, z_q_full, atol=1e-6))
        self.assertEqual(len(indices), 2)
        # z_q1 必须等于第 1 级 codebook 查表向量
        expected_z1 = hvq.levels[1].embedding.weight[indices[1]]
        self.assertTrue(torch.allclose(z_q1, expected_z1, atol=1e-6))

    def test_default_forward_unchanged(self):
        # 增加 forward_two_levels 后，默认 forward 行为不变
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q, _, (_, _, indices_list) = hvq(h)
        residual = h.clone()
        expected = torch.zeros_like(h)
        with torch.no_grad():
            for level in hvq.levels:
                _, _, (_, _, idx_l) = level(residual)
                expected = expected + level.embedding.weight[idx_l]
                residual = residual - level.embedding.weight[idx_l]
        self.assertTrue(torch.allclose(z_q, expected, atol=1e-6))
        self.assertEqual(len(indices_list), 2)


@unittest.skipUnless(os.path.isfile(STAGE1_CKPT),
                     "001 Stage 1 checkpoint 不在当前 workspace（artifacts/ 为本地副本）")
class PredictiveResidualModelTests(unittest.TestCase):
    """GenerateReturn 的 006 prediction-residual branch 行为。"""

    @classmethod
    def setUpClass(cls):
        cls.model = _build_model(z1_residual_branch=True)

    def test_main_path_matches_003_z0_only(self):
        # 003 原有 z0 主预测路径仍可独立生成 ŷ0：006 模型关闭 branch 后
        # （仅缺少 residual_head），同一输入下 ŷ0 必须与 003 输出逐位一致
        model_003 = _build_model(z1_residual_branch=False)
        missing, unexpected = model_003.load_state_dict(self.model.state_dict(), strict=False)
        self.assertEqual(sorted(unexpected), ['residual_head.bias', 'residual_head.weight'])
        self.assertEqual(missing, [])

        feature, prior, _ = _make_batch()
        with torch.no_grad():
            out = self.model(feature, prior, return_components=True)
            y_pred_003, _, _, _, _ = model_003(feature, prior)
        self.assertTrue(torch.allclose(out['y0'], y_pred_003, atol=1e-6))

    def test_no_representation_level_fusion(self):
        # z1 不与 z0 在 representation level 融合：主路径输入必须是 z0 而非 z0+z1
        feature, prior, _ = _make_batch()
        with torch.no_grad():
            out = self.model(feature, prior, return_components=True)
            h = self.model.encoder(self.model.revin(feature, mode='norm'))
            z0, _, _ = self.model.quantizer.forward_level0(h)
            z_full, _, _ = self.model.quantizer(h)
        self.assertTrue(torch.allclose(out['z_q'], z0, atol=1e-6))
        self.assertFalse(torch.allclose(out['z_q'], z_full, atol=1e-6))
        # z_q1 确实取自第二级残差量化（forward_two_levels 与两级独立重算一致）
        self.assertTrue(torch.allclose(out['z_q1'], z_full - z0, atol=1e-6))

    def test_residual_target_definition(self):
        # residual target 严格等于 y - stopgrad(ŷ0)
        y0 = torch.randn(8, requires_grad=True)
        delta = torch.randn(8)
        label = torch.randn(8)
        _, _, res_target = self.model._residual_branch_losses(y0, delta, label)
        self.assertTrue(torch.allclose(res_target, label - y0.detach()))
        self.assertFalse(res_target.requires_grad)

    def test_residual_target_stopgrad(self):
        # residual loss 的梯度不得经 residual target 回传到 ŷ0
        w = torch.nn.Parameter(torch.randn(8))
        x = torch.randn(8)
        y0 = w * x
        delta = torch.randn(8, requires_grad=True)
        label = torch.randn(8)
        _, res_loss, _ = self.model._residual_branch_losses(y0, delta, label)
        res_loss.backward()
        self.assertIsNone(w.grad)
        self.assertIsNotNone(delta.grad)

    def test_delta_from_z1_head(self):
        # Δŷ 必须由 z1 residual head 生成：数值上等于 residual_head(z_q1)
        feature, prior, _ = _make_batch()
        with torch.no_grad():
            out = self.model(feature, prior, return_components=True)
            delta_ref = self.model.residual_head(out['z_q1']).squeeze(-1)
        self.assertTrue(torch.allclose(out['delta_y'], delta_ref, atol=1e-6))
        # 扰动 z1 必须改变 Δŷ
        with torch.no_grad():
            delta_perturbed = self.model.residual_head(out['z_q1'] + 1.0).squeeze(-1)
        self.assertFalse(torch.allclose(out['delta_y'], delta_perturbed))

    def test_final_prediction_is_sum(self):
        # 最终 prediction 严格满足 ŷ = ŷ0 + Δŷ
        feature, prior, _ = _make_batch()
        with torch.no_grad():
            out = self.model(feature, prior, return_components=True)
            y_pred_default, _, _, _, _ = self.model(feature, prior)
        self.assertTrue(torch.allclose(out['y_pred'], out['y0'] + out['delta_y'], atol=1e-6))
        # 默认 5 元组返回的第一个元素必须是最终 ŷ（与 run_inference 兼容）
        self.assertTrue(torch.allclose(y_pred_default, out['y_pred'], atol=1e-6))

    def test_residual_head_gets_gradient_and_updates(self):
        # residual head 参数能获得（非零）梯度并在一步优化后更新
        model = _build_model(z1_residual_branch=True)
        feature, prior, label = _make_batch()
        out = model.forward(feature, prior, return_components=True)
        main_loss, res_loss, _ = model._residual_branch_losses(out['y0'], out['delta_y'], label)
        loss = main_loss + res_loss + model.aux_weight * out['loss_imp']
        loss.backward()

        for name, p in model.residual_head.named_parameters():
            self.assertIsNotNone(p.grad, f"residual_head.{name} 无梯度")
            self.assertGreater(p.grad.abs().max().item(), 0.0)

        before = [p.detach().clone() for p in model.residual_head.parameters()]
        opt = torch.optim.SGD(model.residual_head.parameters(), lr=0.1)
        opt.step()
        after = list(model.residual_head.parameters())
        self.assertTrue(any(not torch.allclose(b, a) for b, a in zip(before, after)))

    def test_stage1_frozen(self):
        # encoder / quantizer / RevIN 继续冻结，且完整 loss 反传后仍无梯度
        model = _build_model(z1_residual_branch=True)
        for module in (model.encoder, model.quantizer, model.revin):
            for p in module.parameters():
                self.assertFalse(p.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.residual_head.parameters()))

        feature, prior, label = _make_batch()
        out = model.forward(feature, prior, return_components=True)
        main_loss, res_loss, _ = model._residual_branch_losses(out['y0'], out['delta_y'], label)
        (main_loss + res_loss + model.aux_weight * out['loss_imp']).backward()
        for module in (model.encoder, model.quantizer, model.revin):
            for p in module.parameters():
                self.assertIsNone(p.grad)

    def test_stage1_strict_checkpoint_load(self):
        # Stage 1 strict checkpoint load PASS：encoder / quantizer / revin 全部
        # missing=0 / unexpected=0，且两级量化器配置与 001 checkpoint 一致
        checkpoint = torch.load(STAGE1_CKPT, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        enc_sd, vq_sd, revin_sd = {}, {}, {}
        for k, v in state_dict.items():
            if k.startswith('vqvae.spatial_encoder.'):
                enc_sd[k.replace('vqvae.spatial_encoder.', '')] = v
            elif k.startswith('vqvae.quantizer.'):
                vq_sd[k.replace('vqvae.quantizer.', '')] = v
            elif k.startswith('vqvae.revin.'):
                revin_sd[k.replace('vqvae.revin.', '')] = v

        m, u = self.model.encoder.load_state_dict(enc_sd, strict=True)
        self.assertEqual((len(m), len(u)), (0, 0))
        m, u = self.model.quantizer.load_state_dict(vq_sd, strict=True)
        self.assertEqual((len(m), len(u)), (0, 0))
        m, u = self.model.revin.load_state_dict(revin_sd, strict=True)
        self.assertEqual((len(m), len(u)), (0, 0))

        self.assertIsInstance(self.model.quantizer, ResidualVectorQuantiser)
        self.assertEqual(self.model.quantizer.num_levels, 2)
        self.assertEqual(self.model.quantizer.level_num_embed, [256, 256])

    def test_checkpoint_save_restore_residual_head(self):
        # checkpoint 保存/恢复后 residual head 状态正确，输出一致
        feature, prior, _ = _make_batch()
        with torch.no_grad():
            out_before = self.model(feature, prior, return_components=True)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.ckpt')
            torch.save({'state_dict': self.model.state_dict()}, path)
            restored = _build_model(z1_residual_branch=True)
            missing, unexpected = restored.load_state_dict(
                torch.load(path, map_location='cpu')['state_dict'], strict=True)
            self.assertEqual((len(missing), len(unexpected)), (0, 0))
        restored.eval()

        for p_new, p_old in zip(restored.residual_head.parameters(),
                                self.model.residual_head.parameters()):
            self.assertTrue(torch.allclose(p_new, p_old))
        with torch.no_grad():
            out_after = restored(feature, prior, return_components=True)
        self.assertTrue(torch.allclose(out_after['y_pred'], out_before['y_pred'], atol=1e-6))
        self.assertTrue(torch.allclose(out_after['delta_y'], out_before['delta_y'], atol=1e-6))


class PredictiveResidualConfigTests(unittest.TestCase):
    """默认 configs/config.yaml 必须直接代表 006，无需实验特有 CLI override。"""

    def test_default_config_enables_006(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("omegaconf 不可用")
        cfg = OmegaConf.load(os.path.join(REPO_ROOT, 'configs', 'config.yaml'))
        self.assertTrue(bool(cfg.predictor.z1_residual_branch))
        self.assertTrue(bool(cfg.predictor.z0_only))
        self.assertEqual(int(cfg.train.seed), 0)
        self.assertEqual(str(cfg.vqvae.quantizer.type), 'hvq')
        self.assertEqual(int(cfg.vqvae.quantizer.num_levels), 2)


if __name__ == '__main__':
    unittest.main()
