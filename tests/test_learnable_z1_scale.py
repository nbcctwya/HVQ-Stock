import os
import sys
import tempfile
import unittest

import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from module.quantise import VectorQuantiser
from module.quantise_hvq import ResidualVectorQuantiser, create_quantizer
from module.layers.encoder import SpatialEncoder
from module.layers.src import RevIN


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


class ForwardPerLevelTests(unittest.TestCase):
    """ResidualVectorQuantiser.forward_per_level 的行为契约。"""

    def test_per_level_sum_matches_forward(self):
        # sum(z_q_levels) 数值上必须等于默认 forward 的 z0 + z1
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q_full, _, (_, _, indices_full) = hvq(h)
            z_q_levels, loss_pl, (perplexities, _, indices_pl) = hvq.forward_per_level(h)
        self.assertEqual(len(z_q_levels), 2)
        self.assertTrue(torch.allclose(z_q_levels[0] + z_q_levels[1], z_q_full, atol=1e-6))
        # 各级索引与默认 forward 完全一致（同一残差链）
        for level_idx in range(2):
            self.assertTrue(torch.equal(indices_pl[level_idx], indices_full[level_idx]))
        self.assertEqual(len(perplexities), 2)
        self.assertTrue(torch.isfinite(loss_pl).item())

    def test_level_outputs_match_codebook_lookup(self):
        # z_q_levels[l] 必须等于第 l 级 codebook 按索引查到的向量，
        # 即 z0/z1 的生成方式与 001 完全一致，未被重新设计
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q_levels, _, (_, _, indices_list) = hvq.forward_per_level(h)
        for level_idx in range(2):
            expected = hvq.levels[level_idx].embedding.weight[indices_list[level_idx]]
            self.assertTrue(torch.allclose(z_q_levels[level_idx], expected, atol=1e-6))

    def test_default_forward_unchanged(self):
        # 增加 forward_per_level 后，默认 forward 仍为完整的 z0+z1 残差量化
        torch.manual_seed(0)
        hvq = _make_hvq()
        hvq.eval()
        h = torch.randn(32, 128)
        with torch.no_grad():
            z_q, _, (_, _, indices_list) = hvq(h)
            residual = h.clone()
            expected = torch.zeros_like(h)
            for level in hvq.levels:
                _, _, (_, _, idx_l) = level(residual)
                expected = expected + level.embedding.weight[idx_l]
                residual = residual - level.embedding.weight[idx_l]
        self.assertTrue(torch.allclose(z_q, expected, atol=1e-6))
        self.assertEqual(len(indices_list), hvq.num_levels)


def _make_tiny_config(saved_model, **pred_overrides):
    predictor = {
        'saved_model': saved_model,
        'aux_weight': 0.01,
        'kernel_size': 3,
        'k': 1,
        'n_expert': 2,
        'pred_len': 4,
        'moe_hidden': 16,
        'dropout': 0.1,
        'individual': False,
        'num_features': 8,
        'transformer': {
            'pe_kind': 'rope',
            'num_heads': 2,
            'num_layers': 1,
            'd_model': 16,
            'dim_feedforward': 32,
            'dropout': 0.1,
            'batch_first': True,
            'prepend_structure_token': True,
        },
        'rank': 0,
        'target_day': 3,
        'use_prior': True,
        'aux_imp': 3,
        'learnable_z1_scale': True,
        'z1_scale_init': 3.0,
    }
    predictor.update(pred_overrides)
    return {
        'vqvae': {
            'num_prior_factors': 3,
            'num_embed': 32,
            'num_features': 8,
            'hidden_size': 16,
            'vq_embed_dim': 16,
            'seq_len': 5,
            'encoder': {'num_heads': 2, 'num_layers': 1},
            'quantizer': {
                'type': 'hvq',
                'num_levels': 2,
                'level_num_embed': [16, 16],
                'decay': 0.95,
                'commit_weight': 0.25,
                'distance': 'l2',
                'anchor': 'probrandom',
                'first_batch': False,
                'contras_loss': False,
            },
            'decoder': {'initial_T': 3, 'hidden_channels': 16},
        },
        'predictor': predictor,
        'train': {'learning_rate': 1e-4},
    }


def _write_tiny_stage1_checkpoint(vqvae_cfg, path):
    """构造与 GenerateReturn 的 encoder/quantizer/revin 严格同构的
    最小 Stage 1 checkpoint（键名带 vqvae.* 前缀，模拟真实 Stage 1 产物）。"""
    torch.manual_seed(42)
    encoder = SpatialEncoder(
        input_features_C=vqvae_cfg['num_features'],
        T_window=vqvae_cfg['seq_len'],
        gru_hidden_size=vqvae_cfg['hidden_size'],
        num_transformer_heads=vqvae_cfg['encoder']['num_heads'],
        num_transformer_layers=vqvae_cfg['encoder']['num_layers'],
        final_embed_dim_d=vqvae_cfg['vq_embed_dim'],
    )
    quantizer = create_quantizer(vqvae_cfg)
    revin = RevIN(vqvae_cfg['num_features'])

    state_dict = {}
    for k, v in encoder.state_dict().items():
        state_dict[f'vqvae.spatial_encoder.{k}'] = v
    for k, v in quantizer.state_dict().items():
        state_dict[f'vqvae.quantizer.{k}'] = v
    for k, v in revin.state_dict().items():
        state_dict[f'vqvae.revin.{k}'] = v
    torch.save({'state_dict': state_dict}, path)


class LearnableZ1ScaleModelTests(unittest.TestCase):
    """GenerateReturn 层面的 learnable z1 scale（alpha = sigmoid(a)）契约。"""

    @classmethod
    def setUpClass(cls):
        from trainer.train_ypred import GenerateReturn
        cls.GenerateReturn = GenerateReturn
        cls._tmp = tempfile.TemporaryDirectory()
        ckpt_path = os.path.join(cls._tmp.name, 'tiny_stage1.ckpt')
        _write_tiny_stage1_checkpoint(_make_tiny_config(ckpt_path)['vqvae'], ckpt_path)
        cls.ckpt_path = ckpt_path

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _make_model(self, **pred_overrides):
        torch.manual_seed(0)
        config = _make_tiny_config(self.ckpt_path, **pred_overrides)
        return self.GenerateReturn(config, T_max=10)

    def _make_batch(self, config, batch_size=6):
        torch.manual_seed(7)
        n_features = config['vqvae']['num_features']
        seq_len = config['vqvae']['seq_len']
        n_prior = config['vqvae']['num_prior_factors']
        feature = torch.randn(batch_size, seq_len, n_features)
        prior = torch.randn(batch_size, n_prior)
        return feature, prior

    def test_alpha_is_trainable_stage2_parameter(self):
        # alpha 的原始参数 a 必须是 Stage 2 模块上的可训练参数，
        # 且进入 configure_optimizers 的优化器参数组
        model = self._make_model()
        self.assertIsInstance(model.z1_scale_raw, torch.nn.Parameter)
        self.assertTrue(model.z1_scale_raw.requires_grad)
        self.assertEqual(model.z1_scale_raw.numel(), 1)  # 全局单个标量

        optimizers, _ = model.configure_optimizers()
        opt_params = {id(p) for group in optimizers[0].param_groups for p in group['params']}
        self.assertIn(id(model.z1_scale_raw), opt_params)

    def test_alpha_bounded_in_01(self):
        # sigmoid 后 alpha 始终位于 [0,1]，包括极端取值
        model = self._make_model()
        for raw in (-100.0, -3.0, 0.0, 3.0, 100.0):
            with torch.no_grad():
                model.z1_scale_raw.fill_(raw)
            alpha = torch.sigmoid(model.z1_scale_raw).item()
            self.assertGreaterEqual(alpha, 0.0)
            self.assertLessEqual(alpha, 1.0)

    def test_alpha_init_matches_config(self):
        # 初始化：a=3.0 -> alpha = sigmoid(3.0) ≈ 0.9526（接近 001 的 alpha=1 起点）
        model = self._make_model()
        self.assertAlmostEqual(model.z1_scale_raw.item(), 3.0, places=6)
        alpha = torch.sigmoid(model.z1_scale_raw).item()
        self.assertAlmostEqual(alpha, 0.9525741, places=5)

    def test_fusion_is_exactly_z0_plus_alpha_z1(self):
        # Stage 2 融合必须严格为 z_q = z0 + alpha * z1
        model = self._make_model()
        model.eval()
        config = model.config
        feature, prior = self._make_batch(config)
        with torch.no_grad():
            _, _, _, z_q, _ = model.forward(feature, prior)
            h_batch = model.encoder(model.revin(feature, mode='norm'))
            z_q_levels, _, _ = model.quantizer.forward_per_level(h_batch)
            alpha = torch.sigmoid(model.z1_scale_raw)
            expected = z_q_levels[0] + alpha * z_q_levels[1]
        self.assertTrue(torch.allclose(z_q, expected, atol=1e-6))
        # alpha=1 时退化为 001 的 z0+z1（数值一致性）
        self.assertFalse(torch.allclose(z_q, z_q_levels[0] + z_q_levels[1], atol=1e-6))

    def test_alpha_receives_gradient_and_updates(self):
        # alpha 必须能从 Stage 2 损失获得梯度并在优化步骤中实际更新
        model = self._make_model()
        model.train()
        config = model.config
        feature, prior = self._make_batch(config)

        optimizer = torch.optim.SGD([model.z1_scale_raw], lr=0.1)
        before = model.z1_scale_raw.item()

        y_pred, _, _, _, aux_loss = model.forward(feature, prior)
        loss = y_pred.sum() + aux_loss
        loss.backward()

        self.assertIsNotNone(model.z1_scale_raw.grad)
        self.assertTrue(torch.isfinite(model.z1_scale_raw.grad).all().item())
        self.assertNotEqual(model.z1_scale_raw.grad.item(), 0.0)

        optimizer.step()
        self.assertNotEqual(model.z1_scale_raw.item(), before)
        # 更新后 alpha 仍在 [0,1]
        alpha = torch.sigmoid(model.z1_scale_raw).item()
        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)

    def test_stage1_remains_frozen(self):
        # Stage 1（encoder/quantizer/revin）全部参数保持冻结，仅 a 可训练
        model = self._make_model()
        for module in (model.encoder, model.quantizer, model.revin):
            for param in module.parameters():
                self.assertFalse(param.requires_grad)
        self.assertTrue(model.z1_scale_raw.requires_grad)
        # Stage 1 子模块被强制 eval
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.quantizer.training)
        self.assertFalse(model.revin.training)

    def test_checkpoint_save_load_restores_alpha(self):
        # checkpoint 保存/加载后 alpha 能正确恢复（strict 加载）
        model = self._make_model()
        with torch.no_grad():
            model.z1_scale_raw.fill_(-1.25)
        expected_alpha = torch.sigmoid(model.z1_scale_raw).item()

        ckpt = {'state_dict': model.state_dict()}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.ckpt')
            torch.save(ckpt, path)
            loaded = torch.load(path, map_location='cpu')

        new_model = self._make_model()
        new_model.load_state_dict(loaded['state_dict'], strict=True)
        self.assertAlmostEqual(new_model.z1_scale_raw.item(), -1.25, places=6)
        self.assertAlmostEqual(torch.sigmoid(new_model.z1_scale_raw).item(),
                               expected_alpha, places=6)

    def test_requires_hvq_quantizer(self):
        # learnable_z1_scale=True 必须要求两级 hvq 量化器
        model = self._make_model()
        self.assertIsInstance(model.quantizer, ResidualVectorQuantiser)
        self.assertEqual(model.quantizer.num_levels, 2)

        ckpt_single = os.path.join(self._tmp.name, 'tiny_stage1_single.ckpt')
        single_cfg = _make_tiny_config(ckpt_single)['vqvae']
        single_cfg['quantizer'] = dict(single_cfg['quantizer'])
        single_cfg['quantizer']['type'] = 'single'
        _write_tiny_stage1_checkpoint(single_cfg, ckpt_single)
        with self.assertRaises(ValueError):
            config = _make_tiny_config(ckpt_single)
            config['vqvae']['quantizer']['type'] = 'single'
            self.GenerateReturn(config, T_max=10)


class LearnableZ1ScaleConfigTests(unittest.TestCase):
    """默认 configs/config.yaml 必须直接代表 004，无需实验特有 CLI override。"""

    def test_default_config_enables_learnable_z1_scale(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("omegaconf 不可用")
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(os.path.dirname(__file__))),
            'configs', 'config.yaml',
        )
        cfg = OmegaConf.load(config_path)
        self.assertTrue(bool(cfg.predictor.learnable_z1_scale))
        self.assertAlmostEqual(float(cfg.predictor.z1_scale_init), 3.0, places=6)
        self.assertEqual(int(cfg.train.seed), 0)
        self.assertEqual(str(cfg.vqvae.quantizer.type), 'hvq')
        self.assertEqual(int(cfg.vqvae.quantizer.num_levels), 2)
        self.assertEqual(list(cfg.vqvae.quantizer.level_num_embed), [256, 256])


if __name__ == '__main__':
    unittest.main()
