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

SIGMOID_3 = 0.9525741  # sigmoid(3.0)


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
    """ResidualVectorQuantiser.forward_per_level 的行为契约（沿用 004）。"""

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
        # 默认 forward 仍为完整的 z0+z1 残差量化
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
        'samplewise_z1_gate': True,
        'z1_gate_bias_init': 3.0,
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


class SamplewiseZ1GateModelTests(unittest.TestCase):
    """GenerateReturn 层面的 sample-wise z1 gate 契约。

    gate：z_i = z0_i + g_i * z1_i，g_i = sigmoid(Linear(concat(z0_i, z1_i)))。
    """

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

    def _forward_gate(self, model, feature, prior):
        with torch.no_grad():
            model.forward(feature, prior)
        return model._last_z1_gate

    def test_gate_module_shape_and_param_count(self):
        # gate 为单个 Linear(2*d -> 1)，参数量 2*d + 1（weight + bias）
        model = self._make_model()
        d = model.vq_embed_dim
        self.assertIsInstance(model.z1_gate, torch.nn.Linear)
        self.assertEqual(model.z1_gate.in_features, 2 * d)
        self.assertEqual(model.z1_gate.out_features, 1)
        n_params = sum(p.numel() for p in model.z1_gate.parameters())
        self.assertEqual(n_params, 2 * d + 1)

    def test_gate_output_shape_per_sample(self):
        # gate 输出必须为每个 sample 一个 scalar：(B, 1)
        model = self._make_model()
        model.eval()
        feature, prior = self._make_batch(model.config, batch_size=6)
        g = self._forward_gate(model, feature, prior)
        self.assertEqual(tuple(g.shape), (6, 1))

    def test_gate_bounded_in_01(self):
        # sigmoid 后 gate 始终位于 [0,1]，包括极端 weight/bias 取值
        model = self._make_model()
        model.eval()
        feature, prior = self._make_batch(model.config)
        for fill in (-100.0, -3.0, 0.0, 3.0, 100.0):
            with torch.no_grad():
                model.z1_gate.weight.fill_(fill)
                model.z1_gate.bias.fill_(fill)
            g = self._forward_gate(model, feature, prior)
            self.assertTrue((g >= 0.0).all().item())
            self.assertTrue((g <= 1.0).all().item())

    def test_gate_init_all_sigmoid_3(self):
        # weight 全零、bias=3.0 时所有样本初始 gate 均为 sigmoid(3.0)
        model = self._make_model()
        self.assertTrue(torch.all(model.z1_gate.weight == 0).item())
        self.assertAlmostEqual(model.z1_gate.bias.item(), 3.0, places=6)
        model.eval()
        feature, prior = self._make_batch(model.config)
        g = self._forward_gate(model, feature, prior)
        self.assertTrue(torch.allclose(g, torch.full_like(g, SIGMOID_3), atol=1e-6))
        self.assertAlmostEqual(g.std().item(), 0.0, places=6)

    def test_init_fusion_matches_004_initial_behavior(self):
        # 初始化状态下融合严格等价于 z0 + sigmoid(3.0) * z1（与 004 初始一致）
        model = self._make_model()
        model.eval()
        feature, prior = self._make_batch(model.config)
        with torch.no_grad():
            _, _, _, z_q, _ = model.forward(feature, prior)
            h_batch = model.encoder(model.revin(feature, mode='norm'))
            z_q_levels, _, _ = model.quantizer.forward_per_level(h_batch)
            expected = z_q_levels[0] + SIGMOID_3 * z_q_levels[1]
        self.assertTrue(torch.allclose(z_q, expected, atol=1e-6))
        # 且不等价于 001 的固定 z0+z1（数值上确实不同）
        self.assertFalse(torch.allclose(z_q, z_q_levels[0] + z_q_levels[1], atol=1e-6))

    def test_fusion_is_exactly_z0_plus_g_z1(self):
        # 非零 weight 下融合仍严格为 z_q = z0 + g * z1（g 为 per-sample）
        model = self._make_model()
        model.eval()
        d = model.vq_embed_dim
        with torch.no_grad():
            model.z1_gate.weight.copy_(torch.randn(1, 2 * d) * 0.1)
        feature, prior = self._make_batch(model.config)
        with torch.no_grad():
            _, _, _, z_q, _ = model.forward(feature, prior)
            h_batch = model.encoder(model.revin(feature, mode='norm'))
            z_q_levels, _, _ = model.quantizer.forward_per_level(h_batch)
            z0, z1 = z_q_levels[0], z_q_levels[1]
            g = torch.sigmoid(model.z1_gate(torch.cat([z0, z1], dim=1)))
            expected = z0 + g * z1
        self.assertTrue(torch.allclose(z_q, expected, atol=1e-6))
        self.assertTrue(torch.allclose(model._last_z1_gate, g, atol=1e-6))

    def test_different_samples_different_gates(self):
        # 非零 weight 后不同样本必须能产生不同 gate（不退化为常数）
        model = self._make_model()
        model.eval()
        d = model.vq_embed_dim
        with torch.no_grad():
            model.z1_gate.weight.copy_(torch.randn(1, 2 * d))
        feature, prior = self._make_batch(model.config)
        g = self._forward_gate(model, feature, prior)
        self.assertGreater(g.std().item(), 0.0)

    def test_gate_params_in_optimizer(self):
        # gate 的 weight/bias 必须进入 configure_optimizers 的优化器参数组
        model = self._make_model()
        self.assertTrue(model.z1_gate.weight.requires_grad)
        self.assertTrue(model.z1_gate.bias.requires_grad)

        optimizers, _ = model.configure_optimizers()
        opt_params = {id(p) for group in optimizers[0].param_groups for p in group['params']}
        self.assertIn(id(model.z1_gate.weight), opt_params)
        self.assertIn(id(model.z1_gate.bias), opt_params)

    def test_gate_receives_gradient_and_updates(self):
        # gate 必须能从 Stage 2 损失获得非零梯度并在优化步骤中实际更新
        model = self._make_model()
        model.train()
        feature, prior = self._make_batch(model.config)

        optimizer = torch.optim.SGD(model.z1_gate.parameters(), lr=0.1)
        w_before = model.z1_gate.weight.detach().clone()
        b_before = model.z1_gate.bias.detach().clone()

        y_pred, _, _, _, aux_loss = model.forward(feature, prior)
        loss = y_pred.sum() + aux_loss
        loss.backward()

        for p in model.z1_gate.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
        self.assertNotEqual(model.z1_gate.weight.grad.abs().sum().item(), 0.0)
        self.assertNotEqual(model.z1_gate.bias.grad.abs().sum().item(), 0.0)

        optimizer.step()
        self.assertFalse(torch.allclose(model.z1_gate.weight, w_before))
        self.assertFalse(torch.allclose(model.z1_gate.bias, b_before))

        # 更新后 gate 仍在 [0,1]
        model.eval()
        g = self._forward_gate(model, feature, prior)
        self.assertTrue((g >= 0.0).all().item())
        self.assertTrue((g <= 1.0).all().item())

    def test_stage1_remains_frozen(self):
        # Stage 1（encoder/quantizer/revin）全部参数保持冻结，仅 gate 等
        # Stage 2 参数可训练；z0/z1 detach 保证 Stage 1 拿不到梯度
        model = self._make_model()
        for module in (model.encoder, model.quantizer, model.revin):
            for param in module.parameters():
                self.assertFalse(param.requires_grad)
        self.assertTrue(model.z1_gate.weight.requires_grad)
        self.assertTrue(model.z1_gate.bias.requires_grad)
        # Stage 1 子模块被强制 eval
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.quantizer.training)
        self.assertFalse(model.revin.training)

        # backward 后 Stage 1 参数无任何梯度
        feature, prior = self._make_batch(model.config)
        y_pred, _, _, _, aux_loss = model.forward(feature, prior)
        (y_pred.sum() + aux_loss).backward()
        for module in (model.encoder, model.quantizer, model.revin):
            for param in module.parameters():
                self.assertIsNone(param.grad)

    def test_checkpoint_save_load_restores_gate(self):
        # checkpoint 保存/加载后 gate 参数能正确恢复（strict 加载）
        model = self._make_model()
        with torch.no_grad():
            model.z1_gate.weight.copy_(torch.randn_like(model.z1_gate.weight) * 0.5)
            model.z1_gate.bias.fill_(1.25)
        expected_w = model.z1_gate.weight.detach().clone()
        expected_b = model.z1_gate.bias.detach().clone()

        ckpt = {'state_dict': model.state_dict()}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.ckpt')
            torch.save(ckpt, path)
            loaded = torch.load(path, map_location='cpu')

        new_model = self._make_model()
        new_model.load_state_dict(loaded['state_dict'], strict=True)
        self.assertTrue(torch.allclose(new_model.z1_gate.weight, expected_w))
        self.assertTrue(torch.allclose(new_model.z1_gate.bias, expected_b))

        # 加载后同一输入产生相同 gate
        new_model.eval()
        model.eval()
        feature, prior = self._make_batch(model.config)
        g_loaded = self._forward_gate(new_model, feature, prior)
        g_orig = self._forward_gate(model, feature, prior)
        self.assertTrue(torch.allclose(g_loaded, g_orig, atol=1e-6))

    def test_no_global_alpha_param(self):
        # 005 不保留 004 的全局 learnable alpha，禁止双重缩放
        model = self._make_model()
        self.assertFalse(hasattr(model, 'z1_scale_raw'))

    def test_requires_hvq_quantizer(self):
        # samplewise_z1_gate=True 必须要求两级 hvq 量化器
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


class SamplewiseZ1GateConfigTests(unittest.TestCase):
    """默认 configs/config.yaml 必须直接代表 005，无需实验特有 CLI override。"""

    def test_default_config_enables_samplewise_z1_gate(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("omegaconf 不可用")
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(os.path.dirname(__file__))),
            'configs', 'config.yaml',
        )
        cfg = OmegaConf.load(config_path)
        self.assertTrue(bool(cfg.predictor.samplewise_z1_gate))
        self.assertAlmostEqual(float(cfg.predictor.z1_gate_bias_init), 3.0, places=6)
        # 004 的全局 alpha 开关不得残留在默认配置中
        self.assertNotIn('learnable_z1_scale', cfg.predictor)
        self.assertNotIn('z1_scale_init', cfg.predictor)
        self.assertEqual(int(cfg.train.seed), 0)
        self.assertEqual(str(cfg.vqvae.quantizer.type), 'hvq')
        self.assertEqual(int(cfg.vqvae.quantizer.num_levels), 2)
        self.assertEqual(list(cfg.vqvae.quantizer.level_num_embed), [256, 256])


if __name__ == '__main__':
    unittest.main()
