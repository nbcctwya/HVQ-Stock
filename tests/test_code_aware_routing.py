"""Regression tests for experiment 022 explicit code-aware routing."""

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module.layers.moe import FactorGatedMoE
from trainer.train_ypred import GenerateReturn


def tiny_config(code_aware_routing=True):
    return {
        "vqvae": {
            "num_features": 8,
            "seq_len": 5,
            "hidden_size": 8,
            "num_prior_factors": 3,
            "vq_embed_dim": 8,
            "num_embed": 16,
            "encoder": {"num_heads": 2, "num_layers": 1},
            "quantizer": {
                "decay": 0.95,
                "commit_weight": 0.25,
                "distance": "l2",
                "anchor": "probrandom",
                "first_batch": False,
                "contras_loss": True,
            },
            "decoder": {"initial_T": 2, "hidden_channels": 8},
        },
        "predictor": {
            "saved_model": "unused.ckpt",
            "num_features": 8,
            "individual": False,
            "aux_weight": 0.01,
            "aux_imp": 3,
            "kernel_size": 3,
            "n_expert": 2,
            "k": 1,
            "pred_len": 4,
            "moe_hidden": 8,
            "dropout": 0.1,
            "rank": 0,
            "target_day": 2,
            "use_prior": True,
            "code_aware_routing": code_aware_routing,
            "transformer": {
                "num_heads": 2,
                "num_layers": 1,
                "d_model": 8,
                "dim_feedforward": 16,
                "dropout": 0.1,
                "batch_first": True,
                "prepend_structure_token": True,
            },
        },
        "train": {"learning_rate": 0.0001},
    }


def build_model(code_aware_routing=True, seed=0):
    torch.manual_seed(seed)
    config = copy.deepcopy(tiny_config(code_aware_routing))
    with mock.patch.object(
        GenerateReturn,
        "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(config, T_max=10)


def routing_module(model):
    return model.loadings.fusion.moe


def quantizer_code_ids(model, feature):
    """Recompute the discrete code ids exactly as the Stage 1 quantizer does."""
    with torch.no_grad():
        h_batch = model.encoder(model.revin(feature, mode="norm"))
        _, _, (_, _, vq_idx) = model.quantizer(h_batch)
    return vq_idx


class CodeAwareRoutingTest(unittest.TestCase):
    def test_default_config_enables_code_aware_routing(self):
        with (Path(__file__).parents[1] / "configs" / "config.yaml").open() as stream:
            config = yaml.safe_load(stream)
        self.assertIs(config["predictor"]["code_aware_routing"], True)
        self.assertEqual(config["predictor"]["n_expert"], 2)
        self.assertEqual(config["predictor"]["k"], "${half:${predictor.n_expert}}")
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_vq_idx_passed_from_quantizer_to_moe_router(self):
        model = build_model(True, seed=5).eval()
        moe = routing_module(model)
        feature = torch.randn(7, 5, 8)
        prior = torch.randn(7, 3)
        captured = {}
        original = FactorGatedMoE.clean_routing_logits

        def spy(self, x, vq_idx=None):
            captured["vq_idx"] = vq_idx
            return original(self, x, vq_idx=vq_idx)

        with mock.patch.object(FactorGatedMoE, "clean_routing_logits", spy):
            with torch.no_grad():
                model(feature, prior)
        self.assertIn("vq_idx", captured)
        expected = quantizer_code_ids(model, feature)
        self.assertTrue(torch.equal(captured["vq_idx"], expected))

    def test_code_bias_table_shape_and_zero_init(self):
        model = build_model(True, seed=3)
        moe = routing_module(model)
        table = moe.code_bias
        # Dimensions come from vqvae.num_embed and predictor.n_expert.
        self.assertEqual(tuple(table.shape), (16, 2))
        self.assertIsInstance(table, torch.nn.Parameter)
        self.assertTrue(table.requires_grad)
        self.assertEqual(torch.count_nonzero(table.detach()).item(), 0)
        # The real configuration must give exactly [512, n_expert].
        real = FactorGatedMoE(
            gate_input_size=128,
            expert_input_size=64,
            hidden_size=64,
            num_experts=2,
            k=1,
            code_aware_routing=True,
            num_codes=512,
        )
        self.assertEqual(tuple(real.code_bias.shape), (512, 2))
        self.assertEqual(torch.count_nonzero(real.code_bias.detach()).item(), 0)
        # Missing or invalid num_codes must fail loudly.
        for bad in (None, 0, -3, 2.5):
            with self.assertRaises((ValueError, TypeError)):
                FactorGatedMoE(
                    gate_input_size=8, expert_input_size=8, hidden_size=8,
                    num_experts=2, k=1, code_aware_routing=True, num_codes=bad,
                )

    def test_code_bias_does_not_perturb_any_existing_state_initialization(self):
        base = build_model(False, seed=1234)
        adapted = build_model(True, seed=1234)
        prefix = "loadings.fusion.moe.code_bias"
        adapted_base_state = {
            key: value for key, value in adapted.state_dict().items()
            if not key.startswith(prefix)
        }
        self.assertEqual(base.state_dict().keys(), adapted_base_state.keys())
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, adapted_base_state[key]), msg=key)

    def test_zero_init_clean_logits_routing_aux_and_full_forward_equal_base(self):
        base = build_model(False, seed=4321).eval()
        adapted = build_model(True, seed=4321).eval()
        base_moe, adapted_moe = routing_module(base), routing_module(adapted)
        z = torch.randn(9, 8)
        x = torch.randn(9, 8)
        vq_idx = torch.randint(0, 16, (9,))

        self.assertTrue(torch.equal(
            base_moe.clean_routing_logits(z),
            adapted_moe.clean_routing_logits(z, vq_idx),
        ))
        base_gates, base_load = base_moe.noisy_top_k_gating(z, False)
        adapted_gates, adapted_load = adapted_moe.noisy_top_k_gating(
            z, False, vq_idx=vq_idx
        )
        self.assertTrue(torch.equal(base_gates, adapted_gates))
        self.assertTrue(torch.equal(base_load, adapted_load))
        base_y, base_aux = base_moe(x, z)
        adapted_y, adapted_aux = adapted_moe(x, z, vq_idx=vq_idx)
        self.assertTrue(torch.equal(base_y, adapted_y))
        self.assertTrue(torch.equal(base_aux, adapted_aux))

        feature = torch.randn(9, 5, 8)
        prior = torch.randn(9, 3)
        base_output = base(feature, prior)
        adapted_output = adapted(feature, prior)
        for index, (base_value, adapted_value) in enumerate(zip(base_output, adapted_output)):
            self.assertTrue(torch.equal(base_value, adapted_value), msg=f"output[{index}]")

    def test_original_noisy_topk_noise_wh_and_load_behavior_is_unchanged(self):
        base = routing_module(build_model(False, seed=88)).train()
        adapted = routing_module(build_model(True, seed=88)).train()
        z = torch.randn(32, 8)
        vq_idx = torch.randint(0, 16, (32,))
        self.assertTrue(torch.equal(base.W_h, adapted.W_h))
        for left, right in zip(base.gate.state_dict().values(), adapted.gate.state_dict().values()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(base.noise.state_dict().values(), adapted.noise.state_dict().values()):
            self.assertTrue(torch.equal(left, right))
        torch.manual_seed(2020)
        base_gates, base_load = base.noisy_top_k_gating(z, True)
        torch.manual_seed(2020)
        adapted_gates, adapted_load = adapted.noisy_top_k_gating(z, True, vq_idx=vq_idx)
        self.assertTrue(torch.equal(base_gates, adapted_gates))
        self.assertTrue(torch.equal(base_load, adapted_load))

    def test_nonzero_code_bias_changes_logits_and_expert_preference_for_fixed_z(self):
        moe = FactorGatedMoE(
            gate_input_size=8,
            expert_input_size=8,
            hidden_size=8,
            num_experts=2,
            k=1,
            code_aware_routing=True,
            num_codes=16,
        ).eval()
        for parameter in moe.gate.parameters():
            torch.nn.init.zeros_(parameter)
        with torch.no_grad():
            moe.code_bias[0].copy_(torch.tensor([2.0, -2.0]))
            moe.code_bias[1].copy_(torch.tensor([-2.0, 2.0]))
        z = torch.zeros(6, 8)
        idx_a = torch.zeros(6, dtype=torch.long)
        idx_b = torch.ones(6, dtype=torch.long)
        logits_a = moe.clean_routing_logits(z, idx_a)
        logits_b = moe.clean_routing_logits(z, idx_b)
        gates_a, _ = moe.noisy_top_k_gating(z, False, vq_idx=idx_a)
        gates_b, _ = moe.noisy_top_k_gating(z, False, vq_idx=idx_b)
        self.assertFalse(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(gates_a.argmax(1), torch.zeros(6, dtype=torch.long)))
        self.assertTrue(torch.equal(gates_b.argmax(1), torch.ones(6, dtype=torch.long)))

    def test_code_bias_receives_gradient_updates_and_stage1_remains_frozen(self):
        model = build_model(True, seed=7).eval()
        feature = torch.randn(12, 5, 8)
        prior = torch.randn(12, 3)
        output = model(feature, prior)
        (output[0].sum() + output[4]).backward()
        table = routing_module(model).code_bias
        self.assertIsNotNone(table.grad)
        self.assertTrue(torch.isfinite(table.grad).all())
        self.assertGreater(table.grad.abs().sum().item(), 0.0)
        before = table.detach().clone()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3,
        )
        optimizer.step()
        self.assertFalse(torch.equal(before, table.detach()))
        for module in (model.encoder, model.quantizer, model.revin):
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)
                self.assertIsNone(parameter.grad, msg=name)

    def test_invalid_code_id_raises(self):
        moe = routing_module(build_model(True, seed=11)).eval()
        z = torch.randn(4, 8)
        for bad in (
            torch.tensor([-1, 0, 1, 2]),
            torch.tensor([0, 1, 2, 16]),
            torch.zeros(4, 1, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
        ):
            with self.assertRaises(ValueError):
                moe.clean_routing_logits(z, bad)
        with self.assertRaises(TypeError):
            moe.clean_routing_logits(z, torch.zeros(4))
        with self.assertRaises(ValueError):
            moe.clean_routing_logits(z, None)

    def test_code_id_does_not_bypass_router_into_other_paths(self):
        model = build_model(True, seed=99).eval()
        moe = routing_module(model)
        feature = torch.randn(7, 5, 8)
        prior = torch.randn(7, 3)
        seen = {}

        def capture(name):
            return lambda _module, inputs: seen.setdefault(name, tuple(inputs))

        handles = [
            model.loadings.temporal_transformer.register_forward_pre_hook(capture("temporal")),
            model.latent_value_head.register_forward_pre_hook(capture("latent_head")),
            model.return_predictor.register_forward_pre_hook(capture("return_predictor")),
            model.loadings.fusion.film_prior_head.register_forward_pre_hook(capture("film_prior")),
            model.loadings.fusion.film_latent_head.register_forward_pre_hook(capture("film_latent")),
            model.loadings.fusion.alpha_head.register_forward_pre_hook(capture("alpha")),
        ]
        with torch.no_grad():
            model(feature, prior)
        for handle in handles:
            handle.remove()

        int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        for name, inputs in seen.items():
            for tensor in inputs:
                if isinstance(tensor, torch.Tensor):
                    self.assertNotIn(tensor.dtype, int_dtypes, msg=name)

        # With the zero-initialized table, the routing-relevant input z_q is
        # untouched by vq_idx: permuting the code ids leaves the whole
        # LoadingGenerator output bitwise identical.
        z_q = torch.randn(7, 8)
        idx_a = torch.arange(7) % 16
        idx_b = (torch.arange(7) + 5) % 16
        with torch.no_grad():
            out_a = model.loadings(feature, z_q, vq_idx=idx_a)
            out_b = model.loadings(feature, z_q, vq_idx=idx_b)
        for left, right in zip(out_a, out_b):
            self.assertTrue(torch.equal(left, right))

        # The MoE gating itself is the only consumer of vq_idx.
        captured = {}
        original = FactorGatedMoE.clean_routing_logits

        def spy(self, x, vq_idx=None):
            captured["vq_idx"] = vq_idx
            return original(self, x, vq_idx=vq_idx)

        with mock.patch.object(FactorGatedMoE, "clean_routing_logits", spy):
            with torch.no_grad():
                model(feature, prior)
        self.assertIsNotNone(captured["vq_idx"])
        self.assertEqual(captured["vq_idx"].dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
