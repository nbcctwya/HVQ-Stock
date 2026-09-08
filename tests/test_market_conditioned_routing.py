"""Regression tests for experiment 020 market-conditioned expert routing."""

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.schema import GROUP_SLICES, TOTAL_DIM, unpack_batch
from module.layers.moe import FactorGatedMoE
from trainer.train_ypred import GenerateReturn


def tiny_config(market_routing=True):
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
            "market_conditioned_routing": market_routing,
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


def build_model(market_routing=True, seed=0):
    torch.manual_seed(seed)
    config = copy.deepcopy(tiny_config(market_routing))
    with mock.patch.object(
        GenerateReturn,
        "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(config, T_max=10)


def routing_module(model):
    return model.loadings.fusion.moe


class MarketConditionedRoutingTest(unittest.TestCase):
    def test_default_config_enables_only_required_adapter(self):
        with (Path(__file__).parents[1] / "configs" / "config.yaml").open() as stream:
            config = yaml.safe_load(stream)
        self.assertIs(config["predictor"]["market_conditioned_routing"], True)
        self.assertEqual(config["predictor"]["n_expert"], 2)
        self.assertEqual(config["predictor"]["k"], "${half:${predictor.n_expert}}")
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_canonical_market_shape_and_latest_timestep_extraction(self):
        batch = torch.arange(3 * 20 * TOTAL_DIM).reshape(3, 20, TOTAL_DIM).float()
        parts = unpack_batch(batch)
        latest = GenerateReturn.current_market_state(parts.market_feature)
        self.assertEqual(parts.market_feature.shape, (3, 20, 63))
        self.assertEqual(latest.shape, (3, 63))
        self.assertTrue(torch.equal(latest, batch[:, -1, GROUP_SLICES["market"]]))
        for bad in (torch.zeros(3, 63), torch.zeros(3, 0, 63), torch.zeros(3, 20, 62)):
            with self.assertRaises(ValueError):
                GenerateReturn.current_market_state(bad)

    def test_adapter_is_bias_free_linear_63_to_experts_and_zero_initialized(self):
        moe = routing_module(build_model())
        adapter = moe.market_routing_adapter
        self.assertIsInstance(adapter, torch.nn.Linear)
        self.assertEqual((adapter.in_features, adapter.out_features), (63, 2))
        self.assertIsNone(adapter.bias)
        self.assertEqual(torch.count_nonzero(adapter.weight).item(), 0)
        self.assertEqual(list(moe.normalize_market(torch.randn(4, 63)).shape), [4, 63])
        self.assertEqual(sum(p.numel() for p in moe.parameters() if p is adapter.weight), 126)

    def test_adapter_does_not_perturb_any_existing_state_initialization(self):
        base = build_model(False, seed=1234)
        adapted = build_model(True, seed=1234)
        prefix = "loadings.fusion.moe.market_routing_adapter."
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
        market = torch.randn(9, 63)

        self.assertTrue(torch.equal(
            base_moe.clean_routing_logits(z),
            adapted_moe.clean_routing_logits(z, market),
        ))
        base_gates, base_load = base_moe.noisy_top_k_gating(z, False)
        adapted_gates, adapted_load = adapted_moe.noisy_top_k_gating(
            z, False, market_state=market
        )
        self.assertTrue(torch.equal(base_gates, adapted_gates))
        self.assertTrue(torch.equal(base_load, adapted_load))
        base_y, base_aux = base_moe(x, z)
        adapted_y, adapted_aux = adapted_moe(x, z, market_state=market)
        self.assertTrue(torch.equal(base_y, adapted_y))
        self.assertTrue(torch.equal(base_aux, adapted_aux))

        feature = torch.randn(9, 5, 8)
        prior = torch.randn(9, 3)
        market_sequence = torch.randn(9, 5, 63)
        base_output = base(feature, prior)
        adapted_output = adapted(feature, prior, market_sequence)
        for index, (base_value, adapted_value) in enumerate(zip(base_output, adapted_output)):
            self.assertTrue(torch.equal(base_value, adapted_value), msg=f"output[{index}]")

    def test_original_noisy_topk_noise_wh_and_load_behavior_is_unchanged(self):
        base = routing_module(build_model(False, seed=88)).train()
        adapted = routing_module(build_model(True, seed=88)).train()
        z = torch.randn(32, 8)
        market = torch.randn(32, 63)
        self.assertTrue(torch.equal(base.W_h, adapted.W_h))
        for left, right in zip(base.gate.state_dict().values(), adapted.gate.state_dict().values()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(base.noise.state_dict().values(), adapted.noise.state_dict().values()):
            self.assertTrue(torch.equal(left, right))
        torch.manual_seed(2020)
        base_gates, base_load = base.noisy_top_k_gating(z, True)
        torch.manual_seed(2020)
        adapted_gates, adapted_load = adapted.noisy_top_k_gating(
            z, True, market_state=market
        )
        self.assertTrue(torch.equal(base_gates, adapted_gates))
        self.assertTrue(torch.equal(base_load, adapted_load))

    def test_nonzero_adapter_changes_logits_and_expert_allocation_for_fixed_z(self):
        moe = FactorGatedMoE(
            gate_input_size=8,
            expert_input_size=8,
            hidden_size=8,
            num_experts=2,
            k=1,
            market_conditioned_routing=True,
        ).eval()
        for parameter in moe.gate.parameters():
            torch.nn.init.zeros_(parameter)
        direction = torch.linspace(-2.0, 2.0, 63)
        with torch.no_grad():
            moe.market_routing_adapter.weight[0].copy_(direction)
            moe.market_routing_adapter.weight[1].copy_(-direction)
        z = torch.zeros(6, 8)
        market_a = direction.expand(6, -1)
        market_b = -direction.expand(6, -1)
        logits_a = moe.clean_routing_logits(z, market_a)
        logits_b = moe.clean_routing_logits(z, market_b)
        gates_a, _ = moe.noisy_top_k_gating(z, False, market_state=market_a)
        gates_b, _ = moe.noisy_top_k_gating(z, False, market_state=market_b)
        self.assertFalse(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(gates_a.argmax(1), torch.zeros(6, dtype=torch.long)))
        self.assertTrue(torch.equal(gates_b.argmax(1), torch.ones(6, dtype=torch.long)))

    def test_adapter_receives_gradient_and_stage1_remains_frozen(self):
        model = build_model(True, seed=7).eval()
        feature = torch.randn(12, 5, 8)
        prior = torch.randn(12, 3)
        market = torch.randn(12, 5, 63)
        output = model(feature, prior, market)
        (output[0].sum() + output[4]).backward()
        gradient = routing_module(model).market_routing_adapter.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        for module in (model.encoder, model.quantizer, model.revin):
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)
                self.assertIsNone(parameter.grad, msg=name)

    def test_market_reaches_only_router_adapter_and_only_latest_timestep_matters(self):
        model = build_model(True, seed=99).eval()
        moe = routing_module(model)
        feature = torch.randn(7, 5, 8)
        prior = torch.randn(7, 3)
        market = torch.randn(7, 5, 63)
        changed_history = market.clone()
        changed_history[:, :-1] = torch.randn_like(changed_history[:, :-1]) * 1000
        seen = {}

        def capture(name):
            return lambda _module, inputs: seen.setdefault(name, tuple(inputs))

        handles = [
            moe.market_routing_adapter.register_forward_pre_hook(capture("adapter")),
            model.loadings.temporal_transformer.register_forward_pre_hook(capture("temporal")),
            model.latent_value_head.register_forward_pre_hook(capture("latent_head")),
        ]
        first = model(feature, prior, market)
        second = model(feature, prior, changed_history)
        for handle in handles:
            handle.remove()

        expected = moe.normalize_market(market[:, -1, :])
        self.assertTrue(torch.equal(seen["adapter"][0], expected))
        self.assertEqual(len(seen["temporal"]), 2)
        self.assertEqual(len(seen["latent_head"]), 1)
        for inputs in (seen["temporal"], seen["latent_head"]):
            self.assertFalse(any(tensor.shape[-1] == 63 for tensor in inputs))
        for first_value, second_value in zip(first, second):
            self.assertTrue(torch.equal(first_value, second_value))


if __name__ == "__main__":
    unittest.main()
