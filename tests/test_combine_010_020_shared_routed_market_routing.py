"""Regression tests for experiment 026: 010 Shared-Routed MoE combined with
020 market-conditioned routing on the routed clean logits.

The base behavior is experiment 010 (``shared_expert=True``,
``market_conditioned_routing=False``); the experiment model only adds the
020 zero-initialized ``Linear(63, n_expert, bias=False)`` market bias to the
routed clean logits before noise / W_h / top-k.
"""

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
            "shared_expert": True,
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


class Combine010InheritanceTest(unittest.TestCase):
    """The 010 Shared-Routed structure must be fully preserved."""

    def test_default_config_enables_shared_expert_and_market_routing(self):
        with (Path(__file__).parents[1] / "configs" / "config.yaml").open() as stream:
            config = yaml.safe_load(stream)
        self.assertIs(config["predictor"]["shared_expert"], True)
        self.assertIs(config["predictor"]["market_conditioned_routing"], True)
        self.assertEqual(config["predictor"]["n_expert"], 2)
        self.assertEqual(config["predictor"]["k"], "${half:${predictor.n_expert}}")
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_shared_expert_always_on_outside_routing_and_topk(self):
        moe = routing_module(build_model())
        self.assertIsNotNone(moe.shared_expert)
        self.assertEqual(moe.num_experts, 2)
        self.assertEqual(moe.k, 1)
        self.assertEqual(len(moe.experts), 2)
        # Shared expert is not part of the routed expert list seen by the
        # dispatcher and does not consume top-k quota.
        self.assertNotIn(moe.shared_expert, list(moe.experts))

    def test_moe_out_remains_shared_plus_routed(self):
        moe = routing_module(build_model()).eval()
        x = torch.randn(9, 8)
        z = torch.randn(9, 8)
        market = torch.randn(9, 63)
        moe_out, _ = moe(x, z, market_state=market)
        shared_out = moe.shared_expert(x)
        self.assertEqual(moe_out.shape, shared_out.shape)
        # routed_out = moe_out - shared_out must be reproducible from the
        # dispatcher path alone (shared expert zero-init => routed_out here).
        routed_moe = FactorGatedMoE(
            gate_input_size=8,
            expert_input_size=8,
            hidden_size=8,
            num_experts=2,
            noisy_gating=True,
            k=1,
            use_shared_expert=False,
        )
        routed_state = {
            key: value
            for key, value in moe.state_dict().items()
            if not key.startswith("shared_expert.")
            and not key.startswith("market_routing_adapter.")
        }
        routed_moe.load_state_dict(routed_state, strict=True)
        routed_moe.eval()
        routed_out, routed_loss = routed_moe(x, z)
        self.assertTrue(torch.equal(moe_out, shared_out + routed_out))

    def test_no_other_experiment_mechanism_leaks_in(self):
        model = build_model()
        state_keys = set(model.state_dict().keys())
        for forbidden in ("z_conf", "confidence", "code_bias", "transition"):
            self.assertFalse(
                any(forbidden in key for key in state_keys), msg=forbidden
            )
        # 026 state = 010 state + exactly one new adapter parameter tensor.
        base_keys = set(build_model(False).state_dict().keys())
        new_keys = state_keys - base_keys
        self.assertEqual(
            new_keys, {"loadings.fusion.moe.market_routing_adapter.weight"}
        )


class MarketRoutingMechanismFidelityTest(unittest.TestCase):
    """020 mechanism must be inherited exactly."""

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

    def test_market_normalization_is_parameter_free_layer_norm(self):
        moe = routing_module(build_model())
        market = torch.randn(5, 63)
        normalized = moe.normalize_market(market)
        expected = torch.nn.functional.layer_norm(market, (63,))
        self.assertTrue(torch.equal(normalized, expected))
        # Parameter-free: no affine parameters are involved anywhere.
        self.assertEqual(
            [name for name, _ in moe.named_parameters() if "norm" in name.lower()], []
        )

    def test_market_bias_shape_and_clean_logits_composition(self):
        moe = routing_module(build_model()).eval()
        z = torch.randn(7, 8)
        market = torch.randn(7, 63)
        with torch.no_grad():
            moe.market_routing_adapter.weight.normal_(0, 0.1)
            bias = moe.market_routing_adapter(moe.normalize_market(market))
            self.assertEqual(bias.shape, (7, 2))
            clean = moe.clean_routing_logits(z, market)
            self.assertTrue(torch.equal(clean, moe.gate(z) + bias))

    def test_market_bias_added_before_noise_wh_and_topk(self):
        moe = routing_module(build_model()).train()
        z = torch.randn(16, 8)
        market = torch.randn(16, 63)
        with torch.no_grad():
            moe.market_routing_adapter.weight.normal_(0, 0.5)
        torch.manual_seed(2020)
        gates, _ = moe.noisy_top_k_gating(z, True, market_state=market)
        # Recompute manually: bias must enter clean logits before noise/W_h.
        torch.manual_seed(2020)
        clean = moe.gate(z) + moe.market_routing_adapter(moe.normalize_market(market))
        raw_noise_stddev = moe.noise(z)
        noise_stddev = moe.softplus(raw_noise_stddev) + 1e-2
        noise = torch.randn_like(clean)
        noisy_logits = clean + noise * noise_stddev
        logits = moe.softmax(noisy_logits @ moe.W_h)
        top_logits, top_indices = logits.topk(min(moe.k + 1, moe.num_experts), dim=1)
        top_k_gates = top_logits[:, : moe.k] / (top_logits[:, : moe.k].sum(1, keepdim=True) + 1e-6)
        expected = torch.zeros_like(logits).scatter(1, top_indices[:, : moe.k], top_k_gates)
        self.assertTrue(torch.equal(gates, expected))


class BaselineEquivalenceTest(unittest.TestCase):
    """Zero-init 026 must be bitwise identical to 010 everywhere."""

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

    def test_zero_init_clean_logits_routing_aux_and_full_forward_equal_010(self):
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
        # Shared expert output is untouched by the market path.
        self.assertTrue(torch.equal(
            base_moe.shared_expert(x), adapted_moe.shared_expert(x)
        ))

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


class MarketRoutingSemanticsTest(unittest.TestCase):
    def test_nonzero_adapter_changes_logits_and_expert_allocation_for_fixed_z(self):
        moe = FactorGatedMoE(
            gate_input_size=8,
            expert_input_size=8,
            hidden_size=8,
            num_experts=2,
            k=1,
            use_shared_expert=True,
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
        # Same market + same z => identical clean logits.
        self.assertTrue(torch.equal(logits_a, moe.clean_routing_logits(z, market_a.clone())))

    def test_invalid_market_inputs_fail_loudly(self):
        moe = routing_module(build_model())
        z = torch.randn(4, 8)
        with self.assertRaises(ValueError):
            moe.clean_routing_logits(z, None)
        with self.assertRaises(ValueError):
            moe.clean_routing_logits(z, torch.randn(4, 62))
        with self.assertRaises(ValueError):
            moe.clean_routing_logits(z, torch.randn(4, 5, 63))
        with self.assertRaises(ValueError):
            moe.clean_routing_logits(z, torch.randn(3, 63))

        model = build_model()
        with self.assertRaises(ValueError):
            model(torch.randn(4, 5, 8), torch.randn(4, 3), None)
        with self.assertRaises(ValueError):
            model(torch.randn(4, 5, 8), torch.randn(4, 3), torch.randn(3, 5, 63))
        with self.assertRaises(ValueError):
            model(torch.randn(4, 5, 8), torch.randn(4, 3), torch.randn(4, 5, 62))

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
            moe.shared_expert.register_forward_pre_hook(capture("shared")),
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
        for inputs in (seen["shared"], seen["temporal"], seen["latent_head"]):
            self.assertFalse(any(tensor.shape[-1] == 63 for tensor in inputs))
        for first_value, second_value in zip(first, second):
            self.assertTrue(torch.equal(first_value, second_value))


class TrainabilityTest(unittest.TestCase):
    def test_adapter_shared_and_routed_all_receive_gradient_and_stage1_frozen(self):
        model = build_model(True, seed=7).eval()
        feature = torch.randn(12, 5, 8)
        prior = torch.randn(12, 3)
        market = torch.randn(12, 5, 63)
        output = model(feature, prior, market)
        (output[0].sum() + output[4]).backward()
        moe = routing_module(model)
        adapter_grad = moe.market_routing_adapter.weight.grad
        self.assertIsNotNone(adapter_grad)
        self.assertTrue(torch.isfinite(adapter_grad).all())
        self.assertGreater(adapter_grad.abs().sum().item(), 0.0)

        shared_grad = moe.shared_expert.net[-1].weight.grad
        self.assertIsNotNone(shared_grad)
        self.assertGreater(shared_grad.abs().sum().item(), 0.0)
        routed_grads = [
            parameter.grad
            for name, parameter in moe.named_parameters()
            if name.startswith("experts.")
        ]
        # top-k=1 routing may leave one expert unused in a small batch; the
        # routed path as a whole must still receive a training signal.
        active_routed_grads = [grad for grad in routed_grads if grad is not None]
        self.assertTrue(active_routed_grads)
        self.assertGreater(
            sum(grad.abs().sum().item() for grad in active_routed_grads), 0.0
        )

        for module in (model.encoder, model.quantizer, model.revin):
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)
                self.assertIsNone(parameter.grad, msg=name)
            self.assertFalse(module.training)

    def test_market_routing_does_not_change_quantizer_assignment(self):
        model = build_model(True, seed=11).eval()
        feature = torch.randn(8, 5, 8)
        prior = torch.randn(8, 3)
        market_a = torch.randn(8, 5, 63)
        market_b = torch.randn(8, 5, 63)
        with torch.no_grad():
            routing_module(model).market_routing_adapter.weight.normal_(0, 0.5)
            z_q_a = model(feature, prior, market_a)[3]
            z_q_b = model(feature, prior, market_b)[3]
        self.assertTrue(torch.equal(z_q_a, z_q_b))


class NoLeakageTest(unittest.TestCase):
    def test_label_poison_does_not_change_market_state_or_bias(self):
        batch = torch.randn(5, 20, TOTAL_DIM)
        poisoned = batch.clone()
        poisoned[:, :, GROUP_SLICES["label"]] = 1e6
        parts = unpack_batch(batch)
        parts_poisoned = unpack_batch(poisoned)
        self.assertFalse(torch.equal(parts.future_returns, parts_poisoned.future_returns))
        self.assertTrue(torch.equal(parts.market_feature, parts_poisoned.market_feature))

        moe = routing_module(build_model(True, seed=5)).eval()
        with torch.no_grad():
            moe.market_routing_adapter.weight.normal_(0, 0.1)
        state = GenerateReturn.current_market_state(parts.market_feature)
        state_poisoned = GenerateReturn.current_market_state(parts_poisoned.market_feature)
        self.assertTrue(torch.equal(state, state_poisoned))
        bias = moe.market_routing_adapter(moe.normalize_market(state))
        bias_poisoned = moe.market_routing_adapter(moe.normalize_market(state_poisoned))
        self.assertTrue(torch.equal(bias, bias_poisoned))

    def test_market_state_reads_only_canonical_market_fields(self):
        batch = torch.randn(4, 20, TOTAL_DIM)
        parts = unpack_batch(batch)
        state = GenerateReturn.current_market_state(parts.market_feature)
        self.assertTrue(torch.equal(state, batch[:, -1, GROUP_SLICES["market"]]))
        # Mutating every non-market field cannot change the market state.
        masked = batch.clone()
        market_mask = torch.zeros(TOTAL_DIM, dtype=torch.bool)
        market_mask[GROUP_SLICES["market"]] = True
        masked[:, :, ~market_mask] = -999.0
        masked_state = GenerateReturn.current_market_state(
            unpack_batch(masked).market_feature
        )
        self.assertTrue(torch.equal(state, masked_state))


if __name__ == "__main__":
    unittest.main()
