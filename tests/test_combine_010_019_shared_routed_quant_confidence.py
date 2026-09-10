"""Tests for experiment 025: 010 Shared-Routed MoE + 019 Quantization
Confidence Adapter.

The "010 base" inside these tests is the same code with
``predictor.quantization_confidence_adapter: false``: the adapter is the only
code difference between 025 and 010, so the disabled configuration is exactly
the 010 behavior.
"""

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module.layers.moe import FactorGatedMoE, SimpleMLP
from trainer.train_ypred import GenerateReturn


def tiny_config(adapter=True, shared_expert=True):
    return {
        "vqvae": {
            "num_features": 8,
            "seq_len": 5,
            "hidden_size": 8,
            "num_prior_factors": 3,
            "vq_embed_dim": 8,
            "num_embed": 16,
            "encoder": {
                "num_heads": 2,
                "num_layers": 1,
                "market_gate": {
                    "input_dim": 3,
                    "beta": {"csi300": 10, "sp500": 5},
                },
            },
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
            "shared_expert": shared_expert,
            "dropout": 0.1,
            "rank": 0,
            "target_day": 2,
            "use_prior": True,
            "quantization_confidence_adapter": adapter,
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
        "data": {"universe": "csi300"},
        "train": {"learning_rate": 0.0001},
    }


def build_model(adapter=True, shared_expert=True, seed=0):
    torch.manual_seed(seed)
    config = copy.deepcopy(tiny_config(adapter=adapter, shared_expert=shared_expert))
    with mock.patch.object(
        GenerateReturn,
        "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(config, T_max=10)


def stage1_latents(model, feature, market):
    with torch.no_grad():
        feature_normalized = model.revin(feature, mode="norm")
        h_batch = model.encoder(feature_normalized, market)
        z_q = model.quantizer(h_batch)[0]
    return h_batch, z_q


class Inheritance010Tests(unittest.TestCase):
    """The 010 Shared-Routed MoE structure must be preserved unchanged."""

    def setUp(self):
        self.model = build_model().eval()
        self.moe = self.model.loadings.fusion.moe

    def test_default_config_represents_experiment_025(self):
        config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)

        predictor = config["predictor"]
        self.assertIs(predictor["shared_expert"], True)
        self.assertIs(predictor["quantization_confidence_adapter"], True)
        self.assertEqual(predictor["n_expert"], 2)
        self.assertEqual(config["vqvae"]["vq_embed_dim"], 128)
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_shared_expert_still_exists_and_is_always_on(self):
        moe = self.moe
        self.assertIsInstance(moe.shared_expert, SimpleMLP)
        self.assertEqual(repr(moe.shared_expert), repr(moe.experts[0]))

        x = torch.randn(9, 8)
        z = torch.randn(9, moe.gate_input_size)
        seen = []
        handle = moe.shared_expert.register_forward_hook(
            lambda _module, inputs, _output: seen.append(inputs[0].detach().clone())
        )
        moe(x, z)
        moe(x, -z)
        handle.remove()

        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.equal(seen[0], x))
        self.assertTrue(torch.equal(seen[1], x))

    def test_routed_path_configuration_unchanged(self):
        moe = self.moe
        self.assertEqual(moe.num_experts, 2)
        self.assertEqual(moe.k, 1)
        self.assertTrue(moe.noisy_gating)
        self.assertTrue(hasattr(moe, "gate"))
        self.assertTrue(hasattr(moe, "noise"))
        self.assertTrue(hasattr(moe, "W_h"))
        self.assertTrue(hasattr(moe, "softplus"))
        self.assertTrue(hasattr(moe, "mean"))
        self.assertTrue(hasattr(moe, "std"))
        # Shared Expert must not be part of the routed expert list and must
        # not consume a top-k quota.
        self.assertNotIn(moe.shared_expert, list(moe.experts))
        self.assertEqual(len(moe.experts), 2)

    def test_no_mechanism_beyond_010_plus_confidence_adapter(self):
        base = build_model(adapter=False, seed=11)
        adapted = build_model(adapter=True, seed=11)

        base_modules = {name for name, _ in base.named_modules()}
        adapted_modules = {name for name, _ in adapted.named_modules()}
        extra = adapted_modules - base_modules
        self.assertEqual(extra, {"quantization_confidence_adapter"})
        self.assertEqual(base_modules - adapted_modules, set())

    def test_auxiliary_loss_definition_unchanged(self):
        base = build_model(adapter=False, seed=13).eval()
        adapted = build_model(adapter=True, seed=13).eval()
        feature = torch.randn(7, 5, 8)
        prior = torch.randn(7, 3)
        market = torch.randn(7, 5, 3)

        base_out = base(feature, prior, market)
        adapted_out = adapted(feature, prior, market)

        self.assertTrue(torch.equal(base_out[4], adapted_out[4]))


class Fidelity019Tests(unittest.TestCase):
    """The ported 019 mechanism must match its exact definition."""

    def setUp(self):
        torch.manual_seed(7)
        self.h = torch.randn(6, 8)
        self.z_q = torch.randn(6, 8)

    def test_adapter_is_linear_1_to_latent_and_exactly_zero_initialized(self):
        model = build_model()
        adapter = model.quantization_confidence_adapter

        self.assertIsInstance(adapter, torch.nn.Linear)
        self.assertEqual(adapter.in_features, 1)
        self.assertEqual(adapter.out_features, 8)
        self.assertEqual(torch.count_nonzero(adapter.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.bias).item(), 0)

    def test_quantization_error_matches_required_definition(self):
        actual = GenerateReturn.quantization_error(self.h, self.z_q)
        expected = torch.mean((self.h - self.z_q) ** 2, dim=-1, keepdim=True)

        self.assertEqual(actual.shape, (6, 1))
        self.assertTrue(torch.equal(actual, expected))

    def test_quantization_error_is_detached_from_both_stage1_outputs(self):
        h = self.h.clone().requires_grad_()
        z_q = self.z_q.clone().requires_grad_()
        q_error = GenerateReturn.quantization_error(h, z_q)

        self.assertFalse(q_error.requires_grad)

    def test_z_conf_equals_z_q_plus_adapter_of_q_error(self):
        model = build_model().eval()
        with torch.no_grad():
            model.quantization_confidence_adapter.weight.fill_(0.5)
            model.quantization_confidence_adapter.bias.fill_(-0.25)

        z_stage2 = model.build_stage2_latent(self.h, self.z_q)
        q_error = torch.mean((self.h - self.z_q) ** 2, dim=-1, keepdim=True)
        expected = self.z_q + model.quantization_confidence_adapter(q_error)

        self.assertTrue(torch.equal(z_stage2, expected))
        self.assertFalse(torch.equal(z_stage2, self.z_q))

    def test_loadings_and_latent_value_head_receive_the_same_z_conf(self):
        model = build_model().eval()
        with torch.no_grad():
            model.quantization_confidence_adapter.bias.fill_(0.25)
        feature = torch.randn(7, 5, 8)
        prior = torch.randn(7, 3)
        market = torch.randn(7, 5, 3)
        seen = {"loadings": [], "latent_head": []}
        loadings_handle = model.loadings.register_forward_pre_hook(
            lambda _module, inputs: seen["loadings"].append(inputs[1].detach().clone())
        )
        latent_handle = model.latent_value_head.register_forward_pre_hook(
            lambda _module, inputs: seen["latent_head"].append(
                inputs[0].detach().clone()
            )
        )

        output = model(feature, prior, market)
        loadings_handle.remove()
        latent_handle.remove()

        self.assertEqual(len(output), 5)
        self.assertEqual(len(seen["loadings"]), 1)
        self.assertEqual(len(seen["latent_head"]), 1)
        self.assertTrue(torch.equal(seen["loadings"][0], output[3]))
        self.assertTrue(torch.equal(seen["latent_head"][0], output[3]))

    def test_router_and_hyperfusion_also_receive_z_conf(self):
        model = build_model().eval()
        with torch.no_grad():
            model.quantization_confidence_adapter.bias.fill_(0.25)
        feature = torch.randn(7, 5, 8)
        prior = torch.randn(7, 3)
        market = torch.randn(7, 5, 3)
        seen = {"moe": [], "fusion": [], "temporal": []}
        handles = [
            # HyperFusion calls the MoE with kwargs and LayerNorms the latent
            # first, so the router receives norm_z(z_conf).
            model.loadings.fusion.moe.register_forward_pre_hook(
                lambda _module, _inputs, kwargs: seen["moe"].append(
                    kwargs["z"].detach().clone()
                ),
                with_kwargs=True,
            ),
            model.loadings.fusion.register_forward_pre_hook(
                lambda _module, _inputs, kwargs: seen["fusion"].append(
                    kwargs["z"].detach().clone()
                ),
                with_kwargs=True,
            ),
            model.loadings.temporal_transformer.register_forward_pre_hook(
                lambda _module, inputs: seen["temporal"].append(
                    inputs[1].detach().clone()
                )
            ),
        ]

        output = model(feature, prior, market)
        for handle in handles:
            handle.remove()

        for key in ("fusion", "temporal"):
            self.assertEqual(len(seen[key]), 1, msg=key)
            self.assertTrue(torch.equal(seen[key][0], output[3]), msg=key)
        self.assertEqual(len(seen["moe"]), 1)
        with torch.no_grad():
            expected_router_latent = model.loadings.fusion.norm_z(output[3])
        self.assertTrue(torch.equal(seen["moe"][0], expected_router_latent))

    def test_confidence_path_does_not_backpropagate_into_stage1(self):
        model = build_model().eval()
        feature = torch.randn(11, 5, 8, requires_grad=True)
        prior = torch.randn(11, 3)
        market = torch.randn(11, 5, 3)
        encoder_outputs = []

        def capture_encoder_output(_module, _inputs, output):
            output.retain_grad()
            encoder_outputs.append(output)

        handle = model.encoder.register_forward_hook(capture_encoder_output)
        y_pred = model(feature, prior, market)[0]
        y_pred.sum().backward()
        handle.remove()

        self.assertEqual(len(encoder_outputs), 1)
        self.assertIsNone(encoder_outputs[0].grad)
        for module in (model.encoder, model.quantizer, model.revin):
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)
                self.assertIsNone(parameter.grad, msg=name)


class BaselineEquivalenceTests(unittest.TestCase):
    """With the zero-initialized adapter, 025 must be bitwise equal to 010."""

    def setUp(self):
        self.base = build_model(adapter=False, seed=4321).eval()
        self.adapted = build_model(adapter=True, seed=4321).eval()
        self.feature = torch.randn(9, 5, 8)
        self.prior = torch.randn(9, 3)
        self.market = torch.randn(9, 5, 3)

    def test_existing_parameter_initialization_bitwise_equal_010(self):
        adapted_base_state = {
            key: value
            for key, value in self.adapted.state_dict().items()
            if not key.startswith("quantization_confidence_adapter.")
        }

        self.assertEqual(self.base.state_dict().keys(), adapted_base_state.keys())
        for key, value in self.base.state_dict().items():
            self.assertTrue(torch.equal(value, adapted_base_state[key]), msg=key)

    def test_zero_init_stage2_latent_bitwise_equal_z_q(self):
        h_batch, z_q = stage1_latents(self.adapted, self.feature, self.market)
        z_stage2 = self.adapted.build_stage2_latent(h_batch, z_q)

        self.assertTrue(torch.equal(z_stage2, z_q))

    def test_temporal_transformer_latent_input_bitwise_equal_010(self):
        seen = {"base": [], "adapted": []}
        handles = [
            self.base.loadings.temporal_transformer.register_forward_pre_hook(
                lambda _module, inputs: seen["base"].append(inputs[1].detach().clone())
            ),
            self.adapted.loadings.temporal_transformer.register_forward_pre_hook(
                lambda _module, inputs: seen["adapted"].append(
                    inputs[1].detach().clone()
                )
            ),
        ]
        self.base(self.feature, self.prior, self.market)
        self.adapted(self.feature, self.prior, self.market)
        for handle in handles:
            handle.remove()

        self.assertTrue(torch.equal(seen["base"][0], seen["adapted"][0]))

    def test_hyperfusion_and_routing_bitwise_equal_010(self):
        fusion_in = {"base": [], "adapted": []}
        moe_out = {"base": [], "adapted": []}
        handles = []
        for name, model in (("base", self.base), ("adapted", self.adapted)):
            handles.append(
                model.loadings.fusion.register_forward_pre_hook(
                    lambda _module, _inputs, kwargs, name=name: fusion_in[name].append(
                        kwargs["z"].detach().clone()
                    ),
                    with_kwargs=True,
                )
            )
            handles.append(
                model.loadings.fusion.moe.register_forward_hook(
                    lambda _module, _inputs, output, name=name: moe_out[name].append(
                        (output[0].detach().clone(), output[1].detach().clone())
                    )
                )
            )

        self.base(self.feature, self.prior, self.market)
        self.adapted(self.feature, self.prior, self.market)
        for handle in handles:
            handle.remove()

        self.assertTrue(torch.equal(fusion_in["base"][0], fusion_in["adapted"][0]))
        self.assertTrue(torch.equal(moe_out["base"][0][0], moe_out["adapted"][0][0]))
        self.assertTrue(torch.equal(moe_out["base"][0][1], moe_out["adapted"][0][1]))

    def test_shared_expert_input_output_bitwise_equal_010(self):
        seen = {"base": [], "adapted": []}
        outputs = {"base": [], "adapted": []}
        handles = []
        for name, model in (("base", self.base), ("adapted", self.adapted)):
            shared = model.loadings.fusion.moe.shared_expert
            handles.append(
                shared.register_forward_pre_hook(
                    lambda _module, inputs, name=name: seen[name].append(
                        inputs[0].detach().clone()
                    )
                )
            )
            handles.append(
                shared.register_forward_hook(
                    lambda _module, _inputs, output, name=name: outputs[name].append(
                        output.detach().clone()
                    )
                )
            )

        self.base(self.feature, self.prior, self.market)
        self.adapted(self.feature, self.prior, self.market)
        for handle in handles:
            handle.remove()

        self.assertTrue(torch.equal(seen["base"][0], seen["adapted"][0]))
        self.assertTrue(torch.equal(outputs["base"][0], outputs["adapted"][0]))
        # Zero-init final Linear -> shared output must be exactly zero.
        self.assertEqual(torch.count_nonzero(outputs["adapted"][0]).item(), 0)

    def test_latent_value_head_output_bitwise_equal_010(self):
        seen = {"base": [], "adapted": []}
        handles = []
        for name, model in (("base", self.base), ("adapted", self.adapted)):
            handles.append(
                model.latent_value_head.register_forward_hook(
                    lambda _module, _inputs, output, name=name: seen[name].append(
                        output.detach().clone()
                    )
                )
            )

        self.base(self.feature, self.prior, self.market)
        self.adapted(self.feature, self.prior, self.market)
        for handle in handles:
            handle.remove()

        self.assertTrue(torch.equal(seen["base"][0], seen["adapted"][0]))

    def test_full_prediction_forward_bitwise_equal_010(self):
        base_out = self.base(self.feature, self.prior, self.market)
        adapted_out = self.adapted(self.feature, self.prior, self.market)

        self.assertEqual(len(base_out), 5)
        self.assertEqual(len(adapted_out), 5)
        for index, (base_value, adapted_value) in enumerate(
            zip(base_out, adapted_out)
        ):
            self.assertTrue(
                torch.equal(base_value, adapted_value), msg=f"output[{index}]"
            )


class TrainabilityTests(unittest.TestCase):
    def setUp(self):
        self.model = build_model().eval()
        self.feature = torch.randn(11, 5, 8)
        self.prior = torch.randn(11, 3)
        self.market = torch.randn(11, 5, 3)

    def _train_one_step(self):
        model = self.model
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.01,
            weight_decay=0.0,
        )
        label = torch.randn(11)
        y_pred, _, _, _, aux_loss = model(self.feature, self.prior, self.market)
        loss = model.rank_loss(y_pred, label) + model.aux_weight * aux_loss
        optimizer.zero_grad()
        loss.backward()
        return model, optimizer, loss

    def test_adapter_gets_finite_nonzero_gradient_and_updates(self):
        model, optimizer, _ = self._train_one_step()
        adapter = model.quantization_confidence_adapter
        before_weight = adapter.weight.detach().clone()
        before_bias = adapter.bias.detach().clone()

        self.assertIsNotNone(adapter.weight.grad)
        self.assertIsNotNone(adapter.bias.grad)
        self.assertTrue(torch.isfinite(adapter.weight.grad).all())
        self.assertTrue(torch.isfinite(adapter.bias.grad).all())
        self.assertGreater(adapter.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(adapter.bias.grad.abs().sum().item(), 0.0)

        optimizer.step()
        self.assertFalse(torch.equal(adapter.weight.detach(), before_weight))
        self.assertFalse(torch.equal(adapter.bias.detach(), before_bias))

    def test_stage1_stays_frozen_eval_and_codebook_unchanged(self):
        model = self.model
        codebook_before = model.quantizer.embedding.weight.detach().clone()
        self._train_one_step()

        for module in (model.encoder, model.quantizer, model.revin):
            self.assertFalse(module.training)
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)
                self.assertIsNone(parameter.grad, msg=name)
        self.assertTrue(
            torch.equal(codebook_before, model.quantizer.embedding.weight.detach())
        )

    def test_shared_and_routed_experts_still_receive_training_signal(self):
        model, optimizer, _ = self._train_one_step()
        moe = model.loadings.fusion.moe

        shared_final = moe.shared_expert.net[-1]
        self.assertIsNotNone(shared_final.weight.grad)
        self.assertGreater(shared_final.weight.grad.abs().sum().item(), 0.0)
        routed_grads = [
            parameter.grad
            for parameter in moe.experts.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(routed_grads)
        self.assertTrue(
            any(grad.abs().sum().item() > 0.0 for grad in routed_grads)
        )

    def test_z_conf_minus_z_q_nonzero_after_adapter_update(self):
        model, optimizer, _ = self._train_one_step()
        optimizer.step()
        model.eval()

        h_batch, z_q = stage1_latents(model, self.feature, self.market)
        z_stage2 = model.build_stage2_latent(h_batch, z_q)

        self.assertGreater((z_stage2 - z_q).abs().sum().item(), 0.0)

    def test_quantizer_assignment_unchanged_by_adapter_path(self):
        model = self.model
        h_batch, z_q_before = stage1_latents(model, self.feature, self.market)
        with torch.no_grad():
            _, _, (_, _, vq_idx_before) = model.quantizer(h_batch)

        _, optimizer, _ = self._train_one_step()
        optimizer.step()
        model.eval()

        h_after, z_q_after = stage1_latents(model, self.feature, self.market)
        with torch.no_grad():
            _, _, (_, _, vq_idx_after) = model.quantizer(h_after)

        self.assertTrue(torch.equal(h_batch, h_after))
        self.assertTrue(torch.equal(z_q_before, z_q_after))
        self.assertTrue(torch.equal(vq_idx_before, vq_idx_after))


class CompatibilityTests(unittest.TestCase):
    def test_stage2_checkpoint_strict_round_trip(self):
        model = build_model().eval()
        with torch.no_grad():
            model.quantization_confidence_adapter.weight.normal_(0.0, 0.1)
            model.quantization_confidence_adapter.bias.normal_(0.0, 0.1)
        feature = torch.randn(7, 5, 8)
        prior = torch.randn(7, 3)
        market = torch.randn(7, 5, 3)
        reference = model(feature, prior, market)

        state = copy.deepcopy(model.state_dict())
        restored = build_model().eval()
        result = restored.load_state_dict(state, strict=True)
        self.assertFalse(result.missing_keys)
        self.assertFalse(result.unexpected_keys)

        reloaded = restored(feature, prior, market)
        self.assertEqual(len(reloaded), 5)
        for index, (reference_value, reloaded_value) in enumerate(
            zip(reference, reloaded)
        ):
            self.assertTrue(
                torch.equal(reference_value, reloaded_value), msg=f"output[{index}]"
            )


if __name__ == "__main__":
    unittest.main()
