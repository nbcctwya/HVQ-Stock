"""Tests for experiment 036: 025 (010 Shared-Routed MoE + 019 Quantization
Confidence Adapter) + 017 Shared-Routed Decoupling.

The only new variable relative to 025 is the 017 decoupling regularizer:

    L_dec = mean(cosine_similarity(shared_out, routed_out)^2)
    L_moe = L_route + 0.01 * L_dec

The "025 base" inside these tests is the same code with
``predictor.decoupling_lambda: 0.0``: the decoupling term is the only code
difference between 036 and 025, so the disabled configuration is exactly the
025 behavior.
"""

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module.layers.fusion import HyperFusion
from module.layers.moe import FactorGatedMoE, SimpleMLP
from trainer.train_ypred import GenerateReturn, ReturnPredictor


def tiny_config(adapter=True, decoupling_lambda=0.01):
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
            "shared_expert": True,
            "decoupling_lambda": decoupling_lambda,
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
        "train": {"learning_rate": 0.0001},
    }


def build_model(adapter=True, decoupling_lambda=0.01, seed=0):
    torch.manual_seed(seed)
    config = copy.deepcopy(
        tiny_config(adapter=adapter, decoupling_lambda=decoupling_lambda)
    )
    with mock.patch.object(
        GenerateReturn,
        "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(config, T_max=10)


class DefaultConfigTests(unittest.TestCase):
    def test_default_config_represents_experiment_036(self):
        config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)

        predictor = config["predictor"]
        self.assertIs(predictor["shared_expert"], True)
        self.assertIs(predictor["quantization_confidence_adapter"], True)
        self.assertEqual(predictor["decoupling_lambda"], 0.01)
        self.assertEqual(predictor["n_expert"], 2)
        self.assertEqual(config["vqvae"]["vq_embed_dim"], 128)
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)


class Fidelity017Tests(unittest.TestCase):
    """The ported 017 decoupling must match its exact definition."""

    def setUp(self):
        torch.manual_seed(17)
        self.batch = 12
        self.input_dim = 8
        self.gate_dim = 6
        self.x = torch.randn(self.batch, self.input_dim)
        self.z = torch.randn(self.batch, self.gate_dim)

    def _make_moe(self, decoupling_lambda):
        return FactorGatedMoE(
            gate_input_size=self.gate_dim,
            expert_input_size=self.input_dim,
            hidden_size=self.input_dim,
            num_experts=3,
            noisy_gating=True,
            k=1,
            use_shared_expert=True,
            decoupling_lambda=decoupling_lambda,
        )

    @staticmethod
    def _make_shared_nonzero(moe):
        with torch.no_grad():
            moe.shared_expert.net[-1].weight.normal_(0.0, 0.1)
            moe.shared_expert.net[-1].bias.normal_(0.0, 0.1)

    def test_loss_matches_mean_squared_per_sample_cosine(self):
        shared = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -0.5, 2.0], [3.0, 0.0, 4.0]]
        )
        routed = torch.tensor(
            [[-1.0, 1.0, 2.0], [1.5, 0.5, -1.0], [0.0, 5.0, 0.0]]
        )
        expected_cosine = torch.sum(shared * routed, dim=-1) / (
            torch.linalg.vector_norm(shared, dim=-1)
            * torch.linalg.vector_norm(routed, dim=-1)
        )
        expected = expected_cosine.square().mean()
        actual = FactorGatedMoE.shared_routed_decoupling_loss(shared, routed)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-7))
        self.assertEqual(actual.ndim, 0)
        self.assertTrue(torch.isfinite(actual))
        self.assertGreaterEqual(actual.item(), 0.0)

    def test_loss_is_finite_for_zero_vectors(self):
        shared = torch.zeros(5, 4)
        routed = torch.randn(5, 4)
        penalty = FactorGatedMoE.shared_routed_decoupling_loss(shared, routed)
        self.assertTrue(torch.isfinite(penalty))
        self.assertEqual(penalty.item(), 0.0)

    def test_decoupling_loss_backpropagates_to_both_paths(self):
        shared = torch.randn(9, 7, requires_grad=True)
        routed = torch.randn(9, 7, requires_grad=True)
        penalty = FactorGatedMoE.shared_routed_decoupling_loss(shared, routed)
        penalty.backward()
        self.assertIsNotNone(shared.grad)
        self.assertIsNotNone(routed.grad)
        self.assertTrue(torch.isfinite(shared.grad).all())
        self.assertTrue(torch.isfinite(routed.grad).all())
        self.assertGreater(shared.grad.abs().sum().item(), 0.0)
        self.assertGreater(routed.grad.abs().sum().item(), 0.0)

    def test_total_moe_loss_adds_penalty_without_changing_route_loss(self):
        base = self._make_moe(decoupling_lambda=0.0).eval()
        self._make_shared_nonzero(base)
        experiment = copy.deepcopy(base)
        experiment.decoupling_lambda = 0.01

        base_output, route_loss = base(self.x, self.z)
        experiment_output, total_loss = experiment(self.x, self.z)
        self.assertTrue(torch.equal(base_output, experiment_output))

        routed_only = copy.deepcopy(base)
        routed_only.shared_expert = None
        routed_only.use_shared_expert = False
        routed_output, routed_only_loss = routed_only(self.x, self.z)
        shared_output = base.shared_expert(self.x)
        penalty = FactorGatedMoE.shared_routed_decoupling_loss(
            shared_output, routed_output
        )

        self.assertTrue(torch.equal(route_loss, routed_only_loss))
        self.assertTrue(
            torch.allclose(total_loss, route_loss + 0.01 * penalty, atol=1e-7, rtol=0)
        )

    def test_prediction_forward_is_bitwise_equal_with_decoupling(self):
        base = HyperFusion(
            d_h=8,
            d_z=6,
            k_prior=3,
            k_latent=6,
            drop=0.0,
            num_experts=3,
            moe_k=1,
            hidden_size=8,
            use_shared_expert=True,
            decoupling_lambda=0.0,
        ).eval()
        experiment = copy.deepcopy(base)
        experiment.moe.decoupling_lambda = 0.01
        self._make_shared_nonzero(base.moe)
        experiment.load_state_dict(base.state_dict(), strict=True)

        h = torch.randn(self.batch, 8)
        z = torch.randn(self.batch, 6)
        base_outputs = base(h, z)
        experiment_outputs = experiment(h, z)
        for base_tensor, experiment_tensor in zip(
            base_outputs[:3], experiment_outputs[:3]
        ):
            self.assertTrue(torch.equal(base_tensor, experiment_tensor))

        predictor = ReturnPredictor(3, 6, use_prior=True)
        prior = torch.randn(self.batch, 3)
        latent = torch.randn(self.batch, 6)
        base_prediction = predictor(*base_outputs[:3], prior, latent)
        experiment_prediction = predictor(*experiment_outputs[:3], prior, latent)
        self.assertTrue(torch.equal(base_prediction, experiment_prediction))

    def test_decoupling_adds_no_trainable_module(self):
        base = self._make_moe(0.0)
        experiment = self._make_moe(0.01)
        self.assertEqual(
            {name for name, _ in base.named_modules()},
            {name for name, _ in experiment.named_modules()},
        )
        self.assertEqual(
            {name for name, _ in base.named_parameters()},
            {name for name, _ in experiment.named_parameters()},
        )


class Integration025Tests(unittest.TestCase):
    """Decoupling must compose with the 025 quantization-confidence adapter
    without perturbing the 025 prediction path or the 010 structure."""

    def setUp(self):
        self.model = build_model(decoupling_lambda=0.01, seed=4321).eval()
        self.base = build_model(decoupling_lambda=0.0, seed=4321).eval()
        self.feature = torch.randn(9, 5, 8)
        self.prior = torch.randn(9, 3)

    def test_structure_inherited_from_025(self):
        moe = self.model.loadings.fusion.moe
        self.assertIsInstance(moe.shared_expert, SimpleMLP)
        self.assertNotIn(moe.shared_expert, list(moe.experts))
        self.assertEqual(len(moe.experts), 2)
        self.assertEqual(moe.k, 1)
        self.assertIsInstance(self.model.quantization_confidence_adapter, torch.nn.Linear)
        self.assertEqual(self.model.quantization_confidence_adapter.in_features, 1)
        self.assertEqual(self.model.quantization_confidence_adapter.out_features, 8)
        self.assertEqual(
            torch.count_nonzero(self.model.quantization_confidence_adapter.weight).item(),
            0,
        )

    def test_decoupling_lambda_flows_from_config_to_moe(self):
        self.assertEqual(self.model.loadings.fusion.moe.decoupling_lambda, 0.01)
        self.assertEqual(self.base.loadings.fusion.moe.decoupling_lambda, 0.0)

    def test_decoupling_adds_no_module_beyond_025(self):
        base_modules = {name for name, _ in self.base.named_modules()}
        experiment_modules = {name for name, _ in self.model.named_modules()}
        self.assertEqual(base_modules, experiment_modules)
        self.assertIn("quantization_confidence_adapter", experiment_modules)

    def test_existing_parameter_initialization_bitwise_equal_025(self):
        base_state = self.base.state_dict()
        experiment_state = self.model.state_dict()
        self.assertEqual(base_state.keys(), experiment_state.keys())
        for key, value in base_state.items():
            self.assertTrue(torch.equal(value, experiment_state[key]), msg=key)

    def test_full_prediction_forward_bitwise_equal_025(self):
        base_out = self.base(self.feature, self.prior)
        experiment_out = self.model(self.feature, self.prior)

        self.assertEqual(len(base_out), 5)
        self.assertEqual(len(experiment_out), 5)
        # The shared expert is zero-initialized, so L_dec == 0 at init and the
        # returned auxiliary loss is also bitwise equal.
        for index, (base_value, experiment_value) in enumerate(
            zip(base_out, experiment_out)
        ):
            self.assertTrue(
                torch.equal(base_value, experiment_value), msg=f"output[{index}]"
            )

    def test_aux_loss_contains_exactly_017_penalty_with_nonzero_shared(self):
        with torch.no_grad():
            shared_final = self.model.loadings.fusion.moe.shared_expert.net[-1]
            shared_final.weight.normal_(0.0, 0.1)
            shared_final.bias.normal_(0.0, 0.1)
        self.base.load_state_dict(self.model.state_dict(), strict=True)

        base_out = self.base(self.feature, self.prior)
        experiment_out = self.model(self.feature, self.prior)

        # Predictions must not change; only the auxiliary loss gains L_dec.
        for index in range(4):
            self.assertTrue(
                torch.equal(base_out[index], experiment_out[index]),
                msg=f"output[{index}]",
            )

        # Capture the raw (pre-softcap) MoE losses and both expert-path
        # representations to verify the closed form
        # L_moe = L_route + 0.01 * mean(cos(shared, routed)^2).
        seen = {"base": {}, "experiment": {}}

        def register(name, model):
            moe = model.loadings.fusion.moe
            handles = [
                moe.shared_expert.register_forward_hook(
                    lambda _m, _i, output, name=name: seen[name].setdefault(
                        "shared", output.detach().clone()
                    )
                ),
                moe.register_forward_hook(
                    lambda _m, _i, output, name=name: seen[name].update(
                        combined=output[0].detach().clone(),
                        moe_loss=output[1].detach().clone(),
                    )
                ),
            ]
            return handles

        handles = register("base", self.base) + register("experiment", self.model)
        self.base(self.feature, self.prior)
        self.model(self.feature, self.prior)
        for handle in handles:
            handle.remove()

        base_seen = seen["base"]
        experiment_seen = seen["experiment"]
        self.assertTrue(
            torch.equal(base_seen["combined"], experiment_seen["combined"])
        )

        shared_out = experiment_seen["shared"]
        routed_out = experiment_seen["combined"] - shared_out
        penalty = FactorGatedMoE.shared_routed_decoupling_loss(shared_out, routed_out)
        self.assertGreater(penalty.item(), 0.0)
        self.assertTrue(
            torch.allclose(
                experiment_seen["moe_loss"],
                base_seen["moe_loss"] + 0.01 * penalty,
                atol=1e-7,
                rtol=0,
            )
        )

    def test_decoupling_and_adapter_both_trainable_in_one_step(self):
        model = build_model(decoupling_lambda=0.01, seed=7)
        feature = torch.randn(11, 5, 8)
        prior = torch.randn(11, 3)
        label = torch.randn(11)

        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.01,
            weight_decay=0.0,
        )
        adapter = model.quantization_confidence_adapter
        shared_final = model.loadings.fusion.moe.shared_expert.net[-1]
        optimizer.zero_grad()
        y_pred, _, _, _, aux_loss = model(feature, prior)
        loss = model.rank_loss(y_pred, label) + model.aux_weight * aux_loss
        loss.backward()

        self.assertIsNotNone(adapter.weight.grad)
        self.assertGreater(adapter.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(shared_final.weight.grad)
        self.assertGreater(shared_final.weight.grad.abs().sum().item(), 0.0)
        for module in (model.encoder, model.quantizer, model.revin):
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)
                self.assertIsNone(parameter.grad, msg=name)

    def test_stage2_checkpoint_strict_round_trip(self):
        model = build_model(decoupling_lambda=0.01, seed=99).eval()
        with torch.no_grad():
            model.quantization_confidence_adapter.weight.normal_(0.0, 0.1)
            model.quantization_confidence_adapter.bias.normal_(0.0, 0.1)
            model.loadings.fusion.moe.shared_expert.net[-1].weight.normal_(0.0, 0.1)
        feature = torch.randn(7, 5, 8)
        prior = torch.randn(7, 3)
        reference = model(feature, prior)

        state = copy.deepcopy(model.state_dict())
        restored = build_model(decoupling_lambda=0.01, seed=123).eval()
        result = restored.load_state_dict(state, strict=True)
        self.assertFalse(result.missing_keys)
        self.assertFalse(result.unexpected_keys)

        reloaded = restored(feature, prior)
        for index, (reference_value, reloaded_value) in enumerate(
            zip(reference, reloaded)
        ):
            self.assertTrue(
                torch.equal(reference_value, reloaded_value), msg=f"output[{index}]"
            )


if __name__ == "__main__":
    unittest.main()
