import copy
import unittest
from pathlib import Path

import torch
import yaml

from module.layers.fusion import HyperFusion
from module.layers.moe import FactorGatedMoE
from trainer.train_ypred import ReturnPredictor


class SharedRoutedDecouplingTests(unittest.TestCase):
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

    def test_default_config_enables_fixed_decoupling_weight(self):
        config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)
        self.assertIs(config["predictor"]["shared_expert"], True)
        self.assertEqual(config["predictor"]["decoupling_lambda"], 0.01)
        self.assertEqual(config["train"]["seed"], 0)

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

    def test_identical_representation_penalty_exceeds_orthogonal_penalty(self):
        shared = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        identical = shared.clone()
        orthogonal = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
        identical_penalty = FactorGatedMoE.shared_routed_decoupling_loss(
            shared, identical
        )
        orthogonal_penalty = FactorGatedMoE.shared_routed_decoupling_loss(
            shared, orthogonal
        )
        self.assertTrue(torch.allclose(identical_penalty, torch.tensor(1.0)))
        self.assertLess(orthogonal_penalty.item(), 1e-12)
        self.assertGreater(identical_penalty.item(), orthogonal_penalty.item())

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

    def test_prediction_forward_is_bitwise_equal_to_010(self):
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


if __name__ == "__main__":
    unittest.main()
