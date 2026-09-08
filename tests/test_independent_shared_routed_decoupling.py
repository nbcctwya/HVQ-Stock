import copy
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from module.layers.fusion import HyperFusion
from module.layers.moe import FactorGatedMoE
from trainer.train_ypred import GenerateReturn, softcap_log1p


class IndependentSharedRoutedDecouplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.batch = 12
        self.input_dim = 8
        self.gate_dim = 6
        self.x = torch.randn(self.batch, self.input_dim)
        self.z = torch.randn(self.batch, self.gate_dim)

    def _make_moe(self):
        return FactorGatedMoE(
            gate_input_size=self.gate_dim,
            expert_input_size=self.input_dim,
            hidden_size=self.input_dim,
            num_experts=3,
            noisy_gating=True,
            k=1,
            use_shared_expert=True,
            use_adaptive_shared_fusion=True,
        )

    @staticmethod
    def _make_shared_nonzero(moe):
        with torch.no_grad():
            moe.shared_expert.net[-1].weight.normal_(0.0, 0.1)
            moe.shared_expert.net[-1].bias.normal_(0.0, 0.1)

    def test_default_config_is_complete_and_keeps_016_protocol(self):
        config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)

        self.assertIs(config["predictor"]["shared_expert"], True)
        self.assertIs(config["predictor"]["adaptive_shared_fusion"], True)
        self.assertEqual(config["predictor"]["decoupling_lambda"], 0.01)
        self.assertEqual(config["predictor"]["aux_weight"], 0.01)
        self.assertEqual(config["predictor"]["aux_imp"], 3)
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_loss_matches_stable_mean_squared_per_sample_cosine(self):
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

        zero_penalty = FactorGatedMoE.shared_routed_decoupling_loss(
            torch.zeros(5, 4), torch.randn(5, 4)
        )
        self.assertTrue(torch.isfinite(zero_penalty))
        self.assertEqual(zero_penalty.item(), 0.0)

    def test_identical_penalty_is_high_and_orthogonal_penalty_is_low(self):
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

    def test_decoupling_loss_backpropagates_to_both_raw_paths(self):
        shared = torch.randn(9, 7, requires_grad=True)
        routed = torch.randn(9, 7, requires_grad=True)
        penalty = FactorGatedMoE.shared_routed_decoupling_loss(shared, routed)
        penalty.backward()

        for representation in (shared, routed):
            self.assertIsNotNone(representation.grad)
            self.assertTrue(torch.isfinite(representation.grad).all())
            self.assertGreater(representation.grad.abs().sum().item(), 0.0)

    def test_penalty_uses_raw_shared_output_before_adaptive_scaling(self):
        moe = self._make_moe().eval()
        self._make_shared_nonzero(moe)
        with torch.no_grad():
            moe.shared_fusion.weight.fill_(0.7)
            moe.shared_fusion.bias.fill_(-0.15)

        output, route_loss, penalty = moe(
            self.x,
            self.z,
            shared_condition=self.z,
            return_decoupling_loss=True,
        )
        raw_shared = moe.shared_expert(self.x)
        scale = 1.0 + 0.5 * torch.tanh(moe.shared_fusion(self.z))
        routed = output - scale * raw_shared
        expected_raw = FactorGatedMoE.shared_routed_decoupling_loss(
            raw_shared, routed
        )
        scaled_penalty = FactorGatedMoE.shared_routed_decoupling_loss(
            scale * raw_shared, routed
        )

        self.assertTrue(torch.equal(penalty, expected_raw))
        self.assertTrue(torch.isfinite(route_loss))
        # Per-sample positive scaling is cosine invariant; verify raw usage from
        # the implementation as well as its mathematically equivalent value.
        self.assertTrue(torch.allclose(penalty, scaled_penalty, atol=1e-7, rtol=0))
        forward_source = inspect.getsource(FactorGatedMoE.forward)
        self.assertIn(
            "self.shared_routed_decoupling_loss(\n"
            "                raw_shared_out, routed_out",
            forward_source,
        )

    def test_018_prediction_and_original_aux_are_bitwise_equal_to_016(self):
        fusion_016 = HyperFusion(
            d_h=8,
            d_z=6,
            k_prior=3,
            k_latent=6,
            drop=0.0,
            num_experts=3,
            moe_k=1,
            hidden_size=8,
            use_shared_expert=True,
            use_adaptive_shared_fusion=True,
        ).eval()
        self._make_shared_nonzero(fusion_016.moe)
        with torch.no_grad():
            fusion_016.moe.shared_fusion.weight.normal_(0.0, 0.2)
            fusion_016.moe.shared_fusion.bias.fill_(0.1)
        fusion_018 = copy.deepcopy(fusion_016)

        h = torch.randn(self.batch, 8)
        z_q = torch.randn(self.batch, 6)
        outputs_016 = fusion_016(h, z_q)
        outputs_018 = fusion_018(h, z_q, return_decoupling_loss=True)

        for expected, actual in zip(outputs_016, outputs_018[:4]):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(outputs_018[4].ndim, 0)

    def test_adaptive_formula_zero_init_and_raw_zq_conditioning_remain(self):
        moe = self._make_moe().eval()
        self.assertEqual(moe.shared_fusion_delta, 0.5)
        self.assertEqual(torch.count_nonzero(moe.shared_fusion.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(moe.shared_fusion.bias).item(), 0)
        alpha = 1.0 + 0.5 * torch.tanh(moe.shared_fusion(self.z))
        self.assertTrue(torch.equal(alpha, torch.ones(self.batch, 1)))

        fusion = HyperFusion(
            d_h=8,
            d_z=6,
            k_prior=3,
            k_latent=6,
            drop=0.0,
            num_experts=3,
            moe_k=1,
            hidden_size=8,
            use_shared_expert=True,
            use_adaptive_shared_fusion=True,
        ).eval()
        seen = []
        handle = fusion.moe.shared_fusion.register_forward_hook(
            lambda _module, inputs, _output: seen.append(inputs[0].detach().clone())
        )
        z_q = torch.randn(self.batch, 6)
        fusion(torch.randn(self.batch, 8), z_q, return_decoupling_loss=True)
        handle.remove()
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], z_q))

    def test_final_objective_is_independent_of_aux_pipeline(self):
        holder = SimpleNamespace(aux_weight=0.37, decoupling_lambda=0.01)
        rank_loss = torch.tensor(2.0)
        raw_aux = torch.tensor(12.0)
        aux_loss = softcap_log1p(raw_aux, 3.0)
        decoupling_loss = torch.tensor(0.81)
        actual = GenerateReturn._total_objective(
            holder, rank_loss, aux_loss, decoupling_loss
        )
        expected = rank_loss + 0.37 * aux_loss + 0.01 * decoupling_loss

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(
            torch.equal(
                actual,
                rank_loss + 0.37 * (aux_loss + 0.01 * decoupling_loss),
            )
        )
        self.assertFalse(
            torch.equal(
                actual,
                rank_loss
                + 0.37 * aux_loss
                + 0.01 * softcap_log1p(decoupling_loss, 3.0),
            )
        )

    def test_training_and_validation_share_the_exact_objective_builder(self):
        train_source = inspect.getsource(GenerateReturn.training_step)
        validation_source = inspect.getsource(GenerateReturn.validation_step)
        self.assertEqual(train_source.count("self._total_objective("), 1)
        self.assertEqual(validation_source.count("self._total_objective("), 1)
        self.assertIn("train_decoupling_loss", train_source)
        self.assertIn("val_decoupling_loss", validation_source)

    def test_standard_interfaces_and_state_dict_remain_unchanged(self):
        moe = self._make_moe().eval()
        state = copy.deepcopy(moe.state_dict())
        restored = self._make_moe().eval()
        result = restored.load_state_dict(state, strict=True)
        self.assertFalse(result.missing_keys)
        self.assertFalse(result.unexpected_keys)

        standard = restored(self.x, self.z)
        detailed = restored(
            self.x, self.z, return_decoupling_loss=True
        )
        self.assertEqual(len(standard), 2)
        self.assertEqual(len(detailed), 3)
        self.assertTrue(torch.equal(standard[0], detailed[0]))
        self.assertTrue(torch.equal(standard[1], detailed[1]))


if __name__ == "__main__":
    unittest.main()
