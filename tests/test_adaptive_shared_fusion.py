import copy
import unittest
from pathlib import Path

import torch
import yaml

from module.layers.fusion import HyperFusion
from module.layers.moe import FactorGatedMoE


class AdaptiveSharedFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 11
        self.input_dim = 8
        self.gate_dim = 6
        self.hidden_dim = 8
        self.x = torch.randn(self.batch, self.input_dim)
        self.z = torch.randn(self.batch, self.gate_dim)

    def _make_moe(self, adaptive=True, shared=True):
        return FactorGatedMoE(
            gate_input_size=self.gate_dim,
            expert_input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_experts=3,
            noisy_gating=True,
            k=1,
            use_shared_expert=shared,
            use_adaptive_shared_fusion=adaptive,
        )

    @staticmethod
    def _base_state(adaptive_moe):
        return {
            key: value
            for key, value in adaptive_moe.state_dict().items()
            if not key.startswith("shared_fusion.")
        }

    def _matched_pair(self):
        torch.manual_seed(1234)
        base = self._make_moe(adaptive=False).eval()
        torch.manual_seed(1234)
        adaptive = self._make_moe(adaptive=True).eval()
        return base, adaptive

    def test_default_config_enables_experiment(self):
        config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)

        self.assertIs(config["predictor"]["shared_expert"], True)
        self.assertIs(config["predictor"]["adaptive_shared_fusion"], True)
        self.assertEqual(config["predictor"]["n_expert"], 2)
        self.assertEqual(config["predictor"]["k"], "${half:${predictor.n_expert}}")
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_affine_map_is_exactly_zero_initialized_and_alpha_is_one(self):
        moe = self._make_moe()

        self.assertEqual(moe.shared_fusion_delta, 0.5)
        self.assertEqual(moe.shared_fusion.in_features, self.gate_dim)
        self.assertEqual(moe.shared_fusion.out_features, 1)
        self.assertEqual(torch.count_nonzero(moe.shared_fusion.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(moe.shared_fusion.bias).item(), 0)

        alpha = 1.0 + moe.shared_fusion_delta * torch.tanh(
            moe.shared_fusion(self.z)
        )
        self.assertTrue(torch.equal(alpha, torch.ones(self.batch, 1)))

    def test_new_layer_does_not_perturb_010_parameter_initialization(self):
        base, adaptive = self._matched_pair()
        adaptive_base_state = self._base_state(adaptive)

        self.assertEqual(base.state_dict().keys(), adaptive_base_state.keys())
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, adaptive_base_state[key]), msg=key)

    def test_initial_forward_is_bitwise_equal_to_010_with_nonzero_shared_output(self):
        base, adaptive = self._matched_pair()
        with torch.no_grad():
            shared_weight = torch.linspace(
                -0.2,
                0.2,
                base.shared_expert.net[-1].weight.numel(),
            ).reshape_as(base.shared_expert.net[-1].weight)
            shared_bias = torch.linspace(
                -0.1,
                0.1,
                base.shared_expert.net[-1].bias.numel(),
            )
            base.shared_expert.net[-1].weight.copy_(shared_weight)
            base.shared_expert.net[-1].bias.copy_(shared_bias)
            adaptive.shared_expert.net[-1].weight.copy_(shared_weight)
            adaptive.shared_expert.net[-1].bias.copy_(shared_bias)

        base_out, base_loss = base(self.x, self.z)
        adaptive_out, adaptive_loss = adaptive(self.x, self.z)

        self.assertGreater(adaptive.shared_expert(self.x).abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(adaptive_out, base_out))
        self.assertTrue(torch.equal(adaptive_loss, base_loss))

    def test_forward_uses_bounded_latent_conditioned_scale(self):
        adaptive = self._make_moe().eval()
        base = self._make_moe(adaptive=False).eval()
        result = base.load_state_dict(self._base_state(adaptive), strict=True)
        self.assertFalse(result.missing_keys)
        self.assertFalse(result.unexpected_keys)

        with torch.no_grad():
            adaptive.shared_expert.net[-1].weight.normal_(0.0, 0.1)
            adaptive.shared_expert.net[-1].bias.normal_(0.0, 0.1)
            base.shared_expert.net[-1].load_state_dict(
                adaptive.shared_expert.net[-1].state_dict()
            )
            adaptive.shared_fusion.weight.fill_(0.75)
            adaptive.shared_fusion.bias.fill_(-0.2)

        shared_out = adaptive.shared_expert(self.x)
        scale = 1.0 + 0.5 * torch.tanh(adaptive.shared_fusion(self.z))
        base_out, base_loss = base(self.x, self.z)
        adaptive_out, adaptive_loss = adaptive(self.x, self.z)
        expected = base_out + (scale - 1.0) * shared_out

        self.assertTrue(torch.all(scale > 0.5))
        self.assertTrue(torch.all(scale < 1.5))
        self.assertGreater(torch.unique(scale).numel(), 1)
        self.assertTrue(torch.allclose(adaptive_out, expected, rtol=0, atol=1e-7))
        self.assertTrue(torch.equal(adaptive_loss, base_loss))

    def test_fusion_map_receives_gradient_and_updates(self):
        moe = self._make_moe().train()
        with torch.no_grad():
            moe.shared_expert.net[-1].weight.normal_(0.0, 0.1)
            moe.shared_expert.net[-1].bias.fill_(0.1)

        optimizer = torch.optim.SGD(moe.parameters(), lr=0.1)
        before_weight = moe.shared_fusion.weight.detach().clone()
        before_bias = moe.shared_fusion.bias.detach().clone()
        output, _ = moe(self.x, self.z)
        optimizer.zero_grad()
        output.sum().backward()

        self.assertGreater(moe.shared_fusion.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(moe.shared_fusion.bias.grad.abs().sum().item(), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(moe.shared_fusion.weight, before_weight))
        self.assertFalse(torch.equal(moe.shared_fusion.bias, before_bias))

    def test_hyperfusion_wires_adaptive_shared_fusion(self):
        fusion = HyperFusion(
            d_h=8,
            d_z=6,
            k_prior=3,
            k_latent=6,
            drop=0.0,
            num_experts=2,
            moe_k=1,
            hidden_size=8,
            use_shared_expert=True,
            use_adaptive_shared_fusion=True,
        ).eval()

        self.assertIsNotNone(fusion.moe.shared_expert)
        self.assertIsNotNone(fusion.moe.shared_fusion)
        seen = []
        handle = fusion.moe.shared_fusion.register_forward_hook(
            lambda _module, inputs, _output: seen.append(inputs[0].detach().clone())
        )
        outputs = fusion(torch.randn(self.batch, 8), self.z)
        handle.remove()
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], self.z))
        self.assertEqual(outputs[0].shape, (self.batch,))
        self.assertEqual(outputs[1].shape, (self.batch, 3))
        self.assertEqual(outputs[2].shape, (self.batch, 6))
        self.assertEqual(outputs[3].ndim, 0)

    def test_checkpoint_round_trip_is_strict_and_exact(self):
        source = self._make_moe().eval()
        with torch.no_grad():
            source.shared_expert.net[-1].weight.normal_(0.0, 0.1)
            source.shared_fusion.weight.normal_(0.0, 0.1)
        state = copy.deepcopy(source.state_dict())
        restored = self._make_moe().eval()
        result = restored.load_state_dict(state, strict=True)

        self.assertFalse(result.missing_keys)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(
            torch.equal(source(self.x, self.z)[0], restored(self.x, self.z)[0])
        )

    def test_adaptive_fusion_requires_shared_expert(self):
        with self.assertRaisesRegex(ValueError, "requires use_shared_expert=True"):
            self._make_moe(adaptive=True, shared=False)


if __name__ == "__main__":
    unittest.main()
