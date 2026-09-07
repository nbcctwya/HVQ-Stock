import copy
import unittest
from pathlib import Path

import torch
import yaml

from module.layers.fusion import HyperFusion
from module.layers.moe import FactorGatedMoE, SimpleMLP
from trainer.train_ypred import ReturnPredictor


class SharedRoutedMoETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 11
        self.input_dim = 8
        self.gate_dim = 6
        self.hidden_dim = 8
        self.x = torch.randn(self.batch, self.input_dim)
        self.z = torch.randn(self.batch, self.gate_dim)

    def _make_moe(self, shared):
        return FactorGatedMoE(
            gate_input_size=self.gate_dim,
            expert_input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_experts=3,
            noisy_gating=True,
            k=1,
            use_shared_expert=shared,
        )

    @staticmethod
    def _copy_routed_state(source, target):
        routed_state = {
            key: value
            for key, value in source.state_dict().items()
            if not key.startswith("shared_expert.")
        }
        result = target.load_state_dict(routed_state, strict=True)
        assert not result.missing_keys
        assert not result.unexpected_keys

    def test_default_config_enables_shared_expert(self):
        config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)
        self.assertIs(config["predictor"]["shared_expert"], True)
        self.assertEqual(config["train"]["seed"], 0)

    def test_shared_expert_matches_routed_expert_structure_and_shapes(self):
        moe = self._make_moe(shared=True).eval()
        self.assertIsInstance(moe.shared_expert, SimpleMLP)
        self.assertEqual(repr(moe.shared_expert), repr(moe.experts[0]))

        shared_out = moe.shared_expert(self.x)
        routed_out, _ = self._routed_only_clone(moe)(self.x, self.z)
        moe_out, _ = moe(self.x, self.z)
        self.assertEqual(shared_out.shape, routed_out.shape)
        self.assertEqual(moe_out.shape, routed_out.shape)
        self.assertEqual(moe_out.shape, (self.batch, self.hidden_dim))

    def _routed_only_clone(self, shared_moe):
        routed = self._make_moe(shared=False)
        self._copy_routed_state(shared_moe, routed)
        return routed.eval()

    def test_shared_expert_is_always_on_and_bypasses_dispatcher(self):
        moe = self._make_moe(shared=True).eval()
        seen = []
        handle = moe.shared_expert.register_forward_hook(
            lambda _module, inputs, _output: seen.append(inputs[0].detach().clone())
        )
        out_a, _ = moe(self.x, self.z)
        out_b, _ = moe(self.x, -self.z)
        handle.remove()

        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.equal(seen[0], self.x))
        self.assertTrue(torch.equal(seen[1], self.x))
        self.assertEqual(seen[0].shape[0], self.batch)
        self.assertEqual(out_a.shape, out_b.shape)

    def test_routed_path_and_auxiliary_loss_are_bitwise_base_equivalent(self):
        shared = self._make_moe(shared=True).eval()
        routed = self._routed_only_clone(shared)

        shared_out, shared_loss = shared(self.x, self.z)
        routed_out, routed_loss = routed(self.x, self.z)

        self.assertTrue(torch.equal(shared_out, routed_out))
        self.assertTrue(torch.equal(shared_loss, routed_loss))
        self.assertEqual(shared.num_experts, routed.num_experts)
        self.assertEqual(shared.k, routed.k)

    def test_enabling_shared_does_not_change_routed_initialization(self):
        torch.manual_seed(1234)
        routed = self._make_moe(shared=False).eval()
        torch.manual_seed(1234)
        shared = self._make_moe(shared=True).eval()

        shared_routed_state = {
            key: value
            for key, value in shared.state_dict().items()
            if not key.startswith("shared_expert.")
        }
        self.assertEqual(routed.state_dict().keys(), shared_routed_state.keys())
        for key, value in routed.state_dict().items():
            self.assertTrue(torch.equal(value, shared_routed_state[key]), msg=key)

        routed_out, routed_loss = routed(self.x, self.z)
        shared_out, shared_loss = shared(self.x, self.z)
        self.assertTrue(torch.equal(routed_out, shared_out))
        self.assertTrue(torch.equal(routed_loss, shared_loss))

    def test_shared_final_linear_is_exactly_zero_initialized(self):
        moe = self._make_moe(shared=True).eval()
        final_linear = moe.shared_expert.net[-1]
        shared_out = moe.shared_expert(self.x)

        self.assertTrue(torch.count_nonzero(final_linear.weight) == 0)
        self.assertTrue(torch.count_nonzero(final_linear.bias) == 0)
        self.assertTrue(torch.count_nonzero(shared_out) == 0)

    def test_shared_final_linear_gets_gradient_and_updates(self):
        moe = self._make_moe(shared=True).train()
        final_linear = moe.shared_expert.net[-1]
        before_weight = final_linear.weight.detach().clone()
        before_bias = final_linear.bias.detach().clone()
        optimizer = torch.optim.SGD(moe.parameters(), lr=0.1)

        output, aux_loss = moe(self.x, self.z)
        loss = output.square().mean() + 0.01 * aux_loss
        optimizer.zero_grad()
        loss.backward()

        self.assertIsNotNone(final_linear.weight.grad)
        self.assertIsNotNone(final_linear.bias.grad)
        self.assertGreater(final_linear.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(final_linear.bias.grad.abs().sum().item(), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(final_linear.weight, before_weight))
        self.assertFalse(torch.equal(final_linear.bias, before_bias))

    def test_hyperfusion_and_return_predictor_interfaces_remain_compatible(self):
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
        ).eval()
        h = torch.randn(self.batch, 8)
        z = torch.randn(self.batch, 6)
        alpha, beta_p, beta_l, loss = fusion(h, z)

        self.assertEqual(alpha.shape, (self.batch,))
        self.assertEqual(beta_p.shape, (self.batch, 3))
        self.assertEqual(beta_l.shape, (self.batch, 6))
        self.assertEqual(loss.ndim, 0)

        predictor = ReturnPredictor(3, 6, use_prior=True)
        prediction = predictor(
            alpha,
            beta_p,
            beta_l,
            torch.randn(self.batch, 3),
            torch.randn(self.batch, 6),
        )
        self.assertEqual(prediction.shape, (self.batch,))

    def test_checkpoint_round_trip_preserves_outputs(self):
        source = self._make_moe(shared=True).eval()
        with torch.no_grad():
            source.shared_expert.net[-1].weight.normal_(0.0, 0.1)
        state = copy.deepcopy(source.state_dict())
        restored = self._make_moe(shared=True).eval()
        result = restored.load_state_dict(state, strict=True)
        self.assertFalse(result.missing_keys)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(torch.equal(source(self.x, self.z)[0], restored(self.x, self.z)[0]))


if __name__ == "__main__":
    unittest.main()
