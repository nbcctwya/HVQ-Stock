"""Tests for experiment 021 latent-conditioned prior/latent allocation."""

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.train_ypred import GenerateReturn, ReturnPredictor


def tiny_config(allocation=True):
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
            "latent_conditioned_allocation": allocation,
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


def build_model(allocation=True, seed=0):
    torch.manual_seed(seed)
    with mock.patch.object(
        GenerateReturn,
        "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(
            copy.deepcopy(tiny_config(allocation=allocation)), T_max=10
        )


class LatentPriorAllocationTest(unittest.TestCase):
    def test_default_config_enables_exact_experiment(self):
        config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)

        self.assertIs(
            config["predictor"]["latent_conditioned_allocation"], True
        )
        self.assertEqual(config["vqvae"]["vq_embed_dim"], 128)
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_gate_is_linear_128_to_one_and_exactly_zero_initialized(self):
        model = build_model()
        gate = model.return_predictor.allocation_gate

        self.assertEqual(gate.in_features, 8)
        self.assertEqual(gate.out_features, 1)
        self.assertEqual(torch.count_nonzero(gate.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(gate.bias).item(), 0)
        self.assertEqual(model.return_predictor.ALLOCATION_DELTA, 0.5)

    def test_zero_init_scales_are_exact_ones(self):
        predictor = ReturnPredictor(3, 8)
        predictor.enable_latent_conditioned_allocation(8)
        z_q = torch.randn(11, 8)

        prior_scale, latent_scale = predictor.allocation_scales(z_q)

        self.assertTrue(torch.equal(prior_scale, torch.ones(11)))
        self.assertTrue(torch.equal(latent_scale, torch.ones(11)))

    def test_scales_are_complementary_and_strictly_bounded(self):
        predictor = ReturnPredictor(3, 8)
        predictor.enable_latent_conditioned_allocation(8)
        with torch.no_grad():
            predictor.allocation_gate.weight.fill_(0.15)
            predictor.allocation_gate.bias.fill_(-0.2)
        z_q = torch.linspace(-1.0, 1.0, 88).reshape(11, 8)

        prior_scale, latent_scale = predictor.allocation_scales(z_q)

        self.assertTrue(torch.all(prior_scale > 0.5))
        self.assertTrue(torch.all(prior_scale < 1.5))
        self.assertTrue(torch.all(latent_scale > 0.5))
        self.assertTrue(torch.all(latent_scale < 1.5))
        self.assertTrue(
            torch.equal(prior_scale + latent_scale, torch.full((11,), 2.0))
        )

    def test_nonzero_gate_gives_different_latents_different_allocations(self):
        predictor = ReturnPredictor(1, 2)
        predictor.enable_latent_conditioned_allocation(2)
        with torch.no_grad():
            predictor.allocation_gate.weight.copy_(torch.tensor([[1.0, 0.0]]))
        z_q = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

        prior_scale, latent_scale = predictor.allocation_scales(z_q)

        self.assertGreater(prior_scale[0].item(), prior_scale[1].item())
        self.assertLess(latent_scale[0].item(), latent_scale[1].item())

    def test_allocation_changes_only_completed_factor_contributions(self):
        predictor = ReturnPredictor(1, 1)
        predictor.enable_latent_conditioned_allocation(2)
        with torch.no_grad():
            predictor.allocation_gate.weight.copy_(torch.tensor([[1.0, 0.0]]))
        z_q = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        alpha = torch.zeros(2)
        beta_p = torch.tensor([[2.0], [2.0]])
        beta_l = torch.tensor([[3.0], [3.0]])
        f_prior = torch.tensor([[2.0], [2.0]])
        f_latent = torch.tensor([[1.0], [1.0]])

        prior_scale, latent_scale = predictor.allocation_scales(z_q)
        output = predictor(
            alpha, beta_p, beta_l, f_prior, f_latent, z_q=z_q
        )
        prior_term = torch.tensor([4.0, 4.0])
        latent_term = torch.tensor([3.0, 3.0])

        self.assertGreater(
            (prior_scale * prior_term)[0].item(),
            (prior_scale * prior_term)[1].item(),
        )
        self.assertLess(
            (latent_scale * latent_term)[0].item(),
            (latent_scale * latent_term)[1].item(),
        )
        self.assertTrue(
            torch.equal(
                output,
                prior_scale * prior_term + latent_scale * latent_term,
            )
        )

    def test_gate_receives_finite_nonzero_gradient(self):
        predictor = ReturnPredictor(1, 1)
        predictor.enable_latent_conditioned_allocation(2)
        z_q = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
        output = predictor(
            alpha=torch.zeros(2),
            beta_p=torch.tensor([[2.0], [1.0]]),
            beta_l=torch.tensor([[0.5], [3.0]]),
            f_prior=torch.ones(2, 1),
            f_latent=torch.ones(2, 1),
            z_q=z_q,
        )

        output.sum().backward()
        gate = predictor.allocation_gate

        self.assertTrue(torch.isfinite(gate.weight.grad).all())
        self.assertTrue(torch.isfinite(gate.bias.grad).all())
        self.assertGreater(gate.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(gate.bias.grad.abs().sum().item(), 0.0)

    def test_new_gate_does_not_perturb_existing_parameter_initialization(self):
        base = build_model(allocation=False, seed=1234)
        allocated = build_model(allocation=True, seed=1234)
        existing_state = {
            key: value
            for key, value in allocated.state_dict().items()
            if not key.startswith("return_predictor.allocation_gate.")
        }

        self.assertEqual(base.state_dict().keys(), existing_state.keys())
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, existing_state[key]), msg=key)

    def test_zero_init_full_prediction_forward_is_bitwise_equal_to_base(self):
        base = build_model(allocation=False, seed=4321).eval()
        allocated = build_model(allocation=True, seed=4321).eval()
        feature = torch.randn(9, 5, 8)
        prior = torch.randn(9, 3)

        with torch.no_grad():
            base_output = base(feature, prior)
            allocated_output = allocated(feature, prior)

        for index, (base_value, allocated_value) in enumerate(
            zip(base_output, allocated_output)
        ):
            self.assertTrue(
                torch.equal(base_value, allocated_value), msg=f"output[{index}]"
            )

    def test_full_model_gate_gradient_does_not_reach_frozen_stage1(self):
        model = build_model().eval()
        feature = torch.randn(11, 5, 8)
        prior = torch.randn(11, 3)

        model(feature, prior)[0].sum().backward()
        gate = model.return_predictor.allocation_gate

        self.assertIsNotNone(gate.weight.grad)
        self.assertIsNotNone(gate.bias.grad)
        self.assertTrue(torch.isfinite(gate.weight.grad).all())
        self.assertTrue(torch.isfinite(gate.bias.grad).all())
        self.assertGreater(gate.weight.grad.abs().sum().item(), 0.0)
        for module in (model.encoder, model.quantizer, model.revin):
            self.assertFalse(module.training)
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)
                self.assertIsNone(parameter.grad, msg=name)


if __name__ == "__main__":
    unittest.main()
