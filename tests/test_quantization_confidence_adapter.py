import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.train_ypred import GenerateReturn


def tiny_config(adapter=True):
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


def build_model(adapter=True, seed=0):
    torch.manual_seed(seed)
    config = copy.deepcopy(tiny_config(adapter=adapter))
    with mock.patch.object(
        GenerateReturn,
        "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(config, T_max=10)


class QuantizationConfidenceAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.h = torch.randn(6, 8)
        self.z_q = torch.randn(6, 8)

    def test_default_config_enables_exact_experiment(self):
        config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)

        self.assertIs(
            config["predictor"]["quantization_confidence_adapter"], True
        )
        self.assertEqual(config["vqvae"]["vq_embed_dim"], 128)
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_adapter_is_linear_1_to_latent_and_exactly_zero_initialized(self):
        model = build_model()
        adapter = model.quantization_confidence_adapter

        self.assertEqual(adapter.in_features, 1)
        self.assertEqual(adapter.out_features, 8)
        self.assertEqual(torch.count_nonzero(adapter.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.bias).item(), 0)

    def test_quantization_error_matches_required_definition(self):
        actual = GenerateReturn.quantization_error(self.h, self.z_q)
        expected = torch.mean((self.h - self.z_q) ** 2, dim=-1, keepdim=True)

        self.assertEqual(actual.shape, (6, 1))
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_init_stage2_latent_is_bitwise_equal_to_z_q(self):
        model = build_model()
        z_stage2 = model.build_stage2_latent(self.h, self.z_q)

        self.assertTrue(torch.equal(z_stage2, self.z_q))

    def test_new_adapter_does_not_perturb_existing_parameter_initialization(self):
        base = build_model(adapter=False, seed=1234)
        adapted = build_model(adapter=True, seed=1234)
        adapted_base_state = {
            key: value
            for key, value in adapted.state_dict().items()
            if not key.startswith("quantization_confidence_adapter.")
        }

        self.assertEqual(base.state_dict().keys(), adapted_base_state.keys())
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, adapted_base_state[key]), msg=key)

    def test_initial_full_prediction_forward_is_bitwise_equal_to_base(self):
        base = build_model(adapter=False, seed=4321).eval()
        adapted = build_model(adapter=True, seed=4321).eval()
        feature = torch.randn(9, 5, 8)
        prior = torch.randn(9, 3)

        base_out = base(feature, prior)
        adapted_out = adapted(feature, prior)

        for index, (base_value, adapted_value) in enumerate(
            zip(base_out, adapted_out)
        ):
            self.assertTrue(
                torch.equal(base_value, adapted_value), msg=f"output[{index}]"
            )

    def test_all_stage2_latent_consumers_receive_z_stage2(self):
        model = build_model().eval()
        with torch.no_grad():
            model.quantization_confidence_adapter.bias.fill_(0.25)
        feature = torch.randn(7, 5, 8)
        prior = torch.randn(7, 3)
        seen = {"loadings": [], "latent_head": []}
        loadings_handle = model.loadings.register_forward_pre_hook(
            lambda _module, inputs: seen["loadings"].append(inputs[1].detach().clone())
        )
        latent_handle = model.latent_value_head.register_forward_pre_hook(
            lambda _module, inputs: seen["latent_head"].append(
                inputs[0].detach().clone()
            )
        )

        output = model(feature, prior)
        loadings_handle.remove()
        latent_handle.remove()

        self.assertEqual(len(seen["loadings"]), 1)
        self.assertEqual(len(seen["latent_head"]), 1)
        self.assertTrue(torch.equal(seen["loadings"][0], output[3]))
        self.assertTrue(torch.equal(seen["latent_head"][0], output[3]))

    def test_adapter_gets_gradient_while_stage1_path_is_detached(self):
        model = build_model().eval()
        feature = torch.randn(11, 5, 8, requires_grad=True)
        prior = torch.randn(11, 3)
        encoder_outputs = []

        def capture_encoder_output(_module, _inputs, output):
            output.retain_grad()
            encoder_outputs.append(output)

        handle = model.encoder.register_forward_hook(capture_encoder_output)
        y_pred = model(feature, prior)[0]
        y_pred.sum().backward()
        handle.remove()

        adapter = model.quantization_confidence_adapter
        self.assertIsNotNone(adapter.weight.grad)
        self.assertIsNotNone(adapter.bias.grad)
        self.assertTrue(torch.isfinite(adapter.weight.grad).all())
        self.assertTrue(torch.isfinite(adapter.bias.grad).all())
        self.assertGreater(adapter.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(adapter.bias.grad.abs().sum().item(), 0.0)
        self.assertEqual(len(encoder_outputs), 1)
        self.assertIsNone(encoder_outputs[0].grad)
        for module in (model.encoder, model.quantizer, model.revin):
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)
                self.assertIsNone(parameter.grad, msg=name)

    def test_quantization_error_is_detached_from_both_stage1_outputs(self):
        h = self.h.clone().requires_grad_()
        z_q = self.z_q.clone().requires_grad_()
        q_error = GenerateReturn.quantization_error(h, z_q)

        self.assertFalse(q_error.requires_grad)


if __name__ == "__main__":
    unittest.main()
