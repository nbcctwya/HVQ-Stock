"""Regression tests: Stage 1 submodules must stay frozen and in eval mode
throughout Stage 2 training.

Background: PyTorch Lightning calls ``model.train()`` on the whole
LightningModule, which recursively re-enables training mode on the frozen
Stage 1 modules. ``requires_grad=False`` does not prevent training-mode side
effects: VectorQuantiser updates ``embed_prob`` and rewrites the codebook via
``.data`` when ``self.training`` is True, and the encoder's Transformer
dropout makes frozen representations stochastic.

These tests build a real ``GenerateReturn`` with tiny dims; checkpoint
loading is patched out (freeze behaviour does not depend on checkpoint
contents).
"""

import ast
import inspect
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.train_ypred import GenerateReturn
from module.layers.encoder import Gate


def tiny_config():
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
            "dropout": 0.1,
            "rank": 0,
            "target_day": 2,
            "use_prior": True,
            "transformer": {
                "num_heads": 2,
                "num_layers": 1,
                "d_model": 8,
                "dim_feedforward": 16,
                "dropout": 0.1,
                "batch_first": True,
            },
        },
        "data": {"universe": "csi300"},
        "train": {"learning_rate": 0.0001},
    }


def build_model():
    with mock.patch.object(GenerateReturn, "load_pretrained_vqvae", lambda self, checkpoint_path=None: None):
        return GenerateReturn(tiny_config(), T_max=10)


class Stage2FreezeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_model()
        self.feature = torch.randn(6, 5, 8)
        self.prior = torch.randn(6, 3)
        self.market = torch.randn(6, 5, 3)

    def test_train_call_keeps_frozen_modules_in_eval(self):
        self.model.train()  # what Lightning does at training start
        self.assertFalse(self.model.encoder.training)
        self.assertFalse(self.model.quantizer.training)
        self.assertFalse(self.model.revin.training)
        # Trainable Stage 2 modules are unaffected.
        self.assertTrue(self.model.loadings.training)
        self.assertTrue(self.model.latent_value_head.training)
        self.assertTrue(self.model.return_predictor.training)

    def test_eval_call_still_works(self):
        self.model.eval()
        self.assertFalse(self.model.encoder.training)
        self.assertFalse(self.model.loadings.training)

    def test_frozen_params_keep_requires_grad_false(self):
        self.model.train()
        for module in (self.model.encoder, self.model.quantizer, self.model.revin):
            for name, param in module.named_parameters():
                self.assertFalse(param.requires_grad, msg=name)
        for name, param in self.model.loadings.named_parameters():
            self.assertTrue(param.requires_grad, msg=name)

    def test_codebook_not_modified_by_forward_in_train_mode(self):
        self.model.train()
        weight_before = self.model.quantizer.embedding.weight.detach().clone()
        embed_prob_before = self.model.quantizer.embed_prob.detach().clone()
        with torch.no_grad():
            self.model(self.feature, self.prior, self.market)
        self.assertTrue(torch.equal(weight_before, self.model.quantizer.embedding.weight.detach()))
        self.assertTrue(torch.equal(embed_prob_before, self.model.quantizer.embed_prob))

    def test_frozen_representation_is_deterministic_in_train_mode(self):
        # The encoder contains dropout; while frozen it must stay in eval so
        # repeated forwards of the same input give identical z_q.
        self.model.train()
        with torch.no_grad():
            z_q1 = self.model(self.feature, self.prior, self.market)[3]
            z_q2 = self.model(self.feature, self.prior, self.market)[3]
        self.assertTrue(torch.equal(z_q1, z_q2))

    def test_market_is_consumed_only_by_frozen_stage1_encoder(self):
        gate_modules = [
            name for name, module in self.model.named_modules()
            if isinstance(module, Gate)
        ]
        self.assertEqual(gate_modules, ["encoder.feature_gate"])

        tree = ast.parse(textwrap.dedent(inspect.getsource(GenerateReturn.forward)))
        market_reads = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "market_feature"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(len(market_reads), 1)
        encoder_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "encoder"
        ]
        self.assertEqual(len(encoder_calls), 1)
        self.assertIn(market_reads[0], encoder_calls[0].args)

    def test_stage2_reconstructs_gate_with_universe_beta(self):
        for universe, beta in (("csi300", 10), ("sp500", 5)):
            with self.subTest(universe=universe):
                config = tiny_config()
                config["data"]["universe"] = universe
                with mock.patch.object(
                    GenerateReturn,
                    "load_pretrained_vqvae",
                    lambda self, checkpoint_path=None: None,
                ):
                    model = GenerateReturn(config, T_max=10)
                self.assertEqual(model.market_beta, beta)
                self.assertEqual(model.encoder.feature_gate.t, beta)


if __name__ == "__main__":
    unittest.main()
