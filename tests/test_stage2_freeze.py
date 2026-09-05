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

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.train_ypred import GenerateReturn


def tiny_config():
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
            "transformer": {
                "num_heads": 2,
                "num_layers": 1,
                "d_model": 8,
                "dim_feedforward": 16,
                "dropout": 0.1,
                "batch_first": True,
            },
        },
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
            self.model(self.feature, self.prior)
        self.assertTrue(torch.equal(weight_before, self.model.quantizer.embedding.weight.detach()))
        self.assertTrue(torch.equal(embed_prob_before, self.model.quantizer.embed_prob))

    def test_frozen_representation_is_deterministic_in_train_mode(self):
        # The encoder contains dropout; while frozen it must stay in eval so
        # repeated forwards of the same input give identical z_q.
        self.model.train()
        with torch.no_grad():
            z_q1 = self.model(self.feature, self.prior)[3]
            z_q2 = self.model(self.feature, self.prior)[3]
        self.assertTrue(torch.equal(z_q1, z_q2))


if __name__ == "__main__":
    unittest.main()
