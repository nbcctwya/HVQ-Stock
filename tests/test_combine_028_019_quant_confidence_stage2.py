"""Tests for experiment 038: 028 temporal-attention Stage 1 + 019 Stage 2.

Experiment 038 is the w/o Shared-Routed ablation of experiment 034: it keeps
experiment 034's 028 temporal-attention Stage 1 (FeatureTransform -> Input
Projection -> PositionalEncoding -> TAttention -> TemporalAttention ->
CrossAssetTransformer; no GRU, no SAttention, no Market Gate) and the 019
Quantization Confidence Adapter unchanged, and removes only the always-on
Shared Expert that 034 inherited from experiment 010.  Stage 2 is therefore
the original routed-only FactorGatedMoE (2 Routed Experts, top-k = 1,
original router/noise network/W_h/SparseDispatcher/auxiliary loss) exactly as
in experiment 019.
"""

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parents[1]
STAGE1_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "028"
    / "run"
    / "checkpoints"
    / "infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5865.ckpt"
)

from module.layers.encoder import (
    PositionalEncoding,
    TAttention,
    TemporalAttention,
    TemporalAttentionEncoder,
)
from module.layers.moe import FactorGatedMoE, SimpleMLP
from module.quantise import VectorQuantiser
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
            "encoder": {
                "type": "temporal-attention",
                "num_heads": 2,
                "num_layers": 1,
                "temporal_dropout": 0.1,
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


def real_config():
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    return OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )


def build_real_model():
    """Construct GenerateReturn from the default config without Stage 1 I/O."""
    config = real_config()
    config["predictor"]["saved_model"] = str(STAGE1_CHECKPOINT)
    with mock.patch.object(
        GenerateReturn,
        "load_pretrained_vqvae",
        lambda self, checkpoint_path=None: None,
    ):
        return GenerateReturn(config, T_max=10)


def capture_strict_load(model, checkpoint_path):
    """Run load_pretrained_vqvae while recording strict load statistics."""
    captured = {}
    for name, module in (
        ("encoder", model.encoder),
        ("quantizer", model.quantizer),
        ("revin", model.revin),
    ):
        original = module.load_state_dict

        def wrapper(state_dict, strict=True, _name=name, _original=original):
            result = _original(state_dict, strict)
            captured[_name] = {
                "missing": len(result.missing_keys),
                "unexpected": len(result.unexpected_keys),
            }
            return result

        module.load_state_dict = wrapper
    model.load_pretrained_vqvae(str(checkpoint_path))
    return captured


class DefaultConfigTests(unittest.TestCase):
    """The default configs/config.yaml must fully represent experiment 038."""

    def setUp(self):
        config_path = ROOT / "configs" / "config.yaml"
        with config_path.open() as stream:
            self.config = yaml.safe_load(stream)

    def test_encoder_selects_temporal_attention_stage1(self):
        encoder = self.config["vqvae"]["encoder"]
        self.assertEqual(encoder["type"], "temporal-attention")
        self.assertEqual(encoder["num_heads"], 2)
        self.assertEqual(encoder["num_layers"], 1)
        self.assertEqual(encoder["temporal_dropout"], 0.1)

    def test_stage1_dimensions_unchanged(self):
        vqvae = self.config["vqvae"]
        self.assertEqual(vqvae["hidden_size"], 128)
        self.assertEqual(vqvae["vq_embed_dim"], 128)
        self.assertEqual(vqvae["num_embed"], 512)

    def test_shared_expert_removed_and_adapter_enabled(self):
        predictor = self.config["predictor"]
        # 010's always-on Shared Expert flag must be gone entirely, so the
        # MoE falls back to the original routed-only code path.
        self.assertNotIn("shared_expert", predictor)
        self.assertIs(predictor["quantization_confidence_adapter"], True)
        self.assertEqual(predictor["n_expert"], 2)

    def test_seed_and_data_splits_unchanged(self):
        self.assertEqual(self.config["train"]["seed"], 0)
        data = self.config["data"]
        self.assertEqual(data["train_period"], ["2009-01-01", "2020-12-31"])
        self.assertEqual(data["valid_period"], ["2021-01-01", "2022-12-31"])
        self.assertEqual(data["test_period"], ["2023-01-01", "2025-12-31"])


class EncoderStructureTests(unittest.TestCase):
    """GenerateReturn's Stage 1 must be the 028 temporal-attention encoder."""

    def setUp(self):
        self.model = build_model().eval()
        self.encoder = self.model.encoder

    def test_temporal_attention_encoder_present_with_required_components(self):
        temporal = self.encoder.temporal_encoder
        self.assertIsInstance(temporal, TemporalAttentionEncoder)
        self.assertIsInstance(temporal.input_projection, nn.Linear)
        self.assertIsInstance(temporal.positional_encoding, PositionalEncoding)
        self.assertIsInstance(temporal.temporal_attention, TAttention)
        self.assertIsInstance(temporal.temporal_aggregation, TemporalAttention)

    def test_no_gru_no_sattention_no_market_gate(self):
        self.assertFalse(
            any(isinstance(module, nn.GRU) for module in self.encoder.modules())
        )
        self.assertFalse(
            any(
                type(module).__name__ == "SAttention"
                for module in self.encoder.modules()
            )
        )
        self.assertFalse(
            any("gate" in name.lower() for name, _ in self.encoder.named_modules())
        )
        self.assertFalse(
            any("gate" in name.lower() for name, _ in self.encoder.named_parameters())
        )

    def test_feature_transform_and_cross_asset_out_layer_preserved(self):
        transform = self.encoder.feature_transform
        self.assertIsInstance(transform.linear, nn.Linear)
        self.assertIsInstance(transform.normalize, nn.LayerNorm)
        self.assertIsInstance(transform.leakyrelu, nn.LeakyReLU)
        out_layer = self.encoder.cross_asset_transformer.out_layer
        self.assertEqual(len(out_layer), 3)
        self.assertIsInstance(out_layer[0], nn.Linear)
        self.assertIsInstance(out_layer[1], nn.GELU)
        self.assertIsInstance(out_layer[2], nn.Linear)

    def test_attention_stack_order(self):
        temporal = self.encoder.temporal_encoder
        seen = []
        modules = [
            ("feature_transform", self.encoder.feature_transform),
            ("input_projection", temporal.input_projection),
            ("positional_encoding", temporal.positional_encoding),
            ("tattention", temporal.temporal_attention),
            ("temporal_aggregation", temporal.temporal_aggregation),
            ("cross_asset_transformer", self.encoder.cross_asset_transformer),
        ]
        handles = [
            module.register_forward_hook(
                lambda _module, _inputs, _output, name=name: seen.append(name)
            )
            for name, module in modules
        ]
        try:
            with torch.no_grad():
                output = self.encoder(torch.randn(6, 5, 8))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(seen, [name for name, _ in modules])
        self.assertEqual(output.shape, (6, 8))


class Stage1CheckpointTests(unittest.TestCase):
    """The 028 official Stage 1 checkpoint must strict-load with zero diffs."""

    def setUp(self):
        if not STAGE1_CHECKPOINT.is_file():
            raise unittest.SkipTest(
                f"028 official Stage 1 checkpoint missing: {STAGE1_CHECKPOINT}"
            )
        self.model = build_real_model().eval()
        self.captured = capture_strict_load(self.model, STAGE1_CHECKPOINT)

    def test_strict_load_has_no_missing_or_unexpected_keys(self):
        self.assertEqual(set(self.captured), {"encoder", "quantizer", "revin"})
        for name, counts in self.captured.items():
            self.assertEqual(counts["missing"], 0, msg=name)
            self.assertEqual(counts["unexpected"], 0, msg=name)

    def test_loaded_encoder_is_temporal_attention_structure(self):
        self.assertIsInstance(
            self.model.encoder.temporal_encoder, TemporalAttentionEncoder
        )
        self.assertFalse(
            any(isinstance(module, nn.GRU) for module in self.model.encoder.modules())
        )
        self.assertFalse(
            any(
                type(module).__name__ == "SAttention"
                for module in self.model.encoder.modules()
            )
        )
        self.assertFalse(
            any(
                "gate" in name.lower()
                for name, _ in self.model.encoder.named_modules()
            )
        )

    def test_forward_h_is_128_dim_and_z_q_from_same_quantizer(self):
        model = self.model
        self.assertIsInstance(model.quantizer, VectorQuantiser)
        quantizers = [
            module for module in model.modules() if isinstance(module, VectorQuantiser)
        ]
        self.assertEqual(len(quantizers), 1)

        feature = torch.randn(4, 20, 158)
        prior = torch.randn(4, 13)
        seen = []
        handle = model.encoder.register_forward_hook(
            lambda _module, _inputs, output: seen.append(output.detach().clone())
        )
        with torch.no_grad():
            y_pred, _, _, z_stage2, _ = model(feature, prior)
        handle.remove()

        self.assertEqual(len(seen), 1)
        h_batch = seen[0]
        self.assertEqual(h_batch.shape, (4, 128))

        with torch.no_grad():
            z_q = model.quantizer(h_batch)[0]
        self.assertEqual(z_q.shape, (4, 128))
        # Zero-initialized adapter: z_conf is bitwise equal to z_q.
        self.assertTrue(torch.equal(z_stage2, z_q))
        self.assertEqual(y_pred.shape, (4,))

    def test_quantization_error_matches_required_definition(self):
        h = torch.randn(6, 128)
        z_q = torch.randn(6, 128)
        actual = GenerateReturn.quantization_error(h, z_q)
        expected = torch.mean((h - z_q) ** 2, dim=-1, keepdim=True)
        self.assertEqual(actual.shape, (6, 1))
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(actual.requires_grad)


class Stage1FreezeTests(unittest.TestCase):
    """Stage 1 (encoder/quantizer/revin) must stay frozen and in eval mode."""

    def setUp(self):
        self.model = build_model().eval()
        self.feature = torch.randn(11, 5, 8)
        self.prior = torch.randn(11, 3)

    def test_stage1_parameters_frozen_and_eval(self):
        for module in (self.model.encoder, self.model.quantizer, self.model.revin):
            self.assertFalse(module.training)
            for name, parameter in module.named_parameters():
                self.assertFalse(parameter.requires_grad, msg=name)

    def test_backward_isolates_stage1_and_trains_adapter(self):
        model = self.model
        model.train()
        # The train() override must keep frozen Stage 1 modules in eval mode.
        for module in (model.encoder, model.quantizer, model.revin):
            self.assertFalse(module.training)

        y_pred, _, _, _, aux_loss = model(self.feature, self.prior)
        label = torch.randn(11)
        loss = model.rank_loss(y_pred, label) + model.aux_weight * aux_loss
        loss.backward()

        for module in (model.encoder, model.quantizer, model.revin):
            for name, parameter in module.named_parameters():
                self.assertIsNone(parameter.grad, msg=name)

        adapter = model.quantization_confidence_adapter
        self.assertIsNotNone(adapter.weight.grad)
        self.assertIsNotNone(adapter.bias.grad)
        self.assertTrue(torch.isfinite(adapter.weight.grad).all())
        self.assertTrue(torch.isfinite(adapter.bias.grad).all())
        self.assertGreater(adapter.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(adapter.bias.grad.abs().sum().item(), 0.0)


class Stage2StructureTests(unittest.TestCase):
    """Stage 2 must be the original 019 routed-only MoE (no Shared Expert)."""

    def setUp(self):
        self.model = build_model().eval()
        self.moe = self.model.loadings.fusion.moe

    def test_moe_is_factor_gated_without_shared_expert(self):
        moe = self.moe
        self.assertIsInstance(moe, FactorGatedMoE)
        # 010's always-on Shared Expert must be entirely absent: the module
        # is the original routed-only FactorGatedMoE from main/019.
        self.assertFalse(hasattr(moe, "shared_expert"))
        self.assertFalse(
            any("shared" in name.lower() for name, _ in moe.named_modules())
        )
        self.assertFalse(
            any("shared" in name.lower() for name, _ in moe.named_parameters())
        )

    def test_routed_configuration_unchanged(self):
        moe = self.moe
        self.assertEqual(moe.num_experts, 2)
        self.assertEqual(moe.k, 1)
        self.assertEqual(len(moe.experts), 2)
        for expert in moe.experts:
            self.assertIsInstance(expert, SimpleMLP)
        self.assertTrue(moe.noisy_gating)
        self.assertTrue(hasattr(moe, "gate"))
        self.assertTrue(hasattr(moe, "noise"))
        self.assertTrue(hasattr(moe, "W_h"))
        self.assertTrue(hasattr(moe, "softplus"))
        self.assertTrue(hasattr(moe, "mean"))
        self.assertTrue(hasattr(moe, "std"))

    def test_moe_output_is_pure_routed_combination(self):
        moe = self.moe.eval()
        x = torch.randn(6, moe.expert_input_size)
        z = torch.randn(6, moe.gate_input_size)
        with torch.no_grad():
            y, loss = moe(x, z)
        self.assertEqual(y.shape[0], x.shape[0])
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(loss))

    def test_confidence_adapter_is_the_only_extra_module(self):
        base = build_model(adapter=False, seed=11)
        adapted = build_model(adapter=True, seed=11)

        base_modules = {name for name, _ in base.named_modules()}
        adapted_modules = {name for name, _ in adapted.named_modules()}
        self.assertEqual(adapted_modules - base_modules, {"quantization_confidence_adapter"})
        self.assertEqual(base_modules - adapted_modules, set())

        adapter = adapted.quantization_confidence_adapter
        self.assertIsInstance(adapter, nn.Linear)
        self.assertEqual(adapter.in_features, 1)
        self.assertEqual(adapter.out_features, 8)
        self.assertEqual(torch.count_nonzero(adapter.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.bias).item(), 0)

    def test_zero_init_stage2_latent_bitwise_equal_z_q(self):
        model = build_model().eval()
        feature = torch.randn(9, 5, 8)
        with torch.no_grad():
            h_batch = model.encoder(model.revin(feature, mode="norm"))
            z_q = model.quantizer(h_batch)[0]
        z_stage2 = model.build_stage2_latent(h_batch, z_q)
        self.assertTrue(torch.equal(z_stage2, z_q))


if __name__ == "__main__":
    unittest.main()
