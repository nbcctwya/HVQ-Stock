"""Tests for experiment 027's MASTER-style Stage 1 encoder."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module.layers.encoder import (
    MASTERStyleEncoder,
    PositionalEncoding,
    SAttention,
    SpatialEncoder,
    TAttention,
    TemporalAttention,
)
from module.quantise import VectorQuantiser


def build_encoder(dropout=0.0):
    return SpatialEncoder(
        input_features_C=8,
        T_window=5,
        gru_hidden_size=8,
        num_transformer_heads=2,
        num_transformer_layers=1,
        final_embed_dim_d=6,
        encoder_type="master",
        temporal_num_heads=2,
        spatial_num_heads=2,
        temporal_dropout=dropout,
        spatial_dropout=dropout,
    )


class MASTERStage1EncoderTest(unittest.TestCase):
    def test_default_config_fully_selects_experiment(self):
        config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)

        encoder = config["vqvae"]["encoder"]
        self.assertEqual(encoder["type"], "master")
        self.assertEqual(encoder["num_layers"], 1)
        self.assertEqual(encoder["temporal_num_heads"], 2)
        self.assertEqual(encoder["spatial_num_heads"], 2)
        self.assertEqual(encoder["temporal_dropout"], 0.1)
        self.assertEqual(encoder["spatial_dropout"], 0.1)
        self.assertEqual(config["vqvae"]["hidden_size"], 128)
        self.assertEqual(config["vqvae"]["vq_embed_dim"], 128)
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_only_gru_and_cross_asset_transformer_are_removed(self):
        encoder = build_encoder()
        module_types = tuple(type(module) for module in encoder.modules())

        self.assertNotIn(nn.GRU, module_types)
        self.assertNotIn(nn.TransformerEncoder, module_types)
        self.assertNotIn(nn.TransformerEncoderLayer, module_types)
        self.assertFalse(any("gate" in name.lower() for name, _ in encoder.named_modules()))
        self.assertFalse(any("gate" in name.lower() for name, _ in encoder.named_parameters()))

    def test_alpha_master_components_are_used(self):
        encoder = build_encoder()
        master = encoder.master_encoder

        self.assertIsInstance(master, MASTERStyleEncoder)
        self.assertIsInstance(master.pe, PositionalEncoding)
        self.assertIsInstance(master.tatten, TAttention)
        self.assertIsInstance(master.satten, SAttention)
        self.assertIsInstance(master.temporalatten, TemporalAttention)
        self.assertEqual(master.x2y.in_features, 8)
        self.assertEqual(master.x2y.out_features, 8)

    def test_preserved_feature_transform_and_projection_mlp(self):
        encoder = build_encoder()
        transform = encoder.feature_transform

        self.assertIsInstance(transform.linear, nn.Linear)
        self.assertEqual((transform.linear.in_features, transform.linear.out_features), (8, 8))
        self.assertIsInstance(transform.normalize, nn.LayerNorm)
        self.assertIsInstance(transform.leakyrelu, nn.LeakyReLU)
        self.assertEqual(len(encoder.out_layer), 3)
        self.assertIsInstance(encoder.out_layer[0], nn.Linear)
        self.assertEqual((encoder.out_layer[0].in_features, encoder.out_layer[0].out_features), (8, 32))
        self.assertIsInstance(encoder.out_layer[1], nn.GELU)
        self.assertIsInstance(encoder.out_layer[2], nn.Linear)
        self.assertEqual((encoder.out_layer[2].in_features, encoder.out_layer[2].out_features), (32, 6))

    def test_required_data_flow_order_and_full_time_axis(self):
        encoder = build_encoder().eval()
        seen = []
        shapes = {}

        def record(name):
            def hook(_module, inputs, _output):
                seen.append(name)
                shapes[name] = tuple(inputs[0].shape)
            return hook

        modules = [
            ("front_linear", encoder.feature_transform.linear),
            ("front_norm", encoder.feature_transform.normalize),
            ("front_activation", encoder.feature_transform.leakyrelu),
            ("input_projection", encoder.master_encoder.x2y),
            ("positional_encoding", encoder.master_encoder.pe),
            ("temporal_attention", encoder.master_encoder.tatten),
            ("spatial_attention", encoder.master_encoder.satten),
            ("temporal_aggregation", encoder.master_encoder.temporalatten),
            ("projection_mlp", encoder.out_layer),
        ]
        handles = [module.register_forward_hook(record(name)) for name, module in modules]
        try:
            output = encoder(torch.randn(7, 5, 8))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(seen, [name for name, _ in modules])
        self.assertEqual(shapes["temporal_attention"], (7, 5, 8))
        self.assertEqual(shapes["spatial_attention"], (7, 5, 8))
        self.assertEqual(shapes["temporal_aggregation"], (7, 5, 8))
        self.assertEqual(shapes["projection_mlp"], (7, 8))
        self.assertEqual(output.shape, (7, 6))

    def test_output_connects_to_unchanged_vector_quantiser(self):
        encoder = build_encoder().train()
        quantizer = VectorQuantiser(
            num_embed=16,
            embed_dim=6,
            beta=0.25,
            distance="l2",
            anchor="probrandom",
            first_batch=False,
            contras_loss=True,
        )
        inputs = torch.randn(9, 5, 8, requires_grad=True)
        latent = encoder(inputs)
        quantized, vq_loss, (_, _, indices) = quantizer(latent)
        (quantized.square().mean() + vq_loss).backward()

        self.assertEqual(latent.shape, (9, 6))
        self.assertEqual(quantized.shape, (9, 6))
        self.assertEqual(indices.shape[0], 9)
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)

    def test_spatial_attention_couples_assets(self):
        torch.manual_seed(23)
        encoder = build_encoder().eval()
        inputs = torch.randn(4, 5, 8)
        changed = inputs.clone()
        changed[1:] = changed[1:] + 4.0

        with torch.no_grad():
            original_first = encoder(inputs)[0]
            changed_first = encoder(changed)[0]

        self.assertFalse(torch.equal(original_first, changed_first))

    def test_state_dict_strict_round_trip(self):
        torch.manual_seed(31)
        source = build_encoder().eval()
        target = build_encoder().eval()
        result = target.load_state_dict(source.state_dict(), strict=True)
        inputs = torch.randn(5, 5, 8)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        with torch.no_grad():
            self.assertTrue(torch.equal(source(inputs), target(inputs)))

    def test_invalid_non_master_config_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "requires.*master"):
            SpatialEncoder(
                input_features_C=8,
                T_window=5,
                gru_hidden_size=8,
                num_transformer_heads=2,
                num_transformer_layers=1,
                final_embed_dim_d=6,
                encoder_type="gru-transformer",
            )


if __name__ == "__main__":
    unittest.main()
