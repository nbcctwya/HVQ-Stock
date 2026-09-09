"""Tests for experiment 028's Stage 1 temporal-attention replacement."""

import ast
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from module.layers.encoder import (
    CrossAssetTransformerEncoder,
    PositionalEncoding,
    SpatialEncoder,
    TAttention,
    TemporalAttention,
    TemporalAttentionEncoder,
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
        encoder_type="temporal-attention",
        temporal_dropout=dropout,
    )


def class_nodes(path):
    tree = ast.parse(path.read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


class TemporalAttentionStage1Test(unittest.TestCase):
    def test_default_config_fully_selects_experiment(self):
        with (ROOT / "configs" / "config.yaml").open() as stream:
            config = yaml.safe_load(stream)

        encoder = config["vqvae"]["encoder"]
        self.assertEqual(encoder["type"], "temporal-attention")
        self.assertEqual(encoder["num_heads"], 2)
        self.assertEqual(encoder["num_layers"], 1)
        self.assertEqual(encoder["temporal_dropout"], 0.1)
        self.assertEqual(config["vqvae"]["hidden_size"], 128)
        self.assertEqual(config["vqvae"]["vq_embed_dim"], 128)
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_alpha_master_classes_are_exact_copies(self):
        source_path = ROOT.parent / "AlphaMaster" / "src" / "alphamaster" / "model.py"
        copied_path = ROOT / "module" / "layers" / "encoder.py"
        self.assertTrue(source_path.is_file())
        source = class_nodes(source_path)
        copied = class_nodes(copied_path)

        for name in ("PositionalEncoding", "TAttention", "TemporalAttention"):
            self.assertEqual(
                ast.dump(source[name], include_attributes=False),
                ast.dump(copied[name], include_attributes=False),
                f"{name} differs from AlphaMaster source",
            )

    def test_only_gru_summarization_is_replaced(self):
        encoder = build_encoder()
        module_types = tuple(type(module) for module in encoder.modules())

        self.assertNotIn(nn.GRU, module_types)
        self.assertIn(nn.TransformerEncoder, module_types)
        self.assertIn(nn.TransformerEncoderLayer, module_types)
        self.assertIsInstance(
            encoder.cross_asset_transformer, CrossAssetTransformerEncoder
        )
        self.assertFalse(any(type(module).__name__ == "SAttention" for module in encoder.modules()))
        self.assertFalse(any("gate" in name.lower() for name, _ in encoder.named_modules()))

    def test_required_modules_and_dimensions(self):
        encoder = build_encoder()
        temporal = encoder.temporal_encoder

        self.assertIsInstance(temporal, TemporalAttentionEncoder)
        self.assertIsInstance(temporal.input_projection, nn.Linear)
        self.assertEqual(
            (temporal.input_projection.in_features, temporal.input_projection.out_features),
            (8, 8),
        )
        self.assertIsInstance(temporal.positional_encoding, PositionalEncoding)
        self.assertIsInstance(temporal.temporal_attention, TAttention)
        self.assertIsInstance(temporal.temporal_aggregation, TemporalAttention)

    def test_preserved_feature_transform(self):
        encoder = build_encoder()
        transform = encoder.feature_transform

        self.assertIsInstance(transform.linear, nn.Linear)
        self.assertEqual(
            (transform.linear.in_features, transform.linear.out_features), (8, 8)
        )
        self.assertIsInstance(transform.normalize, nn.LayerNorm)
        self.assertIsInstance(transform.leakyrelu, nn.LeakyReLU)

        inputs = torch.randn(3, 5, 8)
        expected = transform.leakyrelu(transform.normalize(transform.linear(inputs)))
        self.assertTrue(torch.equal(transform(inputs), expected))

    def test_required_data_flow_and_interface_shapes(self):
        encoder = build_encoder().eval()
        seen = []
        shapes = {}

        def record(name):
            def hook(_module, inputs, output):
                seen.append(name)
                shapes[name] = (tuple(inputs[0].shape), tuple(output.shape))
            return hook

        modules = [
            ("feature_linear", encoder.feature_transform.linear),
            ("feature_norm", encoder.feature_transform.normalize),
            ("feature_activation", encoder.feature_transform.leakyrelu),
            ("input_projection", encoder.temporal_encoder.input_projection),
            ("positional_encoding", encoder.temporal_encoder.positional_encoding),
            ("tattention", encoder.temporal_encoder.temporal_attention),
            ("temporal_attention", encoder.temporal_encoder.temporal_aggregation),
            ("cross_asset_transformer", encoder.cross_asset_transformer),
        ]
        handles = [module.register_forward_hook(record(name)) for name, module in modules]
        try:
            output = encoder(torch.randn(7, 5, 8))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(seen, [name for name, _ in modules])
        self.assertEqual(shapes["input_projection"], ((7, 5, 8), (7, 5, 8)))
        self.assertEqual(shapes["tattention"], ((7, 5, 8), (7, 5, 8)))
        self.assertEqual(shapes["temporal_attention"], ((7, 5, 8), (7, 8)))
        self.assertEqual(shapes["cross_asset_transformer"][0], (7, 8))
        self.assertEqual(output.shape, (7, 6))

    def test_temporal_encoder_is_asset_independent_before_cross_asset_stage(self):
        torch.manual_seed(17)
        encoder = build_encoder().eval()
        inputs = torch.randn(4, 5, 8)
        changed = inputs.clone()
        changed[1:] = changed[1:] + 4.0

        with torch.no_grad():
            original_temporal = encoder.temporal_encoder(
                encoder.feature_transform(inputs)
            )[0]
            changed_temporal = encoder.temporal_encoder(
                encoder.feature_transform(changed)
            )[0]
            original_output = encoder(inputs)[0]
            changed_output = encoder(changed)[0]

        self.assertTrue(torch.equal(original_temporal, changed_temporal))
        self.assertFalse(torch.equal(original_output, changed_output))

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

    def test_invalid_encoder_type_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "requires.*temporal-attention"):
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
