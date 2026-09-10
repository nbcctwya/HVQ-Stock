"""Tests for experiment 032's market-gated spatial-first MASTER Stage 1 encoder."""

import ast
import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module.autoencoder import VQVAE
from module.layers.encoder import (
    Gate,
    MASTERStyleEncoder,
    PositionalEncoding,
    SAttention,
    SpatialEncoder,
    TAttention,
    TemporalAttention,
)
from module.quantise import VectorQuantiser


ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with (ROOT / "configs" / "config.yaml").open() as stream:
        return yaml.safe_load(stream)


def build_encoder(dropout=0.0, beta=10):
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
        market_input_dim=3,
        market_beta=beta,
    )


def tiny_vqvae_config():
    config = copy.deepcopy(load_config())
    config["vqvae"].update({
        "num_features": 8,
        "seq_len": 8,
        "hidden_size": 8,
        "num_prior_factors": 3,
        "vq_embed_dim": 6,
        "num_embed": 16,
    })
    config["vqvae"]["encoder"].update({
        "num_heads": 2,
        "num_layers": 1,
        "temporal_num_heads": 2,
        "spatial_num_heads": 2,
        "temporal_dropout": 0.0,
        "spatial_dropout": 0.0,
        "market_gate": {
            "input_dim": 3,
            "beta": {"csi300": 10, "sp500": 5},
        },
    })
    config["vqvae"]["decoder"].update({"initial_T": 2, "hidden_channels": 8})
    config["vqvae"]["predictor"].update({"pred_len": 4, "dropout": 0.0})
    return config


class MarketGatedSpatialFirstStage1EncoderTest(unittest.TestCase):
    def test_default_config_fully_selects_experiment(self):
        config = load_config()
        encoder = config["vqvae"]["encoder"]

        self.assertEqual(encoder["type"], "master")
        self.assertEqual(encoder["num_layers"], 1)
        self.assertEqual(encoder["temporal_num_heads"], 2)
        self.assertEqual(encoder["spatial_num_heads"], 2)
        self.assertEqual(encoder["temporal_dropout"], 0.1)
        self.assertEqual(encoder["spatial_dropout"], 0.1)
        self.assertEqual(encoder["market_gate"], {
            "input_dim": 63,
            "beta": {"csi300": 10, "sp500": 5},
        })
        self.assertEqual(config["vqvae"]["num_features"], 158)
        self.assertEqual(config["vqvae"]["hidden_size"], 128)
        self.assertEqual(config["vqvae"]["vq_embed_dim"], 128)
        self.assertEqual(config["vqvae"]["num_embed"], 512)
        self.assertEqual(config["train"]["seed"], 0)

    def test_gate_is_exact_alphamaster_copy(self):
        source_path = ROOT.parent / "AlphaMaster" / "src" / "alphamaster" / "model.py"
        if not source_path.is_file():
            self.skipTest("standalone AlphaMaster checkout is unavailable")

        def class_node(path, name):
            tree = ast.parse(path.read_text())
            return next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == name
            )

        source = class_node(source_path, "Gate")
        copied = class_node(ROOT / "module" / "layers" / "encoder.py", "Gate")
        self.assertEqual(
            ast.dump(source, include_attributes=False),
            ast.dump(copied, include_attributes=False),
        )

    def test_gate_matches_experiment_031_exactly(self):
        def class_node(tree):
            return next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Gate"
            )

        import subprocess
        blob_031 = subprocess.run(
            ["git", "show",
             "exp/031-prism-market-gated-stage1-encoder:module/layers/encoder.py"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout
        gate_031 = class_node(ast.parse(blob_031))
        gate_032 = class_node(
            ast.parse((ROOT / "module" / "layers" / "encoder.py").read_text())
        )
        self.assertEqual(
            ast.dump(gate_031, include_attributes=False),
            ast.dump(gate_032, include_attributes=False),
        )

    def test_gate_canonical_shapes_and_weight_sum(self):
        gate_module = Gate(63, 158, beta=10).eval()
        market_current = torch.randn(7, 63)
        with torch.no_grad():
            gate = gate_module(market_current)

        self.assertEqual(market_current.shape, (7, 63))
        self.assertEqual(gate.shape, (7, 158))
        torch.testing.assert_close(
            gate.sum(dim=-1), torch.full((7,), 158.0), rtol=1e-6, atol=1e-5
        )

    def test_csi300_and_sp500_beta_rules_reach_gate(self):
        for universe, beta in (("csi300", 10), ("sp500", 5)):
            with self.subTest(universe=universe):
                config = tiny_vqvae_config()
                config["data"]["universe"] = universe
                model = VQVAE(config)
                self.assertEqual(model.market_beta, beta)
                self.assertEqual(model.spatial_encoder.feature_gate.t, beta)

    def test_only_market_gate_is_added_to_experiment_029_encoder(self):
        encoder = build_encoder()
        module_types = tuple(type(module) for module in encoder.modules())

        self.assertNotIn(nn.GRU, module_types)
        self.assertNotIn(nn.TransformerEncoder, module_types)
        self.assertNotIn(nn.TransformerEncoderLayer, module_types)
        self.assertIsInstance(encoder.feature_gate, Gate)
        self.assertEqual(encoder.feature_gate.trans.in_features, 3)
        self.assertEqual(encoder.feature_gate.trans.out_features, 8)

    def test_alpha_master_attention_components_are_unchanged(self):
        encoder = build_encoder()
        master = encoder.master_encoder

        self.assertIsInstance(master, MASTERStyleEncoder)
        self.assertIsInstance(master.pe, PositionalEncoding)
        self.assertIsInstance(master.tatten, TAttention)
        self.assertIsInstance(master.satten, SAttention)
        self.assertIsInstance(master.temporalatten, TemporalAttention)
        self.assertEqual(master.x2y.in_features, 8)
        self.assertEqual(master.x2y.out_features, 8)

    def test_revin_gate_feature_transform_and_attention_order(self):
        model = VQVAE(tiny_vqvae_config()).eval()
        encoder = model.spatial_encoder
        seen = []
        shapes = {}

        def record(name):
            def hook(_module, inputs, output):
                seen.append(name)
                shapes[name] = {
                    "input": tuple(inputs[0].shape),
                    "output": tuple(output.shape),
                }
            return hook

        modules = [
            ("revin", model.revin),
            ("market_gate", encoder.feature_gate),
            ("front_linear", encoder.feature_transform.linear),
            ("front_norm", encoder.feature_transform.normalize),
            ("front_activation", encoder.feature_transform.leakyrelu),
            ("input_projection", encoder.master_encoder.x2y),
            ("positional_encoding", encoder.master_encoder.pe),
            ("spatial_attention", encoder.master_encoder.satten),
            ("temporal_attention", encoder.master_encoder.tatten),
            ("temporal_aggregation", encoder.master_encoder.temporalatten),
            ("projection_mlp", encoder.out_layer),
        ]
        handles = [module.register_forward_hook(record(name)) for name, module in modules]
        try:
            feature = torch.randn(7, 8, 8)
            market = torch.randn(7, 8, 3)
            normalized = model.revin(feature, mode="norm")
            output = encoder(normalized, market)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(seen, [name for name, _ in modules])
        self.assertEqual(shapes["market_gate"]["input"], (7, 3))
        self.assertEqual(shapes["market_gate"]["output"], (7, 8))
        self.assertEqual(shapes["spatial_attention"]["input"], (7, 8, 8))
        self.assertEqual(shapes["temporal_attention"]["input"], (7, 8, 8))
        self.assertEqual(shapes["temporal_aggregation"]["input"], (7, 8, 8))
        self.assertEqual(shapes["projection_mlp"]["input"], (7, 8))
        self.assertEqual(output.shape, (7, 6))

    def test_attention_order_stays_spatial_first_with_gate(self):
        torch.manual_seed(29)
        encoder = build_encoder().eval()
        inputs = torch.randn(7, 5, 8)
        market = torch.randn(7, 5, 3)

        with torch.no_grad():
            actual = encoder(inputs, market)
            gate = encoder.feature_gate(market[:, -1, :])
            transformed = encoder.feature_transform(inputs * gate.unsqueeze(1))
            positioned = encoder.master_encoder.pe(
                encoder.master_encoder.x2y(transformed)
            )
            spatial_first = encoder.master_encoder.temporalatten(
                encoder.master_encoder.tatten(
                    encoder.master_encoder.satten(positioned)
                )
            )
            temporal_first = encoder.master_encoder.temporalatten(
                encoder.master_encoder.satten(
                    encoder.master_encoder.tatten(positioned)
                )
            )

        self.assertTrue(torch.equal(actual, encoder.out_layer(spatial_first)))
        self.assertFalse(
            torch.equal(
                encoder.out_layer(spatial_first),
                encoder.out_layer(temporal_first),
            )
        )

    def test_only_last_market_timestep_controls_gate_and_latent(self):
        torch.manual_seed(23)
        encoder = build_encoder().eval()
        stock = torch.randn(6, 8, 8)
        market = torch.randn(6, 8, 3)
        changed_history = market.clone()
        changed_history[:, :-1, :] += 1000
        changed_current = market.clone()
        changed_current[:, -1, 0] += 1000

        with torch.no_grad():
            base_gate = encoder.feature_gate(market[:, -1, :])
            history_gate = encoder.feature_gate(changed_history[:, -1, :])
            current_gate = encoder.feature_gate(changed_current[:, -1, :])
            base_latent = encoder(stock, market)
            history_latent = encoder(stock, changed_history)
            current_latent = encoder(stock, changed_current)

        self.assertTrue(torch.equal(base_gate, history_gate))
        self.assertTrue(torch.equal(base_latent, history_latent))
        self.assertFalse(torch.equal(base_gate, current_gate))
        self.assertFalse(torch.equal(base_latent, current_latent))

    def test_prior_changes_do_not_change_gate(self):
        torch.manual_seed(29)
        model = VQVAE(tiny_vqvae_config()).eval()
        stock = torch.randn(6, 8, 8)
        market = torch.randn(6, 8, 3)
        returns = torch.randn(6, 4)
        prior = torch.randn(6, 3)
        seen_gates = []
        handle = model.spatial_encoder.feature_gate.register_forward_hook(
            lambda _module, _inputs, output: seen_gates.append(output.detach().clone())
        )
        try:
            with torch.no_grad():
                model(stock, prior, market, returns)
                model(stock, prior + 1000, market, returns)
        finally:
            handle.remove()

        self.assertEqual(len(seen_gates), 2)
        self.assertTrue(torch.equal(seen_gates[0], seen_gates[1]))

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
        market = torch.randn(9, 5, 3)
        latent = encoder(inputs, market)
        quantized, vq_loss, (_, _, indices) = quantizer(latent)
        (quantized.square().mean() + vq_loss).backward()

        self.assertEqual(latent.shape, (9, 6))
        self.assertEqual(quantized.shape, (9, 6))
        self.assertEqual(indices.shape[0], 9)
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)

    def test_spatial_attention_couples_assets(self):
        torch.manual_seed(37)
        encoder = build_encoder().eval()
        inputs = torch.randn(4, 5, 8)
        market = torch.randn(4, 5, 3)
        changed = inputs.clone()
        changed[1:] = changed[1:] + 4.0

        with torch.no_grad():
            original_first = encoder(inputs, market)[0]
            changed_first = encoder(changed, market)[0]

        self.assertFalse(torch.equal(original_first, changed_first))

    def test_state_dict_strict_round_trip(self):
        torch.manual_seed(41)
        source = build_encoder().eval()
        target = build_encoder().eval()
        result = target.load_state_dict(source.state_dict(), strict=True)
        inputs = torch.randn(5, 5, 8)
        market = torch.randn(5, 5, 3)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        with torch.no_grad():
            self.assertTrue(torch.equal(
                source(inputs, market), target(inputs, market)
            ))

    def test_invalid_market_shapes_fail_loudly(self):
        encoder = build_encoder()
        stock = torch.randn(4, 5, 8)
        for market in (
            torch.randn(4, 3),
            torch.randn(4, 4, 3),
            torch.randn(4, 5, 2),
        ):
            with self.subTest(shape=tuple(market.shape)), self.assertRaises(ValueError):
                encoder(stock, market)

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
                market_input_dim=3,
                market_beta=10,
            )


if __name__ == "__main__":
    unittest.main()
