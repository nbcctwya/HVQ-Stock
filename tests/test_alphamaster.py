"""Temporal-market AlphaMaster architecture and canonical adapter tests."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

from dataset.dataset import init_data_loader
from dataset.schema import GROUP_SLICES, TOTAL_DIM
from module.alphamaster import MASTER
from trainer.train_alphamaster import AlphaMasterModule


ROOT = Path(__file__).resolve().parent.parent


def load_config(universe="csi300"):
    with (ROOT / "configs" / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config["data"]["universe"] = universe
    return config


class IndexedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, values, index):
        self.values = values
        self.index = index

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def get_index(self):
        return self.index


class LastSnapshot(torch.nn.Module):
    """007 market-state behavior, used only to verify backbone equivalence."""

    def forward(self, market_history):
        return market_history[:, -1, :]


class AlphaMasterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch = torch.randn(6, 20, TOTAL_DIM)

    def test_default_config_is_temporal_market_alphamaster(self):
        config = load_config()
        self.assertEqual(config["train"]["seed"], 0)
        self.assertEqual(config["train"]["learning_rate"], 8e-6)
        self.assertEqual(config["predictor"]["target_day"], 5)
        self.assertNotIn("vqvae", config)
        self.assertEqual(config["alphamaster"], {
            "d_feat": 158,
            "d_market": 63,
            "d_model": 256,
            "t_nhead": 4,
            "s_nhead": 2,
            "T_dropout_rate": 0.5,
            "S_dropout_rate": 0.5,
            "market_encoder": {
                "type": "gru",
                "input_size": 63,
                "hidden_size": 63,
                "num_layers": 1,
            },
            "beta": {"csi300": 10, "sp500": 5},
        })

    def test_canonical_inputs_prior_invariance_and_market_sensitivity(self):
        model = AlphaMasterModule(load_config()).eval()
        self.assertFalse(any("prior" in name for name, _ in model.named_parameters()))
        stock, market, target = model._get_data(self.batch)
        self.assertEqual(stock.shape, (6, 20, 158))
        self.assertEqual(market.shape, (6, 20, 63))
        self.assertEqual(target.shape, (6,))

        prediction = model(stock, market)
        changed_prior = self.batch.clone()
        changed_prior[..., GROUP_SLICES["prior"]] += 1_000_000
        changed_stock, changed_market, _ = model._get_data(changed_prior)
        torch.testing.assert_close(
            prediction, model(changed_stock, changed_market), rtol=0, atol=0
        )

        changed_market = market.clone()
        changed_market[:, -1, 0] += 100
        self.assertFalse(torch.equal(prediction, model(stock, changed_market)))

        changed_history = market.clone()
        changed_history[:, :-1, 0] += 100
        torch.testing.assert_close(
            changed_history[:, -1, :], market[:, -1, :], rtol=0, atol=0
        )
        self.assertFalse(torch.equal(prediction, model(stock, changed_history)))
        self.assertEqual(prediction.shape, (6,))

    def test_temporal_encoder_gate_contract_stock_path_and_gradients(self):
        model = AlphaMasterModule(load_config()).eval()
        stock, market, _ = model._get_data(self.batch)
        encoder = model.master.market_encoder
        self.assertEqual(encoder.gru.input_size, 63)
        self.assertEqual(encoder.gru.hidden_size, 63)
        self.assertEqual(encoder.gru.num_layers, 1)
        self.assertFalse(encoder.gru.bidirectional)

        market_state = encoder(market)
        self.assertEqual(market_state.shape, (6, 63))
        captured = {}

        def capture_gate(module, args):
            captured["gate_input"] = args[0].detach().clone()

        def capture_projection(module, args):
            captured["projection_input"] = args[0].detach().clone()

        gate_handle = model.master.feature_gate.register_forward_pre_hook(capture_gate)
        projection_handle = model.master.x2y.register_forward_pre_hook(capture_projection)
        prediction = model(stock, market)
        gate_handle.remove()
        projection_handle.remove()
        self.assertEqual(captured["gate_input"].shape, (6, 63))
        torch.testing.assert_close(captured["gate_input"], market_state)
        expected_stock = stock * model.master.feature_gate(market_state).unsqueeze(1)
        torch.testing.assert_close(captured["projection_input"], expected_stock)
        self.assertEqual(prediction.shape, (6,))

        model.train()
        model.zero_grad(set_to_none=True)
        model(stock, market).square().mean().backward()
        encoder_parameters = list(model.master.market_encoder.named_parameters())
        self.assertTrue(encoder_parameters)
        for name, parameter in encoder_parameters:
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_csi300_and_sp500_forward_with_expected_beta(self):
        for universe, beta in (("csi300", 10), ("sp500", 5)):
            with self.subTest(universe=universe):
                model = AlphaMasterModule(load_config(universe)).eval()
                stock, market, _ = model._get_data(self.batch)
                self.assertEqual(model.master.feature_gate.t, beta)
                self.assertEqual(model(stock, market).shape, (6,))

    def test_daily_sampler_emits_one_complete_cross_section(self):
        dates = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-03",
                                "2023-01-04", "2023-01-03", "2023-01-04"])
        instruments = ["A", "A", "B", "B", "C", "C"]
        index = pd.MultiIndex.from_arrays(
            [dates, instruments], names=["datetime", "instrument"]
        )
        dataset = IndexedTensorDataset(self.batch, index)
        loader, batches = init_data_loader(dataset, shuffle=False)
        self.assertEqual(batches, 2)
        emitted = loader.batch_sampler.ordered_indices()
        self.assertEqual(emitted.tolist(), [0, 2, 4, 1, 3, 5])
        for positions in loader.batch_sampler:
            batch_dates = index[positions].get_level_values("datetime")
            self.assertEqual(batch_dates.nunique(), 1)
            self.assertEqual(
                set(index[positions].get_level_values("instrument")),
                {"A", "B", "C"},
            )

    def test_checkpoint_load_is_strict(self):
        config = load_config()
        model = AlphaMasterModule(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ckpt"
            torch.save({"state_dict": model.state_dict()}, path)
            restored = AlphaMasterModule.load_strict_checkpoint(path, config)
            for expected, actual in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

            state = model.state_dict()
            state.pop(next(iter(state)))
            torch.save({"state_dict": state}, path)
            with self.assertRaises(RuntimeError):
                AlphaMasterModule.load_strict_checkpoint(path, config)

            legacy_state = {
                name: value for name, value in model.state_dict().items()
                if not name.startswith("master.market_encoder.")
            }
            torch.save({"state_dict": legacy_state}, path)
            with self.assertRaises(RuntimeError):
                AlphaMasterModule.load_strict_checkpoint(path, config)

    def test_core_forward_matches_standalone_alphamaster(self):
        source = ROOT.parent / "AlphaMaster" / "src" / "alphamaster" / "model.py"
        if not source.is_file():
            self.skipTest("standalone AlphaMaster checkout is unavailable")
        spec = importlib.util.spec_from_file_location("standalone_alphamaster", source)
        standalone = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standalone)

        reference = standalone.MASTER(beta=10).eval()
        adapted = MASTER(beta=10).eval()
        adapted.market_encoder = LastSnapshot()
        adapted.load_state_dict(reference.state_dict(), strict=True)
        features = torch.randn(5, 20, 221)
        torch.testing.assert_close(
            adapted(features), reference(features), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
