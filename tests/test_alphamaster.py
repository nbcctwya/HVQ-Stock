"""Pure AlphaMaster architecture and canonical adapter tests."""

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


class AlphaMasterPriorFactorHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch = torch.randn(6, 20, TOTAL_DIM)

    def test_default_config_is_prior_factor_head_experiment(self):
        config = load_config()
        self.assertEqual(config["train"]["seed"], 0)
        self.assertEqual(config["train"]["learning_rate"], 8e-6)
        self.assertEqual(config["predictor"]["target_day"], 5)
        self.assertNotIn("vqvae", config)
        self.assertEqual(config["alphamaster"], {
            "d_feat": 158,
            "d_market": 63,
            "num_prior_factors": 13,
            "d_model": 256,
            "t_nhead": 4,
            "s_nhead": 2,
            "T_dropout_rate": 0.5,
            "S_dropout_rate": 0.5,
            "beta": {"csi300": 10, "sp500": 5},
        })

    def test_canonical_inputs_factor_decomposition_and_market_sensitivity(self):
        model = AlphaMasterModule(load_config()).eval()
        stock, market, prior, target = model._get_data(self.batch)
        self.assertEqual(stock.shape, (6, 20, 158))
        self.assertEqual(market.shape, (6, 20, 63))
        self.assertEqual(prior.shape, (6, 13))
        self.assertEqual(target.shape, (6,))

        prediction, alpha, beta, prior_contribution, hidden = model(
            stock, market, prior, return_components=True
        )
        self.assertEqual(hidden.shape, (6, 256))
        self.assertEqual(alpha.shape, (6,))
        self.assertEqual(beta.shape, (6, 13))
        self.assertEqual(prior_contribution.shape, (6,))
        self.assertEqual(prediction.shape, (6,))
        torch.testing.assert_close(
            prior_contribution, torch.sum(beta * prior, dim=-1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            prediction, alpha + torch.sum(beta * prior, dim=-1), rtol=0, atol=0
        )

        changed_prior = self.batch.clone()
        changed_prior[..., GROUP_SLICES["prior"]] += 1
        changed_stock, changed_market, new_prior, _ = model._get_data(changed_prior)
        changed_prediction, changed_alpha, changed_beta, _, changed_hidden = model(
            changed_stock, changed_market, new_prior, return_components=True
        )
        torch.testing.assert_close(
            hidden, changed_hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(alpha, changed_alpha, rtol=0, atol=0)
        torch.testing.assert_close(beta, changed_beta, rtol=0, atol=0)
        self.assertFalse(torch.equal(prediction, changed_prediction))

        repeated = model(stock, market, prior, return_components=True)
        for expected, actual in zip(
            (prediction, alpha, beta, prior_contribution, hidden), repeated
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

        self.assertFalse(torch.equal(beta[0], beta[1]))
        torch.testing.assert_close(
            beta, model.master.prior_loading_head(hidden), rtol=0, atol=0
        )

        changed_market = market.clone()
        changed_market[:, -1, 0] += 100
        self.assertFalse(
            torch.equal(prediction, model(stock, changed_market, prior))
        )

    def test_csi300_and_sp500_forward_with_expected_beta(self):
        for universe, beta in (("csi300", 10), ("sp500", 5)):
            with self.subTest(universe=universe):
                model = AlphaMasterModule(load_config(universe)).eval()
                stock, market, prior, _ = model._get_data(self.batch)
                self.assertEqual(model.master.feature_gate.t, beta)
                self.assertEqual(model(stock, market, prior).shape, (6,))

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

            legacy_state = model.state_dict()
            alpha_weight = legacy_state.pop("master.alpha_head.weight")
            alpha_bias = legacy_state.pop("master.alpha_head.bias")
            legacy_state.pop("master.prior_loading_head.weight")
            legacy_state.pop("master.prior_loading_head.bias")
            legacy_state["master.decoder.weight"] = alpha_weight
            legacy_state["master.decoder.bias"] = alpha_bias
            torch.save({"state_dict": legacy_state}, path)
            with self.assertRaises(RuntimeError):
                AlphaMasterModule.load_strict_checkpoint(path, config)

    def test_backbone_hidden_matches_standalone_alphamaster(self):
        source = ROOT.parent / "AlphaMaster" / "src" / "alphamaster" / "model.py"
        if not source.is_file():
            self.skipTest("standalone AlphaMaster checkout is unavailable")
        spec = importlib.util.spec_from_file_location("standalone_alphamaster", source)
        standalone = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standalone)

        reference = standalone.MASTER(beta=10).eval()
        adapted = MASTER(beta=10).eval()
        adapted_state = adapted.state_dict()
        for name, value in reference.state_dict().items():
            if not name.startswith("decoder."):
                adapted_state[name] = value
        adapted.load_state_dict(adapted_state, strict=True)
        features = torch.randn(5, 20, 221)

        src = features[:, :, :reference.gate_input_start_index]
        gate_input = features[
            :, -1,
            reference.gate_input_start_index:reference.gate_input_end_index,
        ]
        src = src * torch.unsqueeze(reference.feature_gate(gate_input), dim=1)
        reference_hidden = reference.x2y(src)
        reference_hidden = reference.pe(reference_hidden)
        reference_hidden = reference.tatten(reference_hidden)
        reference_hidden = reference.satten(reference_hidden)
        reference_hidden = reference.temporalatten(reference_hidden)
        torch.testing.assert_close(
            adapted.encode(features), reference_hidden, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
