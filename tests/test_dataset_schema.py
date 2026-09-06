"""Regression coverage for return-block isolation and unchanged legacy sampling."""

import importlib
import contextlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml
from qlib.data.dataset import TSDataSampler

from dataset.market import MarketWindowSampler, market_feature_config
from dataset.prepare_schema_v2 import prepare
from dataset.schema import (
    MARKET_INDICES, SCHEMA_V1, SCHEMA_V2, dataset_basename, dataset_location, unpack_batch,
)

ROOT = Path(__file__).resolve().parent.parent


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.v1 = torch.arange(2 * 20 * SCHEMA_V1.total_dim).reshape(2, 20, -1).float()
        split = SCHEMA_V1.slices["label"].start
        self.v2 = torch.cat((self.v1[..., :split], torch.full((2, 20, 63), -999.),
                             self.v1[..., split:]), dim=-1)

    def test_layout_shapes_and_target(self):
        self.assertEqual(SCHEMA_V2.total_dim, 244)
        self.assertEqual(SCHEMA_V2.slices, {"feature": slice(0, 158), "prior": slice(158, 171),
                                          "market": slice(171, 234), "label": slice(234, 244)})
        a, b = unpack_batch(self.v1), unpack_batch(self.v2)
        self.assertEqual([tuple(x.shape) for x in b], [(2, 20, 158), (2, 13), (2, 20, 63), (2, 10)])
        self.assertIsNone(a.market_feature)
        for i in (0, 1, 3):
            torch.testing.assert_close(a[i], b[i], rtol=0, atol=0)
        torch.testing.assert_close(b.target(5), self.v1[:, -1, -6], rtol=0, atol=0)
        self.assertFalse((b.future_returns == -999).any())

    def test_single_stock_outer_loader_and_invalid_layout(self):
        for batch in (self.v2[:1], self.v2[:1][None]):
            self.assertEqual(unpack_batch(batch).future_returns.shape, (1, 10))
        for batch in (torch.zeros(2, 20, 243), self.v2[0], torch.zeros(2, 0, 244)):
            with self.assertRaises(ValueError):
                unpack_batch(batch)
        with self.assertRaises(ValueError):
            unpack_batch(self.v2, version=1)
        for day in (0, 11):
            with self.assertRaises(ValueError):
                unpack_batch(self.v2).target(day)

    def test_all_trainer_parsers(self):
        config = {"vqvae": {"num_features": 158, "num_prior_factors": 13,
                            "predictor": {"pred_len": 10}}}
        for name in ("train_vqvae", "train_ypred", "train_ypred_autoregressive",
                     "train_ypred_wo_moe", "train_ypred_wo_prior", "train_ypred_wo_stage1"):
            with self.subTest(trainer=name):
                module = importlib.import_module("trainer." + name)
                cls = module.FactorVQVAE if name == "train_vqvae" else module.GenerateReturn
                owner = SimpleNamespace(config=config, target_index=4)
                a = cls._get_data(owner, self.v1[:1], 0)
                b = cls._get_data(owner, self.v2[:1], 0)
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                expected = unpack_batch(self.v1[:1])
                torch.testing.assert_close(b[-1], expected.future_returns if name == "train_vqvae"
                                           else expected.target(5), rtol=0, atol=0)

    def test_default_paths_and_legacy_opt_in(self):
        cfg = {"window_size": 20, "data_path": "old"}
        for universe in MARKET_INDICES:
            self.assertEqual(dataset_location(cfg, universe, 10),
                             ("dataset/schema_v2_data", f"{universe}_20_h10_schema_v2"))
            self.assertEqual(dataset_location({**cfg, "schema_version": 1}, universe, 10),
                             ("old", f"{universe}_20_h10_dl2"))
        self.assertEqual(dataset_location(cfg, "csi500", 10), ("old", "csi500_20_h10_dl2"))

    def test_inference_reads_the_return_block(self):
        from utils.test import run_inference

        class Model(torch.nn.Module):
            def forward(self, stock, prior):
                return stock[:, -1, 0] + prior[:, 0], None, None, None, None

        class Loader(list):
            dataset = SimpleNamespace(get_index=lambda: pd.MultiIndex.from_product(
                [pd.to_datetime(["2023-01-03", "2023-01-04"]), ["A", "B"]],
                names=["datetime", "instrument"]))

        config = {"vqvae": {"num_features": 158, "num_prior_factors": 13},
                  "predictor": {"target_day": 5}}
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(divide="ignore", invalid="ignore"):
            before = run_inference(Model(), Loader([self.v1, self.v1 + 1]), config, "cpu")[0]
            after = run_inference(Model(), Loader([self.v2, self.v2 + 1]), config, "cpu")[0]
        pd.testing.assert_frame_equal(before, after, check_exact=True)
        np.testing.assert_array_equal(after.label.to_numpy(),
                                      torch.cat([unpack_batch(self.v1).target(5),
                                                 unpack_batch(self.v1 + 1).target(5)]).numpy())


class MarketSamplerTest(unittest.TestCase):
    def make_sampler(self, universe):
        dates = pd.bdate_range("2020-01-01", periods=45, name="datetime")
        index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
        values = np.random.default_rng(7).normal(size=(len(index), SCHEMA_V1.total_dim))
        df = pd.DataFrame(values, index=index)
        # Suspension and new-listing-style holes must not alter the market window.
        df = df.drop([(dates[3], "B"), (dates[29], "A")])
        old = TSDataSampler(df, start=dates[20], end=dates[-1], step_len=20,
                            fillna_type="ffill+bfill")
        fields, _ = market_feature_config(MARKET_INDICES[universe])
        daily = pd.DataFrame(np.arange(len(dates) * 63).reshape(-1, 63), index=dates,
                             columns=pd.MultiIndex.from_product([["market"], fields]))
        return old, daily, MarketWindowSampler(old, daily, universe)

    def test_both_universes_windows_groups_and_roundtrip(self):
        for universe in MARKET_INDICES:
            with self.subTest(universe=universe):
                old, daily, new = self.make_sampler(universe)
                self.assertEqual(new.columns.get_level_values(0).value_counts().to_dict(),
                                 {"feature": 158, "prior": 13, "market": 63, "label": 10})
                for date in new.get_index().get_level_values("datetime").unique():
                    ix = np.flatnonzero(new.get_index().get_level_values("datetime") == date)
                    before, after = unpack_batch(old[ix]), unpack_batch(new[ix])
                    for i in (0, 1, 3):
                        np.testing.assert_array_equal(before[i], after[i])
                    expected = daily.loc[:date].iloc[-20:].to_numpy()
                    for window in after.market_feature:
                        np.testing.assert_array_equal(window, expected)
                restored = pickle.loads(pickle.dumps(new))
                np.testing.assert_array_equal(new[0], restored[0])
                np.testing.assert_array_equal(new[0], new[new.get_index()[0]])

    def test_missing_calendar_fails(self):
        old, daily, _ = self.make_sampler("csi300")
        with self.assertRaises(ValueError):
            MarketWindowSampler(old, daily.iloc[1:], "csi300")

    def test_exact_formula_order_and_config_indices(self):
        for universe, indices in MARKET_INDICES.items():
            with (ROOT / f"dataset/2025_{universe}.yaml").open() as stream:
                cfg = yaml.safe_load(stream)
            self.assertEqual(tuple(cfg["market_data_handler_config"]["market_indices"]), indices)
            fields, names = market_feature_config(indices)
            expected = []
            for inst in indices:
                expected.append(f'Mask($close/Ref($close,1)-1, "{inst}")')
                for w in (5, 10, 20, 30, 60):
                    for expression in (f"Mean($close/Ref($close,1)-1,{w})",
                                       f"Std($close/Ref($close,1)-1,{w})",
                                       f"Mean($volume,{w})/$volume", f"Std($volume,{w})/$volume"):
                        expected.append(f'Mask({expression}, "{inst}")')
            self.assertEqual(fields, expected)
            self.assertEqual(names, fields)
            self.assertEqual(len(fields), 63)

    def test_conversion_refuses_overwrite_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CN").mkdir()
            for split in ("train", "valid", "test"):
                (root / "CN" / f"{dataset_basename('csi300', version=1)}_{split}.pkl").touch()
            target = root / "CN" / f"{dataset_basename('csi300')}_train.pkl"
            target.write_bytes(b"existing")
            with mock.patch("dataset.prepare_schema_v2.qlib.init") as init:
                with self.assertRaises(FileExistsError):
                    prepare("csi300", root, root)
                init.assert_not_called()
            self.assertEqual(target.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
