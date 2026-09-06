"""Canonical group, parser, calendar, and raw-factor regression coverage."""

import ast
import contextlib
import importlib
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
from omegaconf import OmegaConf
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP

from dataset.dataset import init_data_loader
from dataset.get_dataset import (
    Alpha158WithJKP, GlobalFactorMerger, build_factor_matrix, label_config,
    load_config, prepare_dataset, prepare_segment,
)
from dataset.market import CanonicalSampler, market_feature_config
from dataset.schema import (
    GROUP_DIMS, GROUP_SLICES, MARKET_INDICES, TOTAL_DIM, dataset_basename,
    unpack_batch, validate_columns,
)

ROOT = Path(__file__).resolve().parent.parent
TRAINERS = ("train_vqvae", "train_ypred", "train_ypred_autoregressive",
            "train_ypred_wo_moe", "train_ypred_wo_prior", "train_ypred_wo_stage1")


def sample_frame(universe):
    dates = pd.bdate_range("2020-01-01", periods=45, name="datetime")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
    columns = pd.MultiIndex.from_tuples([(g, f"{g}_{i}") for g, n in GROUP_DIMS.items() for i in range(n)])
    frame = pd.DataFrame(np.random.default_rng(7).normal(size=(len(index), TOTAL_DIM)), index=index, columns=columns)
    fields, _ = market_feature_config(MARKET_INDICES[universe])
    market = pd.DataFrame(np.arange(len(dates) * 63).reshape(-1, 63), index=dates,
                          columns=pd.MultiIndex.from_product([["market"], fields]))
    frame.columns = pd.MultiIndex.from_tuples([
        (g, fields[i] if g == "market" else f"{g}_{i}") for g, n in GROUP_DIMS.items() for i in range(n)])
    frame.loc[:, "market"] = market.loc[index.get_level_values("datetime")].to_numpy()
    # A stock gap must not shift the common market sequence.
    return frame.drop([(dates[3], "B"), (dates[29], "A")]), market


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.batch = torch.arange(2 * 20 * TOTAL_DIM).reshape(2, 20, -1).float()
        self.batch[..., GROUP_SLICES["market"]] = -999

    def test_layout_shapes_and_target(self):
        self.assertEqual(TOTAL_DIM, 244)
        self.assertEqual(GROUP_SLICES, {"feature": slice(0, 158), "prior": slice(158, 171),
                                       "market": slice(171, 234), "label": slice(234, 244)})
        parts = unpack_batch(self.batch)
        self.assertEqual([tuple(x.shape) for x in parts], [(2, 20, 158), (2, 13), (2, 20, 63), (2, 10)])
        torch.testing.assert_close(parts.target(5), self.batch[:, -1, 238], rtol=0, atol=0)
        self.assertFalse((parts.future_returns == -999).any())
        self.assertEqual(unpack_batch(self.batch[:1]).future_returns.shape, (1, 10))

    def test_reject_every_noncanonical_width(self):
        for width in (0, 158, 171, 181, 234, 243, 245, 307):
            with self.subTest(width=width), self.assertRaises(ValueError):
                unpack_batch(torch.zeros(2, 20, width))
        for batch in (self.batch[0], self.batch[None], torch.zeros(2, 0, 244)):
            with self.assertRaises(ValueError):
                unpack_batch(batch)
        for day in (0, 11, 2.5):
            with self.assertRaises(ValueError):
                unpack_batch(self.batch).target(day)
        with self.assertRaises(ValueError):
            init_data_loader([np.zeros((20, 181))], shuffle=False)

    def test_all_trainers_use_only_canonical_parser(self):
        for name in TRAINERS:
            with self.subTest(trainer=name):
                module = importlib.import_module("trainer." + name)
                cls = module.FactorVQVAE if name == "train_vqvae" else module.GenerateReturn
                owner = SimpleNamespace(target_index=4)
                with mock.patch.object(module, "unpack_batch", wraps=unpack_batch) as parser:
                    result = cls._get_data(owner, self.batch[:1], 0)
                    parser.assert_called_once()
                expected = unpack_batch(self.batch[:1])
                torch.testing.assert_close(result[0], expected.stock_feature, rtol=0, atol=0)
                torch.testing.assert_close(result[-1], expected.future_returns if name == "train_vqvae"
                                           else expected.target(5), rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    cls._get_data(owner, torch.zeros(2, 20, 181), 0)

    def test_inference_uses_canonical_returns_and_ignores_market(self):
        from utils import test as inference

        class Model(torch.nn.Module):
            def forward(self, stock, prior):
                return stock[:, -1, 0] + prior[:, 0], None, None, None, None

        class Loader(list):
            dataset = SimpleNamespace(get_index=lambda: pd.MultiIndex.from_product(
                [pd.to_datetime(["2023-01-03", "2023-01-04"]), ["A", "B"]],
                names=["datetime", "instrument"]))

        changed = self.batch.clone()
        changed[..., GROUP_SLICES["market"]] = 987654
        config = {"predictor": {"target_day": 5}}
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(divide="ignore", invalid="ignore"):
            with mock.patch.object(inference, "unpack_batch", wraps=unpack_batch) as parser:
                first = inference.run_inference(Model(), Loader([self.batch, self.batch + 1]), config, "cpu")[0]
                self.assertEqual(parser.call_count, 2)
            second = inference.run_inference(Model(), Loader([changed, changed + 1]), config, "cpu")[0]
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        np.testing.assert_array_equal(first.label.to_numpy(), torch.cat([
            unpack_batch(self.batch).target(5), unpack_batch(self.batch + 1).target(5)]).numpy())
        with self.assertRaises(ValueError):
            inference.run_inference(Model(), Loader([torch.zeros(2, 20, 181)]), config, "cpu")

    def test_only_one_filename_convention(self):
        for universe in MARKET_INDICES:
            self.assertEqual(dataset_basename(universe), f"{universe}_20_h10")
        for args in (("csi500",), ("csi300", 8), ("sp500", 20, 5)):
            with self.assertRaises(ValueError):
                dataset_basename(*args)


class GenerationTest(unittest.TestCase):
    def test_market_formulas_match_reference_order(self):
        for universe, indices in MARKET_INDICES.items():
            cfg = load_config(universe)
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
        # Read the reference method without importing/writing into AlphaMaster.
        path = ROOT.parent / "AlphaMaster/src/alphamaster/dataset.py"
        if path.exists():
            tree = ast.parse(path.read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "marketDataHandler")
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_feature_config")
            scope = {}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
            for indices in MARKET_INDICES.values():
                self.assertEqual(scope["get_feature_config"](SimpleNamespace(market_indices=indices)),
                                 market_feature_config(indices))

    def test_stock_and_prior_definitions_unchanged(self):
        self.assertIs(Alpha158WithJKP.get_feature_config, Alpha158.get_feature_config)
        dates = pd.bdate_range("2020-01-01", periods=25)
        raw = pd.DataFrame([{"date": d, "name": f"factor{i}", "ret": .01}
                            for d in dates for i in range(13)])
        factors = build_factor_matrix(raw, "ret == .01", dates, 20)
        self.assertTrue(factors.iloc[:20].isna().all().all())
        np.testing.assert_allclose(factors.iloc[20:], 1.01 ** 20 - 1)
        altered = raw.copy()
        altered.loc[altered.date == dates[20], "ret"] = .9
        altered_factors = build_factor_matrix(altered, "ret >= .01", dates, 20)
        np.testing.assert_array_equal(factors.iloc[20], altered_factors.iloc[20])
        self.assertEqual(label_config()[0][4], "Ref($close, -5) / Ref($close, -1) - 1")
        self.assertEqual(len(label_config()[0]), 10)

    def test_broadcast_prior(self):
        frame, market = sample_frame("csi300")
        values = pd.DataFrame(np.ones((len(market), 13)), index=market.index)
        merged = GlobalFactorMerger(values, "prior")(frame[["feature"]])
        self.assertEqual(merged["prior"].shape, (len(frame), 13))
        np.testing.assert_array_equal(merged["prior"], 1)

    def test_both_markets_canonical_sampler_and_dataloader(self):
        for universe in MARKET_INDICES:
            with self.subTest(universe=universe):
                frame, market = sample_frame(universe)
                sampler = CanonicalSampler(frame.copy(), market, universe, market.index[20], market.index[-1])
                self.assertEqual(sampler.data_arr.shape[-1], 244)
                validate_columns(sampler.columns)
                dates = sampler.get_index().get_level_values("datetime")
                for date in dates.unique():
                    indices = np.flatnonzero(dates == date)
                    parts = unpack_batch(sampler[indices])
                    expected = market.loc[:date].iloc[-20:].to_numpy()
                    np.testing.assert_array_equal(parts.market_feature,
                                                  np.broadcast_to(expected, parts.market_feature.shape))
                restored = pickle.loads(pickle.dumps(sampler))
                np.testing.assert_array_equal(sampler[0], restored[0])
                np.testing.assert_array_equal(sampler[0], sampler[sampler.get_index()[0]])
                loader, _ = init_data_loader(sampler, shuffle=False)
                self.assertEqual(unpack_batch(next(iter(loader))).market_feature.shape[1:], (20, 63))
                # __getitem__ must not mutate Qlib's stored rows.
                storage = sampler.data_arr.copy()
                sampler[0]
                np.testing.assert_array_equal(storage, sampler.data_arr)

    def test_bad_market_or_group_layout_fails(self):
        frame, market = sample_frame("csi300")
        for bad_market in (market.iloc[:, :-1], market.iloc[:, ::-1], market.iloc[1:], market * np.nan):
            with self.assertRaises(ValueError):
                CanonicalSampler(frame.copy(), bad_market, "csi300", market.index[20], market.index[-1])
        for bad in (frame.iloc[:, :-1], frame.iloc[:, ::-1]):
            with self.assertRaises(ValueError):
                CanonicalSampler(bad.copy(), market, "csi300", market.index[20], market.index[-1])

    def test_missing_inference_labels_are_preserved(self):
        frame, market = sample_frame("csi300")
        date = market.index[25]
        frame.loc[(date, "A"), ("label", "label_4")] = np.nan
        sampler = CanonicalSampler(frame, market, "csi300", date, date)
        parts = unpack_batch(sampler[[sampler.get_index().get_loc((date, "A"))]])
        self.assertTrue(np.isnan(parts.target(5)[0]))
        self.assertTrue(np.isfinite(parts.market_feature).all())

    def test_prepare_segments_use_correct_qlib_data_key(self):
        frame, market = sample_frame("csi300")
        handler = mock.Mock()
        handler.fetch.side_effect = lambda **kwargs: frame.copy()
        for key in (DataHandlerLP.DK_L, DataHandlerLP.DK_I):
            result = prepare_segment(handler, market, "csi300", [market.index[20], market.index[-1]], key)
            self.assertEqual(result[0].shape, (20, 244))
            self.assertEqual(handler.fetch.call_args.kwargs["col_set"], list(GROUP_DIMS))
            self.assertEqual(handler.fetch.call_args.kwargs["data_key"], key)

    def test_generation_does_not_read_pickles_and_refuses_overwrite(self):
        frame, market = sample_frame("csi300")
        cfg = load_config("csi300")
        cfg["dataset"]["segments"] = {s: [market.index[20], market.index[-1]] for s in ("train", "valid", "test")}
        handler = mock.Mock()
        handler.fetch.side_effect = lambda **kwargs: frame.copy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "prior.csv"
            csv.write_text("date,name,ret\n")
            with mock.patch("dataset.get_dataset.qlib.init"), mock.patch("dataset.get_dataset.build_handler", return_value=(handler, market)), mock.patch("pickle.load", side_effect=AssertionError("must not read pickle")):
                outputs = prepare_dataset("csi300", root, csv, cfg)
            self.assertEqual(len(outputs), 3)
            self.assertEqual([call.kwargs["data_key"] for call in handler.fetch.call_args_list],
                             [DataHandlerLP.DK_L, DataHandlerLP.DK_L, DataHandlerLP.DK_I])
            for path in outputs.values():
                with path.open("rb") as stream:
                    self.assertEqual(pickle.load(stream)[0].shape, (20, 244))
            with mock.patch("dataset.get_dataset.qlib.init") as init:
                with self.assertRaises(FileExistsError):
                    prepare_dataset("csi300", root, csv, cfg)
                init.assert_not_called()

    def test_stage_readers_use_the_same_canonical_files(self):
        stage1 = importlib.import_module("stage1")
        stage2 = importlib.import_module("stage2")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = OmegaConf.create({"data": {"data_path": tmp, "window_size": 20,
                                               "return_horizon": 10},
                                    "train": {"num_workers": 0}})
            for universe, region in (("csi300", "CN"), ("sp500", "US")):
                frame, market = sample_frame(universe)
                sampler = CanonicalSampler(frame, market, universe, market.index[20], market.index[-1])
                (Path(tmp) / region).mkdir()
                for split in ("train", "valid", "test"):
                    with (Path(tmp) / region / f"{dataset_basename(universe)}_{split}.pkl").open("wb") as stream:
                        pickle.dump(sampler, stream)
                first = stage1._load_canonical_datasets(cfg, region, universe)
                self.assertTrue(all(dataset[0].shape == (20, 244) for dataset in first))
                test_dataset = stage2._load_canonical_test_dataset(cfg, region, universe)
                loader, _ = init_data_loader(test_dataset, shuffle=False)
                self.assertEqual(unpack_batch(next(iter(loader))).market_feature.shape[1:], (20, 63))


if __name__ == "__main__":
    unittest.main()
