"""Tests for the formal evaluation layer (Baseline Results Protocol v1.0)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import protocol
from evaluation.backtest import load_prediction, make_signal
from evaluation.discovery import discover
from evaluation.ensemble import ensemble_scores
from evaluation.metrics import portfolio_metrics, ranking_metrics


def _pred_frame(dates, instruments, score_fn, label_fn):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), instruments], names=["datetime", "instrument"]
    )
    score = [score_fn(d, i) for d, i in idx]
    label = [label_fn(d, i) for d, i in idx]
    return pd.DataFrame({"score": score, "label": label}, index=idx, dtype=float)


class TestPortfolioMetrics(unittest.TestCase):
    def test_first_day_minus_10pct_mdd(self):
        r = pd.Series([-0.10, 0.02, 0.01])
        m = portfolio_metrics(r)
        self.assertAlmostEqual(m["MDD"], -0.10, places=12)

    def test_sortino_two_identical_negative_days(self):
        # Downside deviation is the root-mean-square over ALL days, so two
        # identical negative days give DownDev = |g|, not 0 -> Sortino defined.
        r = pd.Series([-0.01, -0.01])
        g = np.log1p(-0.01)
        m = portfolio_metrics(r)
        expected_dd = abs(g)
        expected_sortino = np.sqrt(252) * g / expected_dd
        self.assertTrue(np.isfinite(m["Sortino"]))
        self.assertAlmostEqual(m["Sortino"], expected_sortino, places=12)

    def test_sortino_mixed_days_uses_all_days(self):
        r = pd.Series([-0.02, 0.01, -0.01, 0.03])
        g = np.log1p(r)
        dd = np.sqrt(np.mean(np.minimum(g, 0.0) ** 2))
        expected = np.sqrt(252) * g.mean() / dd
        m = portfolio_metrics(r)
        self.assertAlmostEqual(m["Sortino"], expected, places=12)

    def test_return_le_minus_one_raises(self):
        with self.assertRaises(ValueError):
            portfolio_metrics(pd.Series([-1.0, 0.01]))
        with self.assertRaises(ValueError):
            portfolio_metrics(pd.Series([-1.5]))

    def test_independent_recomputation(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.0005, 0.012, size=500))
        m = portfolio_metrics(r)

        g = np.log1p(r.values)
        ar = np.exp(g.mean() * 252) - 1
        std = g.std(ddof=1) * np.sqrt(252)
        nav = np.concatenate([[1.0], np.exp(np.cumsum(g))])
        mdd = np.min(nav / np.maximum.accumulate(nav) - 1)
        sharpe = np.sqrt(252) * g.mean() / g.std(ddof=1)
        dd = np.sqrt(np.mean(np.minimum(g, 0.0) ** 2))
        sortino = np.sqrt(252) * g.mean() / dd
        calmar = ar / abs(mdd)

        self.assertAlmostEqual(m["AR"], ar, places=12)
        self.assertAlmostEqual(m["STD"], std, places=12)
        self.assertAlmostEqual(m["MDD"], mdd, places=12)
        self.assertAlmostEqual(m["Sharpe"], sharpe, places=12)
        self.assertAlmostEqual(m["Sortino"], sortino, places=12)
        self.assertAlmostEqual(m["Calmar"], calmar, places=12)
        self.assertEqual(m["num_test_days"], 500)

    def test_mdd_peak_includes_initial_nav(self):
        # NAV never exceeds 1.0 -> drawdown measured against initial 1.0.
        r = pd.Series([-0.05, -0.05])
        m = portfolio_metrics(r)
        nav_end = (1 - 0.05) ** 2
        self.assertAlmostEqual(m["MDD"], nav_end - 1, places=12)

    def test_undefined_cases_are_nan_not_zero(self):
        m = portfolio_metrics(pd.Series([0.0, 0.0, 0.0]))
        self.assertTrue(np.isnan(m["Sharpe"]))
        self.assertTrue(np.isnan(m["Sortino"]))
        self.assertTrue(np.isnan(m["Calmar"]))
        self.assertEqual(m["MDD"], 0.0)


class TestRankingMetrics(unittest.TestCase):
    def test_ic_not_annualized(self):
        dates = pd.date_range("2023-01-02", periods=20, freq="B")
        instruments = [f"S{i:03d}" for i in range(30)]
        rng = np.random.default_rng(0)
        base = {i: rng.normal() for i in instruments}
        pred = _pred_frame(
            dates, instruments,
            score_fn=lambda d, i: base[i] + rng.normal(scale=0.1),
            label_fn=lambda d, i: base[i] + rng.normal(scale=0.1),
        )
        m = ranking_metrics(pred["score"], pred["label"])
        daily = pred.join(pred, lsuffix="", rsuffix="_r")  # noqa: F841 (structure sanity)
        # recompute daily ICs independently
        ics, rics = [], []
        for _, day in pred.groupby(level="datetime"):
            ics.append(day["score"].corr(day["label"], method="pearson"))
            rics.append(day["score"].corr(day["label"], method="spearman"))
        ics, rics = np.array(ics), np.array(rics)
        self.assertAlmostEqual(m["IC"], ics.mean(), places=12)
        self.assertAlmostEqual(m["ICIR"], ics.mean() / ics.std(ddof=1), places=12)
        self.assertAlmostEqual(m["RankIC"], rics.mean(), places=12)
        self.assertAlmostEqual(m["RankICIR"], rics.mean() / rics.std(ddof=1), places=12)
        # ICIR must NOT be multiplied by sqrt(252)
        self.assertNotAlmostEqual(m["ICIR"], ics.mean() / ics.std(ddof=1) * np.sqrt(252), places=6)


class TestProtocolParams(unittest.TestCase):
    def test_strategy_kwargs_explicit(self):
        self.assertEqual(
            protocol.strategy_kwargs(),
            {
                "topk": 30,
                "n_drop": 5,
                "method_sell": "bottom",
                "method_buy": "top",
                "hold_thresh": 1,
                "only_tradable": False,
                "forbid_all_trade_at_limit": True,
                "risk_degree": 0.95,
            },
        )
        self.assertEqual(
            protocol.STRATEGY_CLASS,
            "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy",
        )

    def test_cost_and_account(self):
        ex = protocol.exchange_kwargs()
        self.assertEqual(ex["open_cost"], 0.0005)
        self.assertEqual(ex["close_cost"], 0.0015)
        self.assertEqual(ex["min_cost"], 0)
        self.assertEqual(ex["deal_price"], "close")
        self.assertEqual(ex["freq"], "day")
        self.assertEqual(protocol.ACCOUNT, 100000000)

    def test_metric_conventions(self):
        self.assertEqual(protocol.TRADING_DAYS, 252)
        self.assertEqual(protocol.DDOF, 1)
        self.assertEqual(protocol.RISK_FREE_RATE, 0.0)
        self.assertEqual(protocol.MAR_DAILY, 0.0)

    def test_timing_contract(self):
        self.assertEqual(protocol.QLIB_SIGNAL_SHIFT, 1)
        self.assertEqual(protocol.ADAPTER_SHIFT, 0)

    def test_formal_test_period(self):
        import yaml

        with open(protocol.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        declared = [str(d) for d in cfg["data"]["test_period"]]
        self.assertEqual(protocol.load_periods()["test"], declared)
        self.assertEqual(declared, ["2023-01-01", "2025-12-31"])

    def test_label_horizon(self):
        self.assertEqual(protocol.load_label_horizon(), 5)


class TestSignalAdapter(unittest.TestCase):
    def test_signal_not_shifted(self):
        dates = pd.date_range("2023-01-03", periods=5, freq="B")
        instruments = ["A", "B", "C"]
        pred = _pred_frame(dates, instruments, score_fn=lambda d, i: 1.0, label_fn=lambda d, i: 0.5)
        signal = make_signal(pred)
        # adapter must pass the prediction through unshifted: identical index
        self.assertTrue(signal.index.equals(pred.index))
        self.assertEqual(signal.index.get_level_values("datetime").min(), dates.min())


class TestEnsemble(unittest.TestCase):
    def test_avg_none_inner_join_mean(self):
        dates1 = pd.date_range("2023-01-02", periods=4, freq="B")
        dates2 = pd.date_range("2023-01-03", periods=4, freq="B")  # shifted overlap
        instruments = ["A", "B"]
        p0 = _pred_frame(dates1, instruments, score_fn=lambda d, i: 1.0, label_fn=lambda d, i: 0.1)
        p1 = _pred_frame(dates2, instruments, score_fn=lambda d, i: 3.0, label_fn=lambda d, i: 0.1)
        ens = ensemble_scores({0: p0, 1: p1}, method="avg_none")
        expected_dates = dates1.intersection(dates2)
        self.assertEqual(
            set(ens.index.get_level_values("datetime").unique()), set(expected_dates)
        )
        self.assertTrue((ens["score"] == 2.0).all())

    def test_label_mismatch_raises(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        instruments = ["A", "B"]
        p0 = _pred_frame(dates, instruments, score_fn=lambda d, i: 1.0, label_fn=lambda d, i: 0.1)
        p1 = _pred_frame(dates, instruments, score_fn=lambda d, i: 1.0, label_fn=lambda d, i: 0.2)
        with self.assertRaises(ValueError):
            ensemble_scores({0: p0, 1: p1})

    def test_unsupported_method_raises(self):
        with self.assertRaises(ValueError):
            ensemble_scores({0: _pred_frame(["2023-01-02"], ["A"], lambda d, i: 1, lambda d, i: 1)},
                            method="avg_zscore")


class TestDiscovery(unittest.TestCase):
    def test_formal_grid_complete(self):
        found = discover(["baseline", "010", "019", "025", "034"], [0, 1, 2, 3, 4])
        self.assertEqual(len(found), 25)
        for (exp, seed), fp in found.items():
            self.assertTrue(fp.path.is_file(), fp.rel_path)
            self.assertFalse(Path(fp.rel_path).is_absolute())
        # 010 seeds 1-4 must come from the follow-up batch, not the failed r2 tasks
        for seed in (1, 2, 3, 4):
            self.assertEqual(found[("010", seed)].source, "phase3:followup-010-4090d")
        self.assertEqual(found[("baseline", 0)].source, "phase2_run")

    def test_missing_experiment_fails(self):
        with self.assertRaises(Exception):
            discover(["does-not-exist"], [0])

    def test_load_real_prediction_schema(self):
        fp = discover(["baseline"], [0])[("baseline", 0)]
        pred = load_prediction(fp.path)
        self.assertEqual(list(pred.columns), ["score", "label"])
        self.assertEqual(list(pred.index.names), ["datetime", "instrument"])
        self.assertFalse(pred.index.duplicated().any())
        signal = make_signal(pred)
        self.assertTrue(signal.index.equals(pred.index))


if __name__ == "__main__":
    unittest.main()
