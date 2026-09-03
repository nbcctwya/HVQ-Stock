import math
import unittest

import numpy as np
import pandas as pd

from baseline_results.metrics import portfolio_metrics


class ProtocolMetricTests(unittest.TestCase):
    def test_first_day_loss_is_in_drawdown(self):
        self.assertAlmostEqual(portfolio_metrics(pd.Series([-0.10]))["MDD"], -0.10)

    def test_identical_negative_days_have_defined_sortino(self):
        result = portfolio_metrics(pd.Series([-0.01, -0.01]))
        expected = math.sqrt(252) * np.log(0.99) / abs(np.log(0.99))
        self.assertAlmostEqual(result["Sortino"], expected)

    def test_return_at_or_below_minus_one_raises(self):
        for bad in (-1.0, -1.2):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                portfolio_metrics(pd.Series([bad]))

    def test_independent_manual_recalculation(self):
        r = np.array([0.01, -0.02, 0.005, 0.003, -0.004])
        g = np.log(1 + r)
        nav = np.r_[1.0, np.exp(np.cumsum(g))]
        expected = {
            "AR": np.exp(g.mean() * 252) - 1,
            "STD": g.std(ddof=1) * np.sqrt(252),
            "MDD": np.min(nav / np.maximum.accumulate(nav) - 1),
            "Sharpe": np.sqrt(252) * g.mean() / g.std(ddof=1),
            "Sortino": np.sqrt(252) * g.mean() / np.sqrt(np.mean(np.minimum(g, 0) ** 2)),
        }
        expected["Calmar"] = expected["AR"] / abs(expected["MDD"])
        actual = portfolio_metrics(pd.Series(r))
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value)


if __name__ == "__main__":
    unittest.main()
