"""Tests for portfolio optimization and baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pnlp.baselines.equal_weight import equal_weight_portfolio
from pnlp.baselines.shrinkage import ledoit_wolf_portfolio
from pnlp.config import PortfolioConfig
from pnlp.portfolio.optimizer import PortfolioOptimizer


class TestPortfolioOptimizer:
    def test_min_variance(self):
        tickers = ["A", "B", "C"]
        cov = pd.DataFrame(
            np.eye(3) * 0.04,
            index=tickers,
            columns=tickers,
        )
        config = PortfolioConfig(objective="min_variance", long_only=True, max_weight=1.0)
        opt = PortfolioOptimizer(config)
        result = opt.optimize(cov)

        # Equal variance -> should be ~equal weight
        assert abs(result.weights.sum() - 1.0) < 1e-6
        for w in result.weights:
            assert abs(w - 1 / 3) < 0.01

    def test_long_only_constraint(self):
        tickers = ["A", "B"]
        cov = pd.DataFrame(
            [[0.04, 0.03], [0.03, 0.04]],
            index=tickers,
            columns=tickers,
        )
        config = PortfolioConfig(objective="min_variance", long_only=True)
        opt = PortfolioOptimizer(config)
        result = opt.optimize(cov)
        assert (result.weights >= -1e-6).all()

    def test_max_weight_constraint(self):
        tickers = ["A", "B", "C", "D", "E"]
        cov = pd.DataFrame(
            np.eye(5) * 0.04,
            index=tickers,
            columns=tickers,
        )
        config = PortfolioConfig(objective="min_variance", max_weight=0.25)
        opt = PortfolioOptimizer(config)
        result = opt.optimize(cov)
        assert result.weights.max() <= 0.25 + 1e-6

    def test_weights_sum_to_one(self):
        tickers = ["A", "B", "C"]
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.005], [0.01, 0.09, 0.02], [0.005, 0.02, 0.06]],
            index=tickers,
            columns=tickers,
        )
        config = PortfolioConfig(objective="min_variance")
        result = PortfolioOptimizer(config).optimize(cov)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_mean_variance(self):
        tickers = ["A", "B"]
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.04]],
            index=tickers,
            columns=tickers,
        )
        mu = pd.Series([0.10, 0.05], index=tickers)
        config = PortfolioConfig(objective="mean_variance", long_only=True, max_weight=1.0)
        result = PortfolioOptimizer(config).optimize(cov, mu)
        # Higher expected return -> should tilt toward A
        assert result.weights["A"] > result.weights["B"]


class TestEqualWeight:
    def test_basic(self):
        tickers = ["A", "B", "C"]
        result = equal_weight_portfolio(tickers)
        assert abs(result.weights.sum() - 1.0) < 1e-10
        for w in result.weights:
            assert abs(w - 1 / 3) < 1e-10


class TestLedoitWolf:
    def test_basic(self, daily_returns):
        result = ledoit_wolf_portfolio(daily_returns, max_weight=1.0)
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert result.metadata["method"] == "ledoit_wolf"
