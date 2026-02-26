"""Tests for validation module (metrics, statistical tests, backtest)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pnlp.validation.metrics import compute_portfolio_metrics
from pnlp.validation.statistical_tests import (
    block_bootstrap_ci,
    block_permutation_test,
)


class TestMetrics:
    def test_basic_metrics(self):
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.randn(252) * 0.01 + 0.0003)
        metrics = compute_portfolio_metrics(returns)

        assert "annualized_return" in metrics
        assert "annualized_volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert metrics["max_drawdown"] <= 0

    def test_with_benchmark(self):
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2024-01-01", periods=252)
        returns = pd.Series(rng.randn(252) * 0.01, index=dates)
        benchmark = pd.Series(rng.randn(252) * 0.01, index=dates)

        metrics = compute_portfolio_metrics(returns, benchmark_returns=benchmark)
        assert "information_ratio" in metrics
        assert "capm_alpha" in metrics
        assert "capm_beta" in metrics

    def test_empty_returns(self):
        metrics = compute_portfolio_metrics(pd.Series(dtype=float))
        assert metrics == {}


class TestPermutationTest:
    def test_identical_samples(self):
        a = np.ones(500)
        b = np.ones(500)
        result = block_permutation_test(a, b, block_size=63)
        assert result.p_value > 0.5  # Should not reject H0

    def test_different_samples(self):
        a = np.ones(500) * 10
        b = np.zeros(500)
        result = block_permutation_test(a, b, block_size=63)
        assert result.p_value < 0.01  # Should reject H0

    def test_statistic_is_difference(self):
        a = np.array([1.0, 2.0, 3.0] * 100)
        b = np.array([4.0, 5.0, 6.0] * 100)
        result = block_permutation_test(a, b, block_size=10)
        assert abs(result.statistic - (-3.0)) < 1e-10


class TestBlockBootstrap:
    def test_ci_contains_mean(self):
        values = np.random.randn(200)
        result = block_bootstrap_ci(values, block_size=10)
        assert result.ci_low <= result.statistic <= result.ci_high

    def test_wider_ci_for_smaller_sample(self):
        rng = np.random.RandomState(42)
        small = rng.randn(50)
        large = rng.randn(500)

        small_result = block_bootstrap_ci(small, block_size=5)
        large_result = block_bootstrap_ci(large, block_size=5)

        small_width = small_result.ci_high - small_result.ci_low
        large_width = large_result.ci_high - large_result.ci_low
        assert small_width > large_width
