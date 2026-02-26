"""Tests for VaR backtesting, SIC-sector covariance, and date utilities."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pnlp.baselines.sic_sector import sic_sector_covariance, sic_sector_portfolio
from pnlp.data.dates import generate_quarterly_dates
from pnlp.validation.var_tests import christoffersen_test, kupiec_test, parametric_var


# ── VaR tests ────────────────────────────────────────────────────────────────


class TestParametricVar:
    def test_parametric_var_known_values(self):
        vol = 0.01
        mu = 0.0
        cases = [
            (0.05, -stats.norm.ppf(0.05) * vol),
            (0.01, -stats.norm.ppf(0.01) * vol),
            (0.10, -stats.norm.ppf(0.10) * vol),
        ]
        for alpha, expected in cases:
            result = parametric_var(vol, alpha, mu)
            np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Explicit check for the canonical 5% value
        np.testing.assert_allclose(
            parametric_var(0.01, 0.05, 0.0),
            0.01 * 1.6448536269514729,
            rtol=1e-6,
        )


class TestKupiecTest:
    def test_kupiec_exact_violations(self):
        result = kupiec_test(n_observations=1000, n_violations=50, alpha=0.05)
        assert result["violation_rate"] == pytest.approx(0.05)
        np.testing.assert_allclose(result["lr_statistic"], 0.0, atol=1e-10)
        assert result["p_value"] == pytest.approx(1.0, abs=0.01)
        assert not result["reject_5pct"]

    def test_kupiec_zero_violations(self):
        result = kupiec_test(n_observations=1000, n_violations=0, alpha=0.05)
        assert result["violation_rate"] == 0.0
        assert result["lr_statistic"] > 0
        # Zero violations out of 1000 at alpha=0.05 should strongly reject
        assert result["reject_5pct"]

    def test_kupiec_too_many_violations(self):
        result = kupiec_test(n_observations=1000, n_violations=100, alpha=0.05)
        assert result["violation_rate"] == pytest.approx(0.10)
        assert result["lr_statistic"] > 0
        assert result["reject_5pct"]

    def test_kupiec_no_observations(self):
        result = kupiec_test(n_observations=0, n_violations=0, alpha=0.05)
        assert np.isnan(result["violation_rate"])
        assert np.isnan(result["lr_statistic"])
        assert np.isnan(result["p_value"])
        assert not result["reject_5pct"]


class TestChristoffersenTest:
    def test_christoffersen_independent_violations(self):
        # IID Bernoulli violations are independent by construction.
        # Use a large sample so pi01 ≈ pi11 ≈ 0.10, giving lr_ind ≈ 0.
        rng = np.random.default_rng(12345)
        n = 10_000
        violations = rng.random(n) < 0.10
        result = christoffersen_test(violations, alpha=0.10)
        # Independence LR should be small (chi2(1) critical value at 5% is 3.84)
        assert result["lr_ind"] < 3.84
        assert result["n_violations"] == int(violations.sum())

    def test_christoffersen_clustered_violations(self):
        # 10 consecutive violations then 990 non-violations — strong clustering
        n = 1000
        violations = np.zeros(n, dtype=bool)
        violations[100:110] = True  # 10 consecutive violations
        result = christoffersen_test(violations, alpha=0.01)
        # Clustering should produce a large independence LR
        assert result["lr_ind"] > 5.0
        # Conditional coverage test should reject
        assert result["reject_5pct"]

    def test_christoffersen_too_few_obs(self):
        violations = np.array([True])
        result = christoffersen_test(violations, alpha=0.05)
        assert np.isnan(result["lr_cc"])
        assert np.isnan(result["p_value"])
        assert not result["reject_5pct"]


# ── SIC sector covariance tests ──────────────────────────────────────────────


class TestSicSectorCovariance:
    def test_same_sector_higher_corr(self):
        tickers = ["AAPL", "MSFT", "JPM", "GS"]
        sic_codes = {
            "AAPL": "36",  # Electronic equipment
            "MSFT": "36",  # Same sector
            "JPM": "60",   # Banking
            "GS": "60",    # Same sector
        }
        cov = sic_sector_covariance(
            tickers, sic_codes, within_sector_corr=0.5, cross_sector_corr=0.2,
        )
        # Within sector (AAPL-MSFT, JPM-GS) should be 0.5
        assert cov.loc["AAPL", "MSFT"] == pytest.approx(0.5)
        assert cov.loc["JPM", "GS"] == pytest.approx(0.5)
        # Cross sector (AAPL-JPM, etc.) should be 0.2
        assert cov.loc["AAPL", "JPM"] == pytest.approx(0.2)
        assert cov.loc["MSFT", "GS"] == pytest.approx(0.2)
        # Within > cross
        assert cov.loc["AAPL", "MSFT"] > cov.loc["AAPL", "JPM"]

    def test_unknown_sic_unique_sector(self):
        tickers = ["AAPL", "MSFT", "NEWCO"]
        sic_codes = {"AAPL": "36", "MSFT": "36"}  # NEWCO has no SIC
        cov = sic_sector_covariance(
            tickers, sic_codes, within_sector_corr=0.5, cross_sector_corr=0.2,
        )
        # NEWCO should get cross_sector_corr with everything
        assert cov.loc["NEWCO", "AAPL"] == pytest.approx(0.2)
        assert cov.loc["NEWCO", "MSFT"] == pytest.approx(0.2)
        # Diagonal should be 1
        assert cov.loc["NEWCO", "NEWCO"] == pytest.approx(1.0)

    def test_diagonal_is_one_without_sigma(self):
        tickers = ["A", "B", "C"]
        sic_codes = {"A": "10", "B": "10", "C": "20"}
        cov = sic_sector_covariance(tickers, sic_codes, sigma_estimates=None)
        np.testing.assert_allclose(np.diag(cov.values), 1.0)

    def test_sigma_scales_covariance(self):
        tickers = ["A", "B", "C"]
        sic_codes = {"A": "10", "B": "10", "C": "20"}
        sigmas = pd.Series({"A": 0.02, "B": 0.03, "C": 0.01})
        corr = sic_sector_covariance(
            tickers, sic_codes, sigma_estimates=None,
            within_sector_corr=0.5, cross_sector_corr=0.2,
        )
        cov = sic_sector_covariance(
            tickers, sic_codes, sigma_estimates=sigmas,
            within_sector_corr=0.5, cross_sector_corr=0.2,
        )
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                expected = corr.loc[ti, tj] * sigmas[ti] * sigmas[tj]
                assert cov.loc[ti, tj] == pytest.approx(expected, rel=1e-10)

    def test_sic_sector_portfolio_runs(self):
        tickers = [f"T{i:03d}" for i in range(30)]
        sic_codes = {t: str(10 + i % 5) for i, t in enumerate(tickers)}
        sigmas = pd.Series({t: 0.02 for t in tickers})
        result = sic_sector_portfolio(
            tickers, sic_codes, sigma_estimates=sigmas, max_weight=0.10,
        )
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)
        assert (result.weights >= -1e-6).all()
        assert len(result.weights) == 30


# ── Date utility tests ───────────────────────────────────────────────────────


class TestGenerateQuarterlyDates:
    def test_quarterly_dates_basic(self):
        dates = generate_quarterly_dates(date(2020, 1, 1), date(2020, 12, 31))
        expected = [date(2020, 1, 1), date(2020, 4, 1), date(2020, 7, 1), date(2020, 10, 1)]
        assert dates == expected

    def test_quarterly_dates_mid_quarter_start(self):
        dates = generate_quarterly_dates(date(2020, 2, 15), date(2020, 12, 31))
        # Feb 15 is mid-Q1; quarter start Jan 1 < start date, so first included is Apr 1
        assert date(2020, 1, 1) not in dates
        assert dates[0] == date(2020, 4, 1)
        assert len(dates) == 3  # Apr 1, Jul 1, Oct 1

    def test_quarterly_dates_75_quarters(self):
        dates = generate_quarterly_dates(date(2007, 4, 1), date(2025, 12, 31))
        assert len(dates) == 75
        assert dates[0] == date(2007, 4, 1)
        assert dates[-1] == date(2025, 10, 1)
