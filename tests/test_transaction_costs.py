"""Tests for stratified transaction cost model."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pnlp.validation.transaction_costs import (
    DEFAULT_TIERS,
    TCTier,
    TransactionCostModel,
)


@pytest.fixture
def simple_returns() -> pd.DataFrame:
    """Daily returns matrix with 252 days for vol regime testing."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range(start="2022-01-01", periods=500)
    tickers = ["AAPL", "MSFT", "GOOGL", "PENNY"]
    returns = pd.DataFrame(
        rng.randn(500, 4) * 0.015,
        index=dates, columns=tickers,
    )
    return returns


class TestTierAssignment:
    def test_large_cap_tier(self):
        model = TransactionCostModel()
        tier = model._assign_tier(600_000_000.0)
        assert tier.name == "large_cap"
        assert tier.cost_bps == 10.0

    def test_mid_cap_tier(self):
        model = TransactionCostModel()
        tier = model._assign_tier(100_000_000.0)
        assert tier.name == "mid_cap"
        assert tier.cost_bps == 20.0

    def test_small_cap_tier(self):
        model = TransactionCostModel()
        tier = model._assign_tier(10_000_000.0)
        assert tier.name == "small_cap"
        assert tier.cost_bps == 50.0

    def test_boundary_large_cap(self):
        model = TransactionCostModel()
        tier = model._assign_tier(500_000_000.0)
        assert tier.name == "large_cap"

    def test_boundary_mid_cap(self):
        model = TransactionCostModel()
        tier = model._assign_tier(50_000_000.0)
        assert tier.name == "mid_cap"

    def test_zero_adv_is_small_cap(self):
        model = TransactionCostModel()
        tier = model._assign_tier(0.0)
        assert tier.name == "small_cap"

    def test_nan_adv_gets_worst_tier(self, simple_returns):
        """NaN ADV should get the most conservative (highest cost) tier."""
        model = TransactionCostModel()
        adv_series = pd.Series({"AAPL": 1e9, "JUNK": np.nan})
        rates = model.get_tc_rates(adv_series, simple_returns, "2023-06-01")
        # NaN ticker gets small_cap (worst tier)
        assert rates["JUNK"] >= rates["AAPL"]


class TestGetTCRates:
    def test_returns_series_for_all_tickers(self, simple_returns):
        model = TransactionCostModel()
        adv = pd.Series({
            "AAPL": 1e9,
            "MSFT": 8e8,
            "GOOGL": 1e8,
            "PENNY": 5e6,
        })
        rates = model.get_tc_rates(adv, simple_returns, "2023-06-01")
        assert isinstance(rates, pd.Series)
        assert set(rates.index) == {"AAPL", "MSFT", "GOOGL", "PENNY"}

    def test_rates_monotonic_with_adv(self, simple_returns):
        """Higher ADV should get lower TC rates."""
        model = TransactionCostModel()
        adv = pd.Series({
            "LARGE": 1e9,
            "MID": 1e8,
            "SMALL": 1e7,
        })
        rates = model.get_tc_rates(adv, simple_returns, "2023-06-01")
        assert rates["LARGE"] <= rates["MID"] <= rates["SMALL"]

    def test_rates_are_decimal_fractions(self, simple_returns):
        """Rates should be in decimal (e.g., 0.001 = 10bps), not percent."""
        model = TransactionCostModel()
        adv = pd.Series({"AAPL": 1e9})
        rates = model.get_tc_rates(adv, simple_returns, "2023-06-01")
        # 10 bps = 0.001
        assert rates["AAPL"] < 0.01  # Less than 1%
        assert rates["AAPL"] > 0.0   # Positive

    def test_large_cap_base_rate(self, simple_returns):
        """Large-cap base rate should be 10bps in calm markets."""
        model = TransactionCostModel()
        adv = pd.Series({"AAPL": 1e9})
        rates = model.get_tc_rates(adv, simple_returns, "2023-06-01")
        # In calm market, vol_mult ~ 1.0, so rate ~ 10bps = 0.001
        assert 0.0005 <= rates["AAPL"] <= 0.003  # reasonable range

    def test_small_cap_base_rate(self, simple_returns):
        """Small-cap base rate should be 50bps in calm markets."""
        model = TransactionCostModel()
        adv = pd.Series({"PENNY": 1e6})
        rates = model.get_tc_rates(adv, simple_returns, "2023-06-01")
        # 50bps = 0.005
        assert rates["PENNY"] >= 0.003  # at least mid-range


class TestVolRegimeMultiplier:
    def test_calm_market_multiplier_near_one(self, simple_returns):
        """Stable vol should give multiplier close to 1.0."""
        model = TransactionCostModel()
        mult = model._vol_regime_multiplier(simple_returns, "2023-06-01")
        assert 0.9 <= mult <= 1.5

    def test_high_vol_increases_multiplier(self):
        """Spiking vol should increase the multiplier."""
        rng = np.random.RandomState(42)
        dates = pd.bdate_range(start="2020-01-01", periods=500)
        tickers = ["A", "B", "C"]
        # Normal vol for first 400 days
        rets = pd.DataFrame(
            rng.randn(500, 3) * 0.01,
            index=dates, columns=tickers,
        )
        # Spike vol for last 63 days
        rets.iloc[-63:] *= 5.0

        model = TransactionCostModel(vol_lookback=63)
        mult = model._vol_regime_multiplier(rets, str(dates[-1] + pd.Timedelta(days=1)))
        assert mult > 1.0

    def test_multiplier_capped(self):
        """Multiplier should not exceed vol_multiplier_cap."""
        rng = np.random.RandomState(42)
        dates = pd.bdate_range(start="2020-01-01", periods=500)
        tickers = ["A"]
        rets = pd.DataFrame(rng.randn(500, 1) * 0.001, index=dates, columns=tickers)
        # Extreme recent spike
        rets.iloc[-63:] *= 100.0

        model = TransactionCostModel(vol_lookback=63, vol_multiplier_cap=2.0)
        mult = model._vol_regime_multiplier(rets, str(dates[-1] + pd.Timedelta(days=1)))
        assert mult <= 2.0

    def test_insufficient_history_returns_one(self):
        """With insufficient history, multiplier defaults to 1.0."""
        dates = pd.bdate_range(start="2023-01-01", periods=10)
        rets = pd.DataFrame(
            np.random.randn(10, 2) * 0.01,
            index=dates, columns=["A", "B"],
        )
        model = TransactionCostModel(vol_lookback=63)
        mult = model._vol_regime_multiplier(rets, "2023-02-01")
        assert mult == 1.0


class TestCustomTiers:
    def test_custom_tiers_override_defaults(self, simple_returns):
        """Custom tiers should replace defaults."""
        custom = [
            TCTier("mega", adv_floor=1e10, cost_bps=5.0),
            TCTier("other", adv_floor=0, cost_bps=100.0),
        ]
        model = TransactionCostModel(tiers=custom)
        adv = pd.Series({"MEGA": 2e10, "OTHER": 1e6})
        rates = model.get_tc_rates(adv, simple_returns, "2023-06-01")
        # MEGA should get the low rate tier
        assert rates["MEGA"] < rates["OTHER"]
