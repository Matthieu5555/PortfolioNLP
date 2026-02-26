"""Tests for point-in-time investable universe filter."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pnlp.data.universe_filter import LiquidityConfig, filter_investable_universe
from tests.conftest import (
    SYNTHETIC_FIRMS,
    make_dollar_volume,
    make_firm_embeddings,
    make_returns,
)

TICKERS = [f.ticker for f in SYNTHETIC_FIRMS]


@pytest.fixture
def base_returns() -> pd.DataFrame:
    return make_returns(TICKERS, n_days=600, start_date=date(2021, 1, 1))


@pytest.fixture
def base_dv() -> pd.DataFrame:
    return make_dollar_volume(TICKERS, n_days=600, start_date=date(2021, 1, 1))


@pytest.fixture
def base_embeddings() -> dict:
    return make_firm_embeddings(SYNTHETIC_FIRMS, as_of_date=date(2023, 6, 1))


def test_basic_filter_returns_sorted_tickers(base_returns, base_dv, base_embeddings):
    """Filter should return a sorted list of eligible tickers."""
    config = LiquidityConfig(adv_threshold=0.0)  # no ADV filter
    eligible = filter_investable_universe(
        base_embeddings, base_returns, base_dv,
        rebal_date=date(2023, 6, 1), config=config,
    )
    assert eligible == sorted(eligible)
    assert len(eligible) > 0


def test_strict_lag_excludes_rebal_date(base_embeddings):
    """Data from rebal_date itself must NOT be used (strict < T)."""
    tickers = TICKERS[:3]
    rebal_date = date(2023, 1, 5)
    # Create returns with just a few days
    dates = pd.bdate_range(start="2022-01-01", end="2023-01-05")
    rng = np.random.RandomState(0)
    returns = pd.DataFrame(
        rng.randn(len(dates), len(tickers)) * 0.01,
        index=dates, columns=tickers,
    )
    dv = pd.DataFrame(
        np.full((len(dates), len(tickers)), 1e9),
        index=dates, columns=tickers,
    )

    config = LiquidityConfig(
        adv_threshold=0.0,
        covariance_lookback_days=252,
        min_return_completeness=0.0,
    )

    # Set returns on rebal_date to NaN — if filter uses this date, completeness drops
    returns.loc[str(rebal_date)] = np.nan

    embeddings = {t: base_embeddings[t] for t in tickers if t in base_embeddings}
    eligible = filter_investable_universe(
        embeddings, returns, dv, rebal_date, config=config,
    )
    # Should still pass because rebal_date data isn't used
    assert len(eligible) == len(tickers)


def test_adv_threshold_filters_illiquid(base_returns, base_embeddings):
    """Tickers below ADV threshold are excluded."""
    tickers = TICKERS[:5]
    n_days = len(base_returns)
    dv = pd.DataFrame(
        np.full((n_days, 5), 500_000.0),  # $500K — below $1M threshold
        index=base_returns.index,
        columns=tickers,
    )
    # Make first 2 tickers liquid
    dv.iloc[:, :2] = 5_000_000.0  # $5M

    config = LiquidityConfig(adv_threshold=1_000_000.0)
    embeddings = {t: base_embeddings[t] for t in tickers if t in base_embeddings}
    eligible = filter_investable_universe(
        embeddings, base_returns[tickers], dv,
        rebal_date=date(2023, 6, 1), config=config,
    )
    # Only the 2 liquid tickers should pass
    assert set(eligible) == set(tickers[:2])


def test_return_completeness_filter(base_dv, base_embeddings):
    """Tickers with too many NaN returns are excluded."""
    tickers = TICKERS[:3]
    dates = pd.bdate_range(start="2021-01-01", periods=600)
    rng = np.random.RandomState(0)
    returns = pd.DataFrame(
        rng.randn(600, 3) * 0.01,
        index=dates, columns=tickers,
    )
    # Make third ticker mostly NaN in lookback window
    returns.iloc[-504:, 2] = np.nan

    dv = base_dv[tickers].iloc[:600]
    # Ensure ADV is high enough
    dv.iloc[:] = 1e9

    config = LiquidityConfig(
        adv_threshold=0.0,
        min_return_completeness=0.8,
        covariance_lookback_days=504,
    )
    embeddings = {t: base_embeddings[t] for t in tickers if t in base_embeddings}
    eligible = filter_investable_universe(
        embeddings, returns, dv,
        rebal_date=date(2023, 5, 1), config=config,
    )
    assert tickers[2] not in eligible
    assert tickers[0] in eligible
    assert tickers[1] in eligible


def test_max_firms_caps_by_adv_rank(base_returns, base_embeddings):
    """When max_firms is set, top tickers by ADV are kept."""
    tickers = TICKERS[:5]
    n_days = len(base_returns)
    dv = pd.DataFrame(
        np.full((n_days, 5), 10_000_000.0),
        index=base_returns.index,
        columns=tickers,
    )
    # Set descending ADV: ticker 0 highest, ticker 4 lowest
    for i, t in enumerate(tickers):
        dv[t] = (5 - i) * 100_000_000.0

    config = LiquidityConfig(adv_threshold=0.0, max_firms=3)
    embeddings = {t: base_embeddings[t] for t in tickers if t in base_embeddings}
    eligible = filter_investable_universe(
        embeddings, base_returns[tickers], dv,
        rebal_date=date(2023, 6, 1), config=config,
    )
    assert len(eligible) == 3
    # Should include the 3 highest ADV tickers
    assert set(eligible) == set(tickers[:3])


def test_no_data_before_rebal_returns_empty(base_dv, base_embeddings):
    """If no data exists before rebal_date, return empty list."""
    tickers = TICKERS[:3]
    dates = pd.bdate_range(start="2025-01-01", periods=10)
    returns = pd.DataFrame(
        np.zeros((10, 3)), index=dates, columns=tickers,
    )
    dv = pd.DataFrame(
        np.full((10, 3), 1e9), index=dates, columns=tickers,
    )

    embeddings = {t: base_embeddings[t] for t in tickers if t in base_embeddings}
    # Rebal date before all data
    eligible = filter_investable_universe(
        embeddings, returns, dv,
        rebal_date=date(2024, 12, 31),
    )
    assert eligible == []


def test_only_tickers_with_embeddings_included(base_returns, base_dv):
    """Tickers without embeddings are excluded even if liquid."""
    embeddings = make_firm_embeddings(SYNTHETIC_FIRMS[:3])
    config = LiquidityConfig(adv_threshold=0.0)
    eligible = filter_investable_universe(
        embeddings, base_returns, base_dv,
        rebal_date=date(2023, 6, 1), config=config,
    )
    # Should only include the 3 tickers with embeddings
    for t in eligible:
        assert t in embeddings


def test_uses_median_not_mean_for_adv(base_returns, base_embeddings):
    """Median ADV is robust to spike days; verify a single spike doesn't qualify a ticker."""
    tickers = TICKERS[:2]
    n_days = len(base_returns)
    dv = pd.DataFrame(
        np.full((n_days, 2), 100_000.0),  # $100K — well below $1M
        index=base_returns.index,
        columns=tickers,
    )
    # Add a single spike day for ticker 0 (mean would push it above $1M)
    dv.iloc[-5, 0] = 1_000_000_000.0  # $1B spike

    config = LiquidityConfig(adv_threshold=1_000_000.0, adv_lookback_days=21)
    embeddings = {t: base_embeddings[t] for t in tickers if t in base_embeddings}
    eligible = filter_investable_universe(
        embeddings, base_returns[tickers], dv,
        rebal_date=date(2023, 6, 1), config=config,
    )
    # Median is still $100K, so ticker should be excluded despite the spike
    assert len(eligible) == 0
