"""
Point-in-time investable universe filter.

Applies liquidity screening using strictly lagged data to prevent
look-ahead bias. At each rebalance date T, only data from before T
is used (never T itself).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from pnlp.embeddings.firm_aggregator import FirmEmbedding

logger = logging.getLogger(__name__)

__all__ = ["LiquidityConfig", "filter_investable_universe"]


@dataclass(frozen=True)
class LiquidityConfig:
    """Configuration for point-in-time liquidity filtering.

    adv_threshold: minimum trailing median daily dollar volume in USD.
        $1M is a standard institutional threshold for "investable".
        Implicitly screens out penny stocks (a stock at $1 needs 1M shares/day).
    adv_lookback_days: trailing business days for median ADV calculation.
        21 = one calendar month of trading.
    min_return_completeness: fraction of non-NaN returns required in the
        covariance lookback window. 0.8 = must have 80% of days.
    covariance_lookback_days: window for return completeness check.
    max_firms: optional cap on universe size, ranked by ADV.
    """

    adv_threshold: float = 1_000_000.0
    adv_lookback_days: int = 21
    min_return_completeness: float = 0.8
    covariance_lookback_days: int = 504
    max_firms: int | None = None


def filter_investable_universe(
    firm_embeddings: dict[str, FirmEmbedding],
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    rebal_date: date,
    config: LiquidityConfig | None = None,
) -> list[str]:
    """Filter tickers to those investable as of rebal_date.

    Uses strictly lagged data: at date T, only T-lookback through T-1.
    Never touches data from date T itself.

    Selection criteria (all must pass):
        1. Ticker has firm embeddings (semantic data available)
        2. Trailing median daily dollar volume >= threshold (liquidity)
        3. Sufficient return history for covariance estimation (completeness)

    Args:
        firm_embeddings: ticker -> FirmEmbedding as of rebal_date.
        daily_returns: (dates x tickers) log return matrix.
        dollar_volume: (dates x tickers) daily dollar volume matrix.
        rebal_date: the rebalancing date T. All data used is strictly < T.
        config: liquidity filter parameters.

    Returns:
        Sorted list of eligible ticker strings.
    """
    if config is None:
        config = LiquidityConfig()

    rebal_str = str(rebal_date)

    # Strict lag: exclude rebal_date itself
    returns_before_T = daily_returns.loc[daily_returns.index < rebal_str]
    dv_before_T = dollar_volume.loc[dollar_volume.index < rebal_str]

    if len(returns_before_T) == 0 or len(dv_before_T) == 0:
        logger.warning("No data before %s", rebal_date)
        return []

    # ADV filter: trailing median dollar volume (median not mean — robust to spike days)
    adv_window = dv_before_T.iloc[-config.adv_lookback_days:]
    median_adv = adv_window.median()
    liquid_tickers = set(median_adv[median_adv >= config.adv_threshold].index)

    # Return completeness filter
    lookback = config.covariance_lookback_days
    if len(returns_before_T) > lookback:
        lookback_returns = returns_before_T.iloc[-lookback:]
    else:
        lookback_returns = returns_before_T

    n_required = int(len(lookback_returns) * config.min_return_completeness)
    non_nan_counts = lookback_returns.notna().sum()
    complete_tickers = set(non_nan_counts[non_nan_counts >= n_required].index)

    # Intersection of all criteria
    embedding_tickers = set(firm_embeddings.keys())
    eligible = embedding_tickers & liquid_tickers & complete_tickers

    # Optional cap by ADV rank (most liquid first)
    if config.max_firms and len(eligible) > config.max_firms:
        ranked = median_adv.loc[list(eligible)].sort_values(ascending=False)
        eligible = set(ranked.index[: config.max_firms])

    eligible_sorted = sorted(eligible)

    n_emb_liquid = len(liquid_tickers & embedding_tickers)
    n_emb_complete = len(complete_tickers & embedding_tickers)
    logger.info(
        "Universe at %s: %d with embeddings, %d liquid (ADV>=$%.0fM), "
        "%d complete returns → %d eligible%s",
        rebal_date,
        len(embedding_tickers),
        n_emb_liquid,
        config.adv_threshold / 1e6,
        n_emb_complete,
        len(eligible_sorted),
        f" (capped to {config.max_firms})" if config.max_firms and len(eligible_sorted) == config.max_firms else "",
    )

    return eligible_sorted
