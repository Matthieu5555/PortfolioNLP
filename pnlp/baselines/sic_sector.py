"""
SIC-sector covariance baseline.

Constructs a block-diagonal correlation matrix where firms in the
same 2-digit SIC sector have a constant within-sector correlation
and cross-sector pairs have a lower constant correlation. Scaled
by sample volatilities when available.

This is the naive practitioner approach: assign sector membership
and use sector-level correlation assumptions. The text-based method
should outperform this if embeddings capture finer-grained structure
than SIC codes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pnlp.config import DATA_DIR
from pnlp.portfolio.optimizer import PortfolioOptimizer, PortfolioWeights

logger = logging.getLogger(__name__)

__all__ = ["load_sic_codes", "sic_sector_covariance", "sic_sector_portfolio"]

SIC_CACHE_PATH = DATA_DIR / "sic_codes.json"


def load_sic_codes() -> dict[str, str]:
    """Load 2-digit SIC codes for all tickers.

    Returns:
        dict mapping ticker -> 2-digit SIC code string (e.g., "36").
        Tickers without SIC codes are omitted.
    """
    if not SIC_CACHE_PATH.exists():
        logger.warning("SIC codes not found at %s", SIC_CACHE_PATH)
        return {}

    with open(SIC_CACHE_PATH) as f:
        raw = json.load(f)

    sic_2digit = {}
    for ticker, entry in raw.items():
        sic = entry.get("sic")
        if sic and len(sic) >= 2:
            sic_2digit[ticker] = sic[:2]

    return sic_2digit


def sic_sector_covariance(
    tickers: list[str],
    sic_codes: dict[str, str],
    sigma_estimates: pd.Series | None = None,
    within_sector_corr: float = 0.5,
    cross_sector_corr: float = 0.2,
) -> pd.DataFrame:
    """Build a SIC-sector-based covariance matrix.

    Args:
        tickers: sorted list of ticker strings.
        sic_codes: ticker -> 2-digit SIC code mapping.
        sigma_estimates: per-ticker volatility estimates. If None,
            returns a correlation matrix (unit diagonal).
        within_sector_corr: constant pairwise correlation for firms
            in the same 2-digit SIC sector.
        cross_sector_corr: constant pairwise correlation for firms
            in different sectors.

    Returns:
        DataFrame (n_tickers x n_tickers) covariance matrix.
    """
    n = len(tickers)

    # Assign sectors — unknown SIC gets a unique sector per ticker
    sectors = []
    for t in tickers:
        sic = sic_codes.get(t)
        if sic:
            sectors.append(sic)
        else:
            sectors.append(f"_unknown_{t}")

    # Build correlation matrix
    corr = np.full((n, n), cross_sector_corr)
    for i in range(n):
        corr[i, i] = 1.0
        for j in range(i + 1, n):
            if sectors[i] == sectors[j]:
                corr[i, j] = within_sector_corr
                corr[j, i] = within_sector_corr

    # Scale by volatilities if provided
    if sigma_estimates is not None:
        sigmas = np.array([sigma_estimates.get(t, 1.0) for t in tickers])
        cov = corr * np.outer(sigmas, sigmas)
    else:
        cov = corr

    return pd.DataFrame(cov, index=tickers, columns=tickers)


def sic_sector_portfolio(
    tickers: list[str],
    sic_codes: dict[str, str],
    sigma_estimates: pd.Series | None = None,
    objective: str = "min_variance",
    long_only: bool = True,
    max_weight: float = 0.05,
    within_sector_corr: float = 0.5,
    cross_sector_corr: float = 0.2,
) -> PortfolioWeights:
    """Construct a portfolio using SIC-sector covariance.

    Args:
        tickers: sorted list of eligible tickers.
        sic_codes: ticker -> 2-digit SIC code.
        sigma_estimates: per-ticker volatility estimates.
        objective: optimization objective.
        long_only: long-only constraint.
        max_weight: maximum position size.
        within_sector_corr: within-sector correlation assumption.
        cross_sector_corr: cross-sector correlation assumption.

    Returns:
        PortfolioWeights with optimal allocations.
    """
    from pnlp.config import PortfolioConfig

    cov = sic_sector_covariance(
        tickers,
        sic_codes,
        sigma_estimates=sigma_estimates,
        within_sector_corr=within_sector_corr,
        cross_sector_corr=cross_sector_corr,
    )

    config = PortfolioConfig(
        objective=objective,
        long_only=long_only,
        max_weight=max_weight,
    )
    optimizer = PortfolioOptimizer(config)
    result = optimizer.optimize(cov)

    result.metadata["method"] = "sic_sector"
    return result
