"""
Equal-weight (1/N) baseline.

DeMiguel et al. (2009) showed this is surprisingly hard to beat
out of sample. Essential benchmark for any portfolio strategy.
"""

from __future__ import annotations

import pandas as pd

from pnlp.portfolio.optimizer import PortfolioWeights

__all__ = ["equal_weight_portfolio", "constrained_equal_weight_portfolio"]


def equal_weight_portfolio(tickers: list[str]) -> PortfolioWeights:
    """Construct an equal-weight portfolio."""
    n = len(tickers)
    weights = pd.Series(1.0 / n, index=tickers)
    return PortfolioWeights(
        weights=weights,
        objective_value=0.0,
        metadata={"method": "equal_weight", "n_assets": n},
    )


def constrained_equal_weight_portfolio(
    tickers: list[str],
    ticker_sectors: dict[str, str],
    sector_limits: dict[str, float],
) -> PortfolioWeights:
    """Equal-weight portfolio respecting sector limits.

    Iteratively caps sectors that exceed their limit and redistributes
    excess weight equally among uncapped sectors' tickers.
    """
    import numpy as np

    n = len(tickers)
    weights = pd.Series(1.0 / n, index=tickers)

    # Map tickers to sectors
    sector_groups: dict[str, list[str]] = {}
    for t in tickers:
        sec = ticker_sectors.get(t, "_unknown")
        sector_groups.setdefault(sec, []).append(t)

    # Iterative capping: cap sectors over limit, redistribute excess
    for _ in range(10):  # converges in 2-3 iterations
        excess = 0.0
        uncapped_tickers: list[str] = []
        for sec, members in sector_groups.items():
            sector_wt = weights[members].sum()
            limit = sector_limits.get(sec, 1.0)
            if sector_wt > limit + 1e-10:
                # Scale down proportionally within sector
                scale = limit / sector_wt
                weights[members] *= scale
                excess += sector_wt - limit
            else:
                uncapped_tickers.extend(members)

        if excess < 1e-10 or not uncapped_tickers:
            break

        # Redistribute excess equally among uncapped tickers
        per_ticker = excess / len(uncapped_tickers)
        weights[uncapped_tickers] += per_ticker

    # Final normalization
    weights = weights / weights.sum()

    return PortfolioWeights(
        weights=weights,
        objective_value=0.0,
        metadata={
            "method": "constrained_equal_weight",
            "n_assets": n,
            "n_sectors_capped": sum(
                1 for sec, members in sector_groups.items()
                if weights[members].sum() >= sector_limits.get(sec, 1.0) - 1e-6
            ),
        },
    )
