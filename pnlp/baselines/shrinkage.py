"""
Ledoit-Wolf shrinkage baseline.

Traditional covariance estimation using Ledoit-Wolf shrinkage
to a constant-correlation target. Standard benchmark in portfolio
optimization literature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pnlp.portfolio.optimizer import PortfolioOptimizer, PortfolioWeights

logger = logging.getLogger(__name__)

__all__ = ["ledoit_wolf_portfolio"]


def ledoit_wolf_portfolio(
    returns: pd.DataFrame,
    objective: str = "min_variance",
    long_only: bool = True,
    max_weight: float = 0.05,
    *,
    ticker_sectors: dict[str, str] | None = None,
    sector_limits: dict[str, float] | None = None,
    vol_target: float | None = None,
) -> PortfolioWeights:
    """Construct a portfolio using Ledoit-Wolf shrinkage covariance.

    Args:
        returns: DataFrame with tickers as columns, dates as index.
        objective: optimization objective.
        long_only: whether to constrain weights >= 0.
        max_weight: maximum position size.
        ticker_sectors: ticker -> sector mapping (for sector constraints).
        sector_limits: sector -> max weight (for sector constraints).
        vol_target: target annualized vol (weights scaled post-optimization).

    Returns:
        PortfolioWeights with optimal allocations.
    """
    from pnlp.config import PortfolioConfig

    # Use all columns provided (caller enforces universe alignment and completeness)
    clean_returns = returns.fillna(0.0)

    if len(clean_returns) < 10 or len(clean_returns.columns) < 2:
        logger.warning("Insufficient data for Ledoit-Wolf")
        return PortfolioWeights(
            weights=pd.Series(dtype=float),
            objective_value=float("inf"),
            metadata={"method": "ledoit_wolf", "error": "insufficient_data"},
        )

    from pnlp.primitives.gpu_accel import ledoit_wolf_covariance

    cov_arr, shrinkage = ledoit_wolf_covariance(clean_returns.values)
    cov = pd.DataFrame(
        cov_arr,
        index=clean_returns.columns,
        columns=clean_returns.columns,
    )

    config = PortfolioConfig(
        objective=objective,
        long_only=long_only,
        max_weight=max_weight,
    )
    optimizer = PortfolioOptimizer(config)
    result = optimizer.optimize(
        cov,
        ticker_sectors=ticker_sectors,
        sector_limits=sector_limits,
        vol_target=vol_target,
    )

    result.metadata["method"] = "ledoit_wolf"
    result.metadata["shrinkage"] = float(shrinkage)
    return result
