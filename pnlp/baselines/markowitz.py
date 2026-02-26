"""
Sample covariance Markowitz baseline.

Traditional mean-variance optimization using the sample covariance
matrix with no shrinkage or regularization. Known to perform poorly
out of sample due to estimation error, but essential as a baseline.
"""

from __future__ import annotations

import logging

import pandas as pd

from pnlp.portfolio.optimizer import PortfolioOptimizer, PortfolioWeights

logger = logging.getLogger(__name__)

__all__ = ["sample_cov_portfolio"]


def sample_cov_portfolio(
    returns: pd.DataFrame,
    objective: str = "min_variance",
    long_only: bool = True,
    max_weight: float = 0.05,
) -> PortfolioWeights:
    """Construct a portfolio using sample covariance.

    Args:
        returns: DataFrame with tickers as columns, dates as index.
        objective: optimization objective.
        long_only: whether to constrain weights >= 0.
        max_weight: maximum position size.

    Returns:
        PortfolioWeights with optimal allocations.
    """
    from pnlp.config import PortfolioConfig

    valid_cols = returns.columns[returns.notna().sum() > returns.shape[0] * 0.5]
    clean_returns = returns[valid_cols].dropna()

    if len(clean_returns) < 10 or len(clean_returns.columns) < 2:
        logger.warning("Insufficient data for sample covariance")
        return PortfolioWeights(
            weights=pd.Series(dtype=float),
            objective_value=float("inf"),
            metadata={"method": "sample_covariance", "error": "insufficient_data"},
        )

    cov = clean_returns.cov()

    config = PortfolioConfig(
        objective=objective,
        long_only=long_only,
        max_weight=max_weight,
    )
    optimizer = PortfolioOptimizer(config)
    result = optimizer.optimize(cov)

    result.metadata["method"] = "sample_covariance"
    return result
