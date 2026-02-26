"""
Walk-forward rebalancing engine.

At each rebalancing date:
1. Collect all documents up to the rebalancing date
2. Aggregate into firm embeddings (no look-ahead)
3. Construct mu, sigma, Sigma from embedding geometry
4. Optimize portfolio
5. Record weights and metadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from pnlp.config import (
    CovarianceConfig,
    EmbeddingConfig,
    MuConfig,
    PortfolioConfig,
    SigmaConfig,
)
from pnlp.embeddings.document_embedder import DocumentEmbedding
from pnlp.embeddings.firm_aggregator import FirmAggregator, FirmEmbedding
from pnlp.portfolio.optimizer import PortfolioOptimizer, PortfolioWeights
from pnlp.primitives.covariance import CosineSimilarityCovariance
from pnlp.primitives.mu import MuEstimator
from pnlp.primitives.sigma import SigmaEstimator

logger = logging.getLogger(__name__)

__all__ = ["RebalanceResult", "WalkForwardRebalancer"]


@dataclass
class RebalanceResult:
    """Result of a single rebalancing step."""

    date: date
    weights: PortfolioWeights
    n_firms: int
    turnover: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class WalkForwardRebalancer:
    """Walk-forward portfolio rebalancing engine.

    Iterates over rebalancing dates, building text-derived portfolio
    inputs fresh at each date using only information available at that
    time (no look-ahead bias).
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig | None = None,
        mu_config: MuConfig | None = None,
        sigma_config: SigmaConfig | None = None,
        cov_config: CovarianceConfig | None = None,
        portfolio_config: PortfolioConfig | None = None,
        mu_estimator: MuEstimator | None = None,
        sigma_estimator: SigmaEstimator | None = None,
    ) -> None:
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.cov_config = cov_config or CovarianceConfig()

        self.aggregator = FirmAggregator(self.embedding_config)
        self.optimizer = PortfolioOptimizer(self.portfolio_config)
        self.cov_estimator = CosineSimilarityCovariance(self.cov_config)
        self.mu_estimator = mu_estimator
        self.sigma_estimator = sigma_estimator

    def run(
        self,
        doc_embeddings: dict[str, list[DocumentEmbedding]],
        rebalance_dates: list[date],
        historical_returns: pd.DataFrame | None = None,
    ) -> list[RebalanceResult]:
        """Run walk-forward rebalancing.

        Args:
            doc_embeddings: ticker -> list of DocumentEmbedding (all periods).
            rebalance_dates: dates at which to rebalance.
            historical_returns: ticker x date DataFrame of log returns.
                Required for mu estimation and baselines.

        Returns:
            List of RebalanceResult, one per rebalancing date.
        """
        results: list[RebalanceResult] = []
        prev_weights: pd.Series | None = None

        for rebal_date in sorted(rebalance_dates):
            logger.info("Rebalancing at %s", rebal_date)

            # 1. Aggregate firm embeddings as of this date
            firm_embeddings = self.aggregator.aggregate_universe(
                doc_embeddings, as_of_date=rebal_date,
            )

            if len(firm_embeddings) < 10:
                logger.warning("Only %d firms at %s, skipping", len(firm_embeddings), rebal_date)
                continue

            # 2. Estimate sigma (volatility proxy)
            sigma_est = None
            if self.sigma_estimator is not None:
                sigma_est = self.sigma_estimator.estimate(
                    firm_embeddings,
                    doc_embeddings={t: doc_embeddings.get(t, []) for t in firm_embeddings},
                )

            # 3. Estimate covariance
            cov = self.cov_estimator.estimate(firm_embeddings, sigma_est)

            # 4. Estimate mu (expected returns) if needed
            mu = None
            if self.mu_estimator is not None and historical_returns is not None:
                # Compute realized returns up to this date for fitting
                returns_to_date = historical_returns.loc[:str(rebal_date)]
                if len(returns_to_date) > 60:
                    # Use last-period returns as training signal
                    horizon = 63  # quarterly
                    if len(returns_to_date) > horizon:
                        period_returns = returns_to_date.iloc[-horizon:].sum()
                        self.mu_estimator.fit(firm_embeddings, period_returns)
                        mu = self.mu_estimator.predict(firm_embeddings)

            # 5. Optimize
            portfolio = self.optimizer.optimize(cov, mu)

            # 6. Compute turnover
            turnover = 0.0
            if prev_weights is not None:
                common = set(portfolio.weights.index) & set(prev_weights.index)
                for t in common:
                    turnover += abs(portfolio.weights.get(t, 0) - prev_weights.get(t, 0))
                # New positions
                for t in set(portfolio.weights.index) - common:
                    turnover += abs(portfolio.weights.get(t, 0))
                # Closed positions
                for t in set(prev_weights.index) - common:
                    turnover += abs(prev_weights.get(t, 0))

            results.append(
                RebalanceResult(
                    date=rebal_date,
                    weights=portfolio,
                    n_firms=len(firm_embeddings),
                    turnover=turnover,
                )
            )

            prev_weights = portfolio.weights
            logger.info(
                "  -> %d positions, turnover=%.3f, obj=%.6f",
                (portfolio.weights.abs() > 1e-6).sum(),
                turnover,
                portfolio.objective_value,
            )

        return results
