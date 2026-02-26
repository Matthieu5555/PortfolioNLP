"""
Expected return proxies from embedding geometry.

Replaces traditional expected return estimates with signals derived
from the geometric properties of firm embeddings in text space.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from pnlp.config import MuConfig
from pnlp.embeddings.firm_aggregator import FirmEmbedding

logger = logging.getLogger(__name__)

__all__ = [
    "MuEstimator",
    "ReturnsDirectionMu",
    "ClusterDistanceMu",
    "TemporalDriftMu",
]


class MuEstimator(ABC):
    """Abstract base for expected return estimation from embeddings."""

    @abstractmethod
    def fit(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        historical_returns: pd.Series,
    ) -> None:
        """Fit the estimator on historical data.

        Args:
            firm_embeddings: ticker -> FirmEmbedding (at training time).
            historical_returns: ticker -> realized return over the horizon.
        """
        ...

    @abstractmethod
    def predict(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
    ) -> pd.Series:
        """Predict expected returns for new embeddings.

        Returns:
            Series with ticker index and expected return values.
        """
        ...


class ReturnsDirectionMu(MuEstimator):
    """Find the direction in embedding space that correlates with returns.

    The idea: if there exists a vector d in R^768 such that <e_i, d>
    correlates with future returns r_i, then projecting new embeddings
    onto d gives an expected return proxy.

    Implementation: cross-sectional ridge regression.
        r_i = alpha + beta * <e_i, d> + epsilon

    where d is the coefficient vector of the ridge regression.
    Heavy regularization is critical (768 dims, ~2000 firms).
    """

    def __init__(self, config: MuConfig | None = None) -> None:
        self.config = config or MuConfig()
        self._model: Ridge | None = None

    def fit(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        historical_returns: pd.Series,
    ) -> None:
        # Align tickers
        common = sorted(set(firm_embeddings.keys()) & set(historical_returns.index))
        if len(common) < 10:
            logger.warning("Too few firms (%d) for ReturnsDirectionMu", len(common))
            return

        X = np.array([firm_embeddings[t].embedding for t in common])
        y = historical_returns[common].values

        # Remove NaN/inf
        mask = np.isfinite(y)
        X = X[mask]
        y = y[mask]

        self._model = Ridge(alpha=self.config.regularization)
        self._model.fit(X, y)

        # Log out-of-sample R2 via 5-fold CV (not in-sample)
        if len(y) >= 50:
            oos_r2 = cross_val_score(
                Ridge(alpha=self.config.regularization), X, y, cv=5, scoring="r2"
            ).mean()
            logger.info("Fitted returns direction on %d firms (in-sample R2=%.4f, OOS R2=%.4f)",
                        len(y), self._model.score(X, y), oos_r2)
        else:
            logger.info("Fitted returns direction on %d firms (R2=%.4f)", len(y), self._model.score(X, y))

    def predict(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
    ) -> pd.Series:
        if self._model is None:
            logger.warning("ReturnsDirectionMu not fitted; returning zeros")
            return pd.Series(0.0, index=sorted(firm_embeddings.keys()))

        tickers = sorted(firm_embeddings.keys())
        X = np.array([firm_embeddings[t].embedding for t in tickers])
        predictions = self._model.predict(X)
        return pd.Series(predictions, index=tickers)


class ClusterDistanceMu(MuEstimator):
    """Expected returns from relative cluster positioning.

    Clusters the embedding space, labels clusters by average realized
    return, then scores each firm by its distance to high-return vs
    low-return cluster centroids.

    Less prone to overfitting than ReturnsDirectionMu because the
    signal is lower-dimensional (cluster distances vs full ridge).
    """

    def __init__(self, config: MuConfig | None = None) -> None:
        self.config = config or MuConfig()
        self._kmeans: KMeans | None = None
        self._cluster_returns: np.ndarray | None = None

    def fit(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        historical_returns: pd.Series,
    ) -> None:
        common = sorted(set(firm_embeddings.keys()) & set(historical_returns.index))
        if len(common) < self.config.n_clusters * 2:
            logger.warning("Too few firms for ClusterDistanceMu")
            return

        X = np.array([firm_embeddings[t].embedding for t in common])
        y = historical_returns[common].values

        mask = np.isfinite(y)
        X = X[mask]
        y = y[mask]

        self._kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=42, n_init=10)
        labels = self._kmeans.fit_predict(X)

        # Average return per cluster
        self._cluster_returns = np.array([
            y[labels == k].mean() if (labels == k).any() else 0.0
            for k in range(self.config.n_clusters)
        ])

        logger.info(
            "Cluster returns: %s",
            {k: f"{r:.4f}" for k, r in enumerate(self._cluster_returns)},
        )

    def predict(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
    ) -> pd.Series:
        if self._kmeans is None or self._cluster_returns is None:
            return pd.Series(0.0, index=sorted(firm_embeddings.keys()))

        tickers = sorted(firm_embeddings.keys())
        X = np.array([firm_embeddings[t].embedding for t in tickers])

        # Distance from each firm to each cluster centroid
        distances = self._kmeans.transform(X)  # (n_firms, n_clusters)

        # Weighted average: closer to high-return clusters = higher expected return
        # Use inverse distance weighting
        inv_dist = 1.0 / (distances + 1e-8)
        weights = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        predictions = weights @ self._cluster_returns

        return pd.Series(predictions, index=tickers)


class TemporalDriftMu(MuEstimator):
    """Expected returns from embedding movement over time.

    Firms whose embeddings are moving (new filings shift the firm
    representation) may be undergoing fundamental changes. The
    magnitude and direction of drift could predict returns.

    drift_i = e_i(t) - e_i(t-1)
    mu_i = ||drift_i|| * sign(correlation with returns direction)

    Requires embeddings at two time points per firm.
    """

    def __init__(self, config: MuConfig | None = None) -> None:
        self.config = config or MuConfig()
        self._returns_direction: np.ndarray | None = None

    def fit(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        historical_returns: pd.Series,
    ) -> None:
        """Fit to learn the 'returns direction' for projecting drift."""
        common = sorted(set(firm_embeddings.keys()) & set(historical_returns.index))
        if len(common) < 10:
            return

        X = np.array([firm_embeddings[t].embedding for t in common])
        y = historical_returns[common].values

        mask = np.isfinite(y)
        X = X[mask]
        y = y[mask]

        model = Ridge(alpha=self.config.regularization)
        model.fit(X, y)
        self._returns_direction = model.coef_
        self._returns_direction = self._returns_direction / (np.linalg.norm(self._returns_direction) + 1e-10)

    def predict(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        previous_embeddings: dict[str, FirmEmbedding] | None = None,
    ) -> pd.Series:
        """Predict from embedding drift.

        Args:
            firm_embeddings: current period embeddings.
            previous_embeddings: previous period embeddings.
                If None, returns zeros.
        """
        if previous_embeddings is None or self._returns_direction is None:
            return pd.Series(0.0, index=sorted(firm_embeddings.keys()))

        result: dict[str, float] = {}
        for ticker, current in firm_embeddings.items():
            prev = previous_embeddings.get(ticker)
            if prev is None:
                result[ticker] = 0.0
                continue

            drift = current.embedding - prev.embedding
            # Project drift onto the returns direction
            projected = float(np.dot(drift, self._returns_direction))
            result[ticker] = projected

        return pd.Series(result)
