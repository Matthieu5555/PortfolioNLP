"""
Volatility proxies from embedding dispersion.

Replaces realized volatility with estimates derived from the spread
and uncertainty of a firm's document embeddings in text space.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date

import numpy as np
import pandas as pd

from pnlp.config import SigmaConfig
from pnlp.embeddings.document_embedder import DocumentEmbedding
from pnlp.embeddings.firm_aggregator import FirmEmbedding

logger = logging.getLogger(__name__)

__all__ = [
    "SigmaEstimator",
    "DispersionSigma",
    "CentroidDistanceSigma",
    "TemporalPCASigma",
]


class SigmaEstimator(ABC):
    """Abstract base for volatility estimation from embeddings."""

    @abstractmethod
    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        doc_embeddings: dict[str, list[DocumentEmbedding]] | None = None,
    ) -> pd.Series:
        """Estimate per-firm volatility proxy.

        Args:
            firm_embeddings: ticker -> aggregated FirmEmbedding.
            doc_embeddings: ticker -> list of per-document embeddings.
                Required for dispersion and entropy methods.

        Returns:
            Series with ticker index and volatility proxy values.
        """
        ...


class DispersionSigma(SigmaEstimator):
    """Volatility proxy from embedding dispersion.

    Firms discussed with varied, inconsistent language across documents
    are treated as riskier. Measured as the average distance of individual
    document embeddings from the firm's centroid.

    sigma_text_i = mean(||e_ij - centroid_i||) across documents j

    Intuition: a firm with consistent business descriptions is more
    predictable (lower vol). A firm where each filing tells a different
    story has higher narrative uncertainty.
    """

    def __init__(self, config: SigmaConfig | None = None) -> None:
        self.config = config or SigmaConfig()

    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        doc_embeddings: dict[str, list[DocumentEmbedding]] | None = None,
    ) -> pd.Series:
        if doc_embeddings is None:
            logger.warning("No doc_embeddings; returning uniform volatility")
            return pd.Series(1.0, index=sorted(firm_embeddings.keys()))

        result: dict[str, float] = {}
        for ticker, firm_emb in firm_embeddings.items():
            docs = doc_embeddings.get(ticker, [])
            if len(docs) < 2:
                # Single-doc firms: use universe average (1.0 after normalization)
                # rather than 0.0, which would make the optimizer treat them as risk-free
                result[ticker] = np.nan  # will be filled with mean after normalization
                continue

            doc_vecs = np.array([d.embedding for d in docs])
            centroid = firm_emb.embedding

            # Average L2 distance from centroid
            distances = np.linalg.norm(doc_vecs - centroid, axis=1)
            result[ticker] = float(distances.mean())

        series = pd.Series(result)

        # Normalize to have mean 1 (so it's a relative risk measure)
        valid_mean = series.dropna().mean()
        if valid_mean > 0:
            series = series / valid_mean

        # Fill single-doc firms with 1.0 (universe average risk)
        series = series.fillna(1.0)

        return series


class CentroidDistanceSigma(SigmaEstimator):
    """Volatility proxy from distance to sector centroid.

    Firms that are semantically unusual (far from the centroid of
    all firms in the universe) are treated as riskier. This captures
    "uniqueness risk" — firms that don't fit neatly into existing
    categories are harder for the market to price.

    sigma_text_i = ||e_i - universe_centroid||
    """

    def __init__(self, config: SigmaConfig | None = None) -> None:
        self.config = config or SigmaConfig()

    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        doc_embeddings: dict[str, list[DocumentEmbedding]] | None = None,
    ) -> pd.Series:
        tickers = sorted(firm_embeddings.keys())
        embeddings = np.array([firm_embeddings[t].embedding for t in tickers])

        # Universe centroid
        centroid = embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

        distances = np.linalg.norm(embeddings - centroid, axis=1)
        series = pd.Series(distances, index=tickers)

        # Normalize to mean 1
        if series.mean() > 0:
            series = series / series.mean()

        return series


class TemporalPCASigma(SigmaEstimator):
    """Volatility proxy from PCA on per-firm embedding trajectories.

    For each firm, stacks its document embeddings into a T x 768 matrix
    and computes the total dispersion (Frobenius norm of centered matrix)
    as a measure of "embedding volatility" — how much the firm's narrative
    changes over time.

    Also computes a concentration ratio (fraction of variance in top-1 PC)
    that distinguishes firms with a single dominant axis of change from
    firms with diffuse variation across many dimensions.

    Requires doc_embeddings (not just firm_embeddings) since it needs
    the per-document time series.
    """

    def __init__(
        self,
        config: SigmaConfig | None = None,
        as_of_date: date | None = None,
        min_documents: int = 3,
    ) -> None:
        self.config = config or SigmaConfig()
        self.as_of_date = as_of_date
        self.min_documents = min_documents
        self.concentration_ratios_: dict[str, float] = {}

    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        doc_embeddings: dict[str, list[DocumentEmbedding]] | None = None,
    ) -> pd.Series:
        if doc_embeddings is None:
            logger.warning("No doc_embeddings; returning uniform volatility")
            return pd.Series(1.0, index=sorted(firm_embeddings.keys()))

        result: dict[str, float] = {}
        self.concentration_ratios_ = {}

        for ticker in firm_embeddings:
            docs = doc_embeddings.get(ticker, [])

            # Filter by as_of_date if provided (look-ahead prevention)
            if self.as_of_date is not None:
                docs = [d for d in docs if d.doc_date <= self.as_of_date]

            if len(docs) < self.min_documents:
                result[ticker] = np.nan
                self.concentration_ratios_[ticker] = np.nan
                continue

            # Stack into T x embed_dim matrix
            M = np.array([d.embedding for d in docs])

            # Center (subtract temporal mean)
            M_centered = M - M.mean(axis=0)

            # Singular values of centered matrix
            s = np.linalg.svd(M_centered, compute_uv=False)
            s_sq = s ** 2

            # Total embedding volatility = sqrt(sum of eigenvalues of cov)
            # = Frobenius norm of centered matrix / sqrt(T-1)
            total_var = s_sq.sum()
            result[ticker] = float(np.sqrt(total_var / (len(docs) - 1)))

            # Concentration ratio: fraction of variance in top-1 PC
            total = s_sq.sum()
            self.concentration_ratios_[ticker] = float(s_sq[0] / total) if total > 0 else 0.0

        series = pd.Series(result)

        # Normalize: divide by median so median firm has sigma ~ 1.0
        valid_median = series.dropna().median()
        if valid_median > 0:
            series = series / valid_median

        # Fill firms with too few documents with 1.0 (median risk)
        series = series.fillna(1.0)

        return series
