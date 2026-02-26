"""
Aggregate document embeddings into per-firm representations.

Multiple documents (10-K filings, earnings calls, news articles) about
a firm are combined into a single embedding vector using configurable
aggregation strategies.

Document embeddings are aggregated as raw vectors (not unit vectors) —
their norms carry coherence information.  L2-normalization happens only
once, at the final firm-level output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np

from pnlp.config import EmbeddingConfig
from pnlp.embeddings.document_embedder import DocumentEmbedding

logger = logging.getLogger(__name__)

__all__ = ["FirmEmbedding", "FirmAggregator"]


@dataclass(frozen=True)
class FirmEmbedding:
    """Per-firm embedding at a specific point in time."""

    ticker: str
    as_of_date: date
    embedding: np.ndarray  # (embed_dim,) L2-normalized
    n_documents: int


class FirmAggregator:
    """Aggregate document embeddings into per-firm representations.

    Strategies:
        'mean': Simple mean of all document embeddings (raw vectors, not unit).
            Baseline approach. Matches Hoberg-Phillips in spirit.

        'time_weighted': Exponential decay weighting. Recent filings
            contribute more. Half-life controlled by config.
            w_i = exp(-ln(2) * (as_of_date - doc_date) / halflife)
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()

    def aggregate(
        self,
        doc_embeddings: list[DocumentEmbedding],
        as_of_date: date,
    ) -> FirmEmbedding | None:
        """Aggregate document embeddings for a single firm.

        Only includes documents dated on or before as_of_date (no look-ahead).
        Returns None if no valid documents exist.
        """
        # Filter to documents available at as_of_date
        valid = [d for d in doc_embeddings if d.doc_date <= as_of_date]
        if not valid:
            return None

        ticker = valid[0].ticker
        # Use raw embeddings (not unit vectors) — norms carry coherence signal
        embeddings = np.array([d.embedding for d in valid])  # (n, dim)

        if self.config.aggregation == "time_weighted":
            halflife = self.config.time_decay_halflife_days
            days_ago = np.array([(as_of_date - d.doc_date).days for d in valid], dtype=np.float64)
            weights = np.exp(-np.log(2) * days_ago / halflife)
            weights = weights / weights.sum()
            aggregated = (embeddings * weights[:, np.newaxis]).sum(axis=0)
        else:
            # Default: mean pooling
            aggregated = embeddings.mean(axis=0)

        # L2-normalize only at the firm level — this is the single normalization point
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm

        return FirmEmbedding(
            ticker=ticker,
            as_of_date=as_of_date,
            embedding=aggregated.astype(np.float32),
            n_documents=len(valid),
        )

    def aggregate_universe(
        self,
        all_doc_embeddings: dict[str, list[DocumentEmbedding]],
        as_of_date: date,
        min_documents: int = 1,
    ) -> dict[str, FirmEmbedding]:
        """Aggregate embeddings for all firms in the universe.

        Args:
            all_doc_embeddings: ticker -> list of DocumentEmbedding.
            as_of_date: point-in-time for aggregation (no look-ahead).
            min_documents: minimum documents required per firm.

        Returns:
            dict mapping ticker -> FirmEmbedding for firms with enough data.
        """
        result: dict[str, FirmEmbedding] = {}

        for ticker, docs in all_doc_embeddings.items():
            firm_emb = self.aggregate(docs, as_of_date)
            if firm_emb is not None and firm_emb.n_documents >= min_documents:
                result[ticker] = firm_emb

        logger.info(
            "Aggregated %d/%d firms (as_of=%s, min_docs=%d)",
            len(result),
            len(all_doc_embeddings),
            as_of_date,
            min_documents,
        )
        return result
