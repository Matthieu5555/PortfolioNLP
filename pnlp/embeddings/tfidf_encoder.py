"""
TF-IDF document encoder for look-ahead-bias-free baseline.

Computes document embeddings using TF-IDF vectors instead of a
pretrained language model. TF-IDF has zero look-ahead bias by
construction because it uses only corpus-internal term frequencies.

For walk-forward correctness, the vectorizer is fitted only on
documents with doc_date <= as_of_date at each rebalance date.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pnlp.embeddings.document_embedder import DocumentEmbedding

logger = logging.getLogger(__name__)

__all__ = ["TfidfFirmEncoder"]


class TfidfFirmEncoder:
    """Walk-forward TF-IDF encoder that produces firm-level embeddings.

    At each as_of_date:
    1. Collects all document texts with doc_date <= as_of_date
    2. Fits TF-IDF vectorizer on this temporal corpus
    3. For each eligible firm, transforms its most recent document(s)
    4. Mean-pools and L2-normalizes to produce a firm embedding

    The result is compatible with FirmEmbedding and can be fed into
    the same cosine similarity → shrinkage pipeline.
    """

    def __init__(
        self,
        max_features: int = 768,
        min_df: int = 5,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

    def encode_universe(
        self,
        filing_texts: dict[str, list[tuple[date, str]]],
        as_of_date: date,
        eligible_tickers: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Encode all firms using walk-forward TF-IDF.

        Args:
            filing_texts: ticker -> list of (doc_date, text) tuples.
            as_of_date: point-in-time cutoff (strict <=).
            eligible_tickers: optional subset to return. If None, all tickers
                with at least one document are returned.

        Returns:
            dict mapping ticker -> L2-normalized TF-IDF embedding (max_features,).
        """
        # Collect all documents up to as_of_date for vocabulary fitting
        all_texts = []
        all_labels = []  # (ticker, doc_date) for each text

        for ticker, docs in filing_texts.items():
            for doc_date, text in docs:
                if doc_date <= as_of_date and len(text) >= 100:
                    all_texts.append(text)
                    all_labels.append((ticker, doc_date))

        if len(all_texts) < 10:
            logger.warning("Only %d documents before %s, skipping TF-IDF", len(all_texts), as_of_date)
            return {}

        # Fit vectorizer on the temporal corpus
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
            dtype=np.float32,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)  # (n_docs, max_features)
        except ValueError as e:
            logger.warning("TF-IDF fitting failed at %s: %s", as_of_date, e)
            return {}

        # Group document vectors by ticker, take mean of available documents
        ticker_set = set(eligible_tickers) if eligible_tickers else None
        ticker_vectors: dict[str, list[np.ndarray]] = {}

        for i, (ticker, doc_date) in enumerate(all_labels):
            if ticker_set and ticker not in ticker_set:
                continue
            vec = tfidf_matrix[i].toarray().ravel()
            if ticker not in ticker_vectors:
                ticker_vectors[ticker] = []
            ticker_vectors[ticker].append(vec)

        # Mean-pool and L2-normalize per firm
        result: dict[str, np.ndarray] = {}
        for ticker, vecs in ticker_vectors.items():
            mean_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            result[ticker] = mean_vec.astype(np.float32)

        logger.info(
            "TF-IDF encoded %d firms from %d docs (as_of=%s, vocab=%d)",
            len(result), len(all_texts), as_of_date, len(vectorizer.vocabulary_),
        )
        return result
