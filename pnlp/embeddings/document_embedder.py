"""
Embed individual documents (10-K filings, earnings calls, news).

Long documents are split into chunks at sentence boundaries, each chunk
is embedded independently, and the results are mean-pooled. This handles
the 512-token context window limit of bge-base-en-v1.5.

The mean-pooled vector is NOT L2-normalized at the document level — its
norm carries a coherence signal (low norm ≈ diverse/incoherent content).
L2-normalization happens only at the firm-aggregation level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
from nltk.tokenize import sent_tokenize

from pnlp.config import EmbeddingConfig
from pnlp.embeddings.text_encoder import TextEncoder

logger = logging.getLogger(__name__)

__all__ = ["DocumentEmbedding", "DocumentEmbedder"]


@dataclass(frozen=True)
class DocumentEmbedding:
    """Embedding of a single document."""

    ticker: str
    doc_date: date
    doc_type: str  # "10-K", "transcript", "news"
    embedding: np.ndarray  # (embed_dim,) — raw mean-pool, NOT L2-normalized
    norm: float  # L2 norm of the mean-pooled vector (coherence signal)
    n_chunks: int  # how many chunks the document was split into


# Rough approximation: 1 token ~ 3.5 chars for SEC legal text.
# (English prose is ~4, but SEC filings have more abbreviations and dense terms.)
_CHARS_PER_TOKEN = 3.5


def _chunk_text(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks of approximately max_tokens tokens.

    Uses nltk sentence tokenizer which correctly handles abbreviations
    like U.S., Inc., Mr., No., etc.
    """
    max_chars = int(max_tokens * _CHARS_PER_TOKEN)

    if len(text) <= max_chars:
        return [text]

    sentences = sent_tokenize(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > max_chars and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c.strip()) > 50]


class DocumentEmbedder:
    """Embed documents into dense vectors with chunking for long texts.

    For 10-K filings (often 50+ pages), the document is split into
    ~500-word chunks, each embedded independently, then mean-pooled.

    The resulting vector is NOT L2-normalized — the norm of the mean
    carries useful information (document coherence / topical focus).
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        encoder: TextEncoder | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self.encoder = encoder or TextEncoder(self.config.model_name)

    def embed_text(self, text: str) -> tuple[np.ndarray, float, int]:
        """Embed a single text, chunking if necessary.

        Returns (embedding, norm, n_chunks) where embedding is the raw
        mean-pool of chunk embeddings (NOT L2-normalized).
        """
        chunks = _chunk_text(text, self.config.max_chunk_tokens)
        if not chunks:
            return np.zeros(self.config.text_dim, dtype=np.float32), 0.0, 0

        chunk_embeddings = self.encoder.encode(chunks)  # (n_chunks, dim)
        # Mean-pool across chunks — do NOT L2-normalize
        pooled = chunk_embeddings.mean(axis=0)
        norm = float(np.linalg.norm(pooled))

        return pooled.astype(np.float32), norm, len(chunks)

    def embed_documents(
        self,
        texts: list[str],
        tickers: list[str],
        dates: list[date],
        doc_types: list[str],
    ) -> list[DocumentEmbedding]:
        """Embed a batch of documents.

        All four input lists must have the same length.
        """
        results = []
        for text, ticker, doc_date, doc_type in zip(texts, tickers, dates, doc_types):
            embedding, norm, n_chunks = self.embed_text(text)
            results.append(
                DocumentEmbedding(
                    ticker=ticker,
                    doc_date=doc_date,
                    doc_type=doc_type,
                    embedding=embedding,
                    norm=norm,
                    n_chunks=n_chunks,
                )
            )

        logger.info("Embedded %d documents", len(results))
        return results
