"""
Parquet-based document store with embedding cache.

Stores filing documents and their embeddings locally to avoid
re-downloading and re-embedding on every run.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pnlp.config import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["DocumentStore"]

STORE_DIR = DATA_DIR / "store"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
TRANSCRIPT_EMBEDDINGS_DIR = DATA_DIR / "embeddings_transcripts"
NEWS_EMBEDDINGS_DIR = DATA_DIR / "embeddings_news"


class DocumentStore:
    """Persistent storage for firm documents and embeddings.

    Layout:
        data/store/filings_index.parquet   — metadata index
        data/embeddings/{ticker}.npz       — per-firm embedding arrays
    """

    def __init__(
        self,
        store_dir: Path | None = None,
        embeddings_dir: Path | None = None,
    ) -> None:
        self.store_dir = store_dir or STORE_DIR
        self.embeddings_dir = embeddings_dir or EMBEDDINGS_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def save_embeddings(
        self,
        ticker: str,
        embeddings: np.ndarray,
        dates: list,
        metadata: dict | None = None,
    ) -> None:
        """Save per-firm embeddings as a compressed .npz file.

        Args:
            ticker: firm ticker symbol.
            embeddings: (n_docs, embed_dim) array.
            dates: list of filing dates corresponding to each row.
            metadata: optional dict of additional arrays to store.
        """
        path = self.embeddings_dir / f"{ticker}.npz"
        save_dict = {
            "embeddings": embeddings,
            "dates": np.array([str(d) for d in dates]),
        }
        if metadata:
            save_dict.update(metadata)
        np.savez_compressed(path, **save_dict)

    def load_embeddings(self, ticker: str) -> dict[str, np.ndarray] | None:
        """Load per-firm embeddings. Returns None if not cached."""
        path = self.embeddings_dir / f"{ticker}.npz"
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        d = dict(data)
        if "embeddings" not in d:
            raise ValueError(f"Missing 'embeddings' key in {path}")
        if "dates" not in d:
            raise ValueError(f"Missing 'dates' key in {path}")
        if d["embeddings"].ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {d['embeddings'].shape} in {path}")
        if len(d["dates"]) != d["embeddings"].shape[0]:
            raise ValueError(
                f"Row count mismatch: {d['embeddings'].shape[0]} embeddings vs {len(d['dates'])} dates in {path}"
            )
        return d

    def load_document_embeddings(
        self,
        ticker: str,
        doc_type: str = "10-K",
    ) -> list | None:
        """Load per-firm embeddings as DocumentEmbedding objects.

        Returns None if no embeddings are cached for this ticker.
        Reconstructs DocumentEmbedding objects from stored arrays.

        Args:
            ticker: firm ticker symbol.
            doc_type: document type label for reconstructed objects.
        """
        from pnlp.embeddings.document_embedder import DocumentEmbedding

        raw = self.load_embeddings(ticker)
        if raw is None:
            return None

        embeddings = raw["embeddings"]  # (n_docs, embed_dim)
        date_strs = raw["dates"]  # array of 'YYYY-MM-DD' strings
        norms = raw.get("norms")  # (n_docs,) — may not exist in old files
        n_chunks = raw.get("n_chunks")  # (n_docs,) — may not exist

        result = []
        for i in range(len(embeddings)):
            # Parse date string back to date object
            d = date.fromisoformat(str(date_strs[i]))
            emb = embeddings[i]
            norm = float(norms[i]) if norms is not None else float(np.linalg.norm(emb))
            chunks = int(n_chunks[i]) if n_chunks is not None else 1

            result.append(
                DocumentEmbedding(
                    ticker=ticker,
                    doc_date=d,
                    doc_type=doc_type,
                    embedding=emb,
                    norm=norm,
                    n_chunks=chunks,
                )
            )

        return result

    def has_embeddings(self, ticker: str) -> bool:
        """Check if valid embeddings exist for a ticker.

        Does a basic integrity check: file must exist, contain 'embeddings'
        key, and have the right number of dimensions.
        """
        path = self.embeddings_dir / f"{ticker}.npz"
        if not path.exists():
            return False
        try:
            data = np.load(path, allow_pickle=True)
            emb = data["embeddings"]
            return emb.ndim == 2 and emb.shape[1] > 0
        except Exception:
            return False

    def save_index(self, df: pd.DataFrame) -> None:
        """Save a metadata index DataFrame."""
        path = self.store_dir / "filings_index.parquet"
        df.to_parquet(path)

    def load_index(self) -> pd.DataFrame | None:
        """Load the metadata index. Returns None if not saved."""
        path = self.store_dir / "filings_index.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)
