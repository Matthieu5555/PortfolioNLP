"""
Shared embeddings loader with text-source selection.

Loads document embeddings from one or more sources (10-K filings,
earnings transcripts) and merges them per-ticker.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from pnlp.config import DATA_DIR
from pnlp.data.store import (
    EMBEDDINGS_DIR,
    NEWS_EMBEDDINGS_DIR,
    TRANSCRIPT_EMBEDDINGS_DIR,
    DocumentStore,
)
from pnlp.embeddings.document_embedder import DocumentEmbedding

logger = logging.getLogger(__name__)

__all__ = ["load_doc_embeddings"]

# Mapping from CLI text-source names to (directory, doc_type) pairs
TEXT_SOURCE_CONFIG = {
    "10k": [(EMBEDDINGS_DIR, "10-K")],
    "transcript": [(TRANSCRIPT_EMBEDDINGS_DIR, "transcript")],
    "news": [(NEWS_EMBEDDINGS_DIR, "news")],
    "combined": [
        (EMBEDDINGS_DIR, "10-K"),
        (TRANSCRIPT_EMBEDDINGS_DIR, "transcript"),
    ],
    "all": [
        (EMBEDDINGS_DIR, "10-K"),
        (TRANSCRIPT_EMBEDDINGS_DIR, "transcript"),
        (NEWS_EMBEDDINGS_DIR, "news"),
    ],
}


def load_doc_embeddings(
    text_source: str = "10k",
) -> dict[str, list[DocumentEmbedding]]:
    """Load document embeddings from one or more text sources.

    Args:
        text_source: one of "10k", "transcript", "combined".

    Returns:
        dict mapping ticker -> list of DocumentEmbedding objects.
    """
    if text_source not in TEXT_SOURCE_CONFIG:
        raise ValueError(
            f"Unknown text_source={text_source!r}. "
            f"Choose from: {list(TEXT_SOURCE_CONFIG.keys())}"
        )

    sources = TEXT_SOURCE_CONFIG[text_source]
    all_docs: dict[str, list[DocumentEmbedding]] = {}

    for emb_dir, doc_type in sources:
        if not emb_dir.exists():
            logger.warning(
                "Embeddings directory %s does not exist, skipping %s",
                emb_dir, doc_type,
            )
            continue

        store = DocumentStore(embeddings_dir=emb_dir)
        npz_files = sorted(emb_dir.glob("*.npz"))
        logger.info(
            "Loading %s embeddings: %d tickers from %s",
            doc_type, len(npz_files), emb_dir,
        )

        for npz_file in tqdm(npz_files, desc=f"Loading {doc_type}"):
            ticker = npz_file.stem
            docs = store.load_document_embeddings(ticker, doc_type=doc_type)
            if docs:
                if ticker in all_docs:
                    all_docs[ticker].extend(docs)
                else:
                    all_docs[ticker] = docs

    total_docs = sum(len(v) for v in all_docs.values())
    logger.info(
        "Loaded %d tickers, %d total documents (source=%s)",
        len(all_docs), total_docs, text_source,
    )
    return all_docs
