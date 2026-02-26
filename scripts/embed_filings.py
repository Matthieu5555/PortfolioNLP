"""
Embed all downloaded 10-K filings.

Reads filing text from data/filings/{ticker}/{year}.txt, applies XBRL
cleaning, embeds each document (chunking + mean-pooling), and saves
per-firm embedding arrays to data/embeddings/{ticker}.npz.

Uses actual SEC filing dates (from filings_index.parquet) to prevent
look-ahead bias — a 2023 10-K filed in Feb 2024 gets dated Feb 2024.

Resumes where it left off — skips tickers that already have embeddings.

Usage:
    uv run python scripts/embed_filings.py
    uv run python scripts/embed_filings.py --force   # re-embed everything
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR
from pnlp.data.filing_cleaner import clean_filing_text
from pnlp.data.store import DocumentStore
from pnlp.embeddings.document_embedder import DocumentEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embed")


def main() -> None:
    force = "--force" in sys.argv

    filings_dir = DATA_DIR / "filings"
    index_path = DATA_DIR / "filings_index.parquet"

    if not index_path.exists():
        logger.error("No filings_index.parquet found. Run download_data.py first.")
        return

    index = pd.read_parquet(index_path)

    # Verify filed_date column exists
    if "filed_date" not in index.columns:
        logger.error(
            "filings_index.parquet has no 'filed_date' column. "
            "Run scripts/fetch_filing_dates.py first."
        )
        return

    # Build lookup: (ticker, year) -> filed_date
    date_lookup: dict[tuple[str, int], date] = {}
    for _, row in index.iterrows():
        filed_str = str(row["filed_date"])
        try:
            d = date.fromisoformat(filed_str)
        except (ValueError, TypeError):
            # Conservative fallback: March 15 of next year
            d = date(int(row["year"]) + 1, 3, 15)
        date_lookup[(row["ticker"], int(row["year"]))] = d

    tickers = sorted(index["ticker"].unique())
    logger.info("Filings index: %d filings for %d tickers", len(index), len(tickers))

    store = DocumentStore()
    embedder = DocumentEmbedder()

    # Figure out what to embed
    if force:
        to_embed = tickers
    else:
        to_embed = [t for t in tickers if not store.has_embeddings(t)]
    logger.info("To embed: %d tickers (already done: %d)", len(to_embed), len(tickers) - len(to_embed))

    if not to_embed:
        logger.info("All tickers already embedded!")
        return

    t0 = time.time()
    n_docs_total = 0
    n_chunks_total = 0

    for i, ticker in enumerate(to_embed):
        ticker_dir = filings_dir / ticker
        if not ticker_dir.exists():
            continue

        # Read all filing years for this ticker
        year_files = sorted(ticker_dir.glob("*.txt"))
        if not year_files:
            continue

        embeddings_list = []
        dates_list = []
        norms_list = []
        n_chunks_list = []

        for yf in year_files:
            year = int(yf.stem)

            # Skip filings not in the cleaned index
            if (ticker, year) not in date_lookup:
                continue

            raw_text = yf.read_text(errors="ignore")

            # Apply XBRL cleaning
            text = clean_filing_text(raw_text)

            if len(text.split()) < 2000:
                continue

            embedding, norm, n_chunks = embedder.embed_text(text)
            embeddings_list.append(embedding)
            norms_list.append(norm)

            # Use actual SEC filing date, not fiscal year
            filed_date = date_lookup[(ticker, year)]
            dates_list.append(filed_date)

            n_chunks_list.append(n_chunks)
            n_chunks_total += n_chunks

        if not embeddings_list:
            continue

        emb_array = np.stack(embeddings_list)  # (n_docs, 768)
        store.save_embeddings(
            ticker,
            emb_array,
            dates_list,
            metadata={
                "n_chunks": np.array(n_chunks_list),
                "norms": np.array(norms_list),
            },
        )
        n_docs_total += len(embeddings_list)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            logger.info(
                "  [%d/%d] %d docs embedded (%d chunks), %.1f tickers/min",
                i + 1, len(to_embed), n_docs_total, n_chunks_total, rate,
            )

    elapsed = time.time() - t0
    logger.info(
        "Done: %d tickers, %d docs, %d chunks in %.1f min",
        len(to_embed), n_docs_total, n_chunks_total, elapsed / 60,
    )


if __name__ == "__main__":
    main()
