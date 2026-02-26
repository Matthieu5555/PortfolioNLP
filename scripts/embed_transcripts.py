"""
Embed all downloaded earnings call transcripts.

Reads transcript text from data/transcripts/{ticker}/Q{q}_{year}.txt,
embeds each document (chunking + mean-pooling), and saves per-firm
embedding arrays to data/embeddings_transcripts/{ticker}.npz.

Uses transcript dates from transcripts_index.parquet to prevent
look-ahead bias.

Resumes where it left off — skips tickers that already have embeddings.

Usage:
    uv run python scripts/embed_transcripts.py
    uv run python scripts/embed_transcripts.py --force   # re-embed everything
"""

from __future__ import annotations

import logging
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR
from pnlp.data.store import DocumentStore, TRANSCRIPT_EMBEDDINGS_DIR
from pnlp.embeddings.document_embedder import DocumentEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embed_transcripts")


def parse_transcript_filename(fname: str) -> tuple[int, int] | None:
    """Parse Q{quarter}_{year}.txt → (quarter, year) or None."""
    m = re.match(r"Q(\d+)_(\d{4})\.txt", fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def main() -> None:
    force = "--force" in sys.argv

    transcripts_dir = DATA_DIR / "transcripts"
    index_path = DATA_DIR / "transcripts_index.parquet"

    if not transcripts_dir.exists():
        logger.error("No transcripts directory found. Run download_data.py first.")
        return

    # Build date lookup from index
    date_lookup: dict[tuple[str, int, int], date] = {}
    if index_path.exists():
        index = pd.read_parquet(index_path)
        for _, row in index.iterrows():
            ticker = str(row["ticker"])
            year = int(row.get("year", 0))
            quarter = int(row.get("quarter", 0))
            date_str = str(row.get("date", ""))
            if date_str and len(date_str) >= 10:
                try:
                    d = date.fromisoformat(date_str[:10])
                    date_lookup[(ticker, quarter, year)] = d
                except (ValueError, TypeError):
                    pass
        logger.info("Loaded %d transcript dates from index", len(date_lookup))
    else:
        logger.warning("No transcripts_index.parquet found, will estimate dates from filenames")

    # Find all tickers with transcript directories
    ticker_dirs = sorted(
        d for d in transcripts_dir.iterdir()
        if d.is_dir() and any(d.glob("*.txt"))
    )
    logger.info("Found %d tickers with transcripts", len(ticker_dirs))

    # Set up store pointing to transcript embeddings directory
    store = DocumentStore(embeddings_dir=TRANSCRIPT_EMBEDDINGS_DIR)
    embedder = DocumentEmbedder()

    # Figure out what to embed
    if force:
        to_embed = ticker_dirs
    else:
        to_embed = [d for d in ticker_dirs if not store.has_embeddings(d.name)]
    logger.info(
        "To embed: %d tickers (already done: %d)",
        len(to_embed), len(ticker_dirs) - len(to_embed),
    )

    if not to_embed:
        logger.info("All tickers already embedded!")
        return

    t0 = time.time()
    n_docs_total = 0
    n_chunks_total = 0

    for i, ticker_dir in enumerate(to_embed):
        ticker = ticker_dir.name
        txt_files = sorted(ticker_dir.glob("*.txt"))
        if not txt_files:
            continue

        embeddings_list = []
        dates_list = []
        norms_list = []
        n_chunks_list = []

        for txt_file in txt_files:
            parsed = parse_transcript_filename(txt_file.name)
            if parsed is None:
                continue
            quarter, year = parsed

            # Get transcript date from index, or estimate from quarter
            transcript_date = date_lookup.get((ticker, quarter, year))
            if transcript_date is None:
                # Estimate: end of quarter month
                quarter_end_month = quarter * 3
                if quarter_end_month > 12:
                    quarter_end_month = 12
                transcript_date = date(year, quarter_end_month, 28)

            text = txt_file.read_text(errors="ignore")

            # Skip very short transcripts
            if len(text.split()) < 500:
                continue

            embedding, norm, n_chunks = embedder.embed_text(text)
            embeddings_list.append(embedding)
            norms_list.append(norm)
            dates_list.append(transcript_date)
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
