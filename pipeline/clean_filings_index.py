"""
Clean and deduplicate the filings index.

1. Deduplicate on (ticker, year, cik), keeping row with max word_count
2. Drop filings with word_count < 2000 (shell companies, fragments)
3. Drop tickers with no price data (warrants, preferred shares)
4. Update word_count and char_count after XBRL stripping
5. Save cleaned index as filings_index.parquet (overwrite)

Usage:
    uv run python pipeline/clean_filings_index.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR
from pnlp.data.filing_cleaner import clean_filing_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_index")


def main() -> None:
    index_path = DATA_DIR / "filings_index.parquet"
    filings_dir = DATA_DIR / "filings"
    prices_dir = DATA_DIR / "prices"

    if not index_path.exists():
        logger.error("No filings_index.parquet. Run download_data.py first.")
        return

    index = pd.read_parquet(index_path)
    logger.info("Original index: %d rows, %d tickers", len(index), index["ticker"].nunique())

    # ── Step 1: Deduplicate on (ticker, year) ────────────────────────
    before = len(index)
    index = index.sort_values("word_count", ascending=False).drop_duplicates(
        subset=["ticker", "year"], keep="first"
    )
    logger.info("Dedup (ticker, year): %d -> %d (removed %d)", before, len(index), before - len(index))

    # ── Step 2: Update word/char counts after XBRL cleaning ─────────
    logger.info("Updating word/char counts after XBRL cleaning...")
    cleaned_word_counts = []
    cleaned_char_counts = []
    for _, row in index.iterrows():
        fpath = filings_dir / row["ticker"] / f"{row['year']}.txt"
        if fpath.exists():
            raw = fpath.read_text(errors="ignore")
            cleaned = clean_filing_text(raw)
            cleaned_word_counts.append(len(cleaned.split()))
            cleaned_char_counts.append(len(cleaned))
        else:
            cleaned_word_counts.append(0)
            cleaned_char_counts.append(0)

    index["word_count_clean"] = cleaned_word_counts
    index["char_count_clean"] = cleaned_char_counts

    # ── Step 3: Drop short filings (< 2000 words after cleaning) ─────
    before = len(index)
    index = index[index["word_count_clean"] >= 2000].copy()
    logger.info("Drop short filings: %d -> %d (removed %d)", before, len(index), before - len(index))

    # ── Step 4: Drop tickers with no price data ──────────────────────
    if prices_dir.exists():
        price_tickers = {f.stem for f in prices_dir.glob("*.parquet")}
        before = len(index)
        before_tickers = index["ticker"].nunique()
        index = index[index["ticker"].isin(price_tickers)].copy()
        logger.info(
            "Drop no-price tickers: %d -> %d rows (%d -> %d tickers)",
            before, len(index), before_tickers, index["ticker"].nunique(),
        )
    else:
        logger.warning("No prices directory found. Skipping price filter.")

    # ── Step 5: Drop filings with no actual file ─────────────────────
    before = len(index)
    has_file = index.apply(
        lambda r: (filings_dir / r["ticker"] / f"{r['year']}.txt").exists(), axis=1
    )
    index = index[has_file].copy()
    if before != len(index):
        logger.info("Drop missing files: %d -> %d", before, len(index))

    # ── Save ─────────────────────────────────────────────────────────
    index = index.sort_values(["ticker", "year"]).reset_index(drop=True)
    index.to_parquet(index_path)
    logger.info(
        "Final index: %d rows, %d tickers, years %d-%d",
        len(index),
        index["ticker"].nunique(),
        index["year"].min(),
        index["year"].max(),
    )


if __name__ == "__main__":
    main()
