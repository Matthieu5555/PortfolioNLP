"""
Download 10-K filings for tickers that have transcript embeddings but no
10-K embeddings.

Streams the full PleIAs/SEC dataset (required since it doesn't support
CIK-based filtering), but only saves filings matching the target CIKs.

After running this, run:
    uv run python scripts/fetch_filing_dates.py   # get actual SEC filing dates
    uv run python scripts/embed_filings.py         # embed (auto-skips existing)

Usage:
    uv run python scripts/download_missing_filings.py
    uv run python scripts/download_missing_filings.py --dry-run  # just list tickers
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_missing")


def find_missing_tickers() -> list[str]:
    """Find tickers with transcript embeddings but no 10-K embeddings."""
    emb_10k = DATA_DIR / "embeddings"
    emb_transcript = DATA_DIR / "embeddings_transcripts"

    ten_k_tickers = {f.stem for f in emb_10k.glob("*.npz")} if emb_10k.exists() else set()
    transcript_tickers = {f.stem for f in emb_transcript.glob("*.npz")} if emb_transcript.exists() else set()

    missing = sorted(transcript_tickers - ten_k_tickers)
    return missing


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    # 1. Find missing tickers
    missing = find_missing_tickers()
    logger.info("Found %d tickers with transcripts but no 10-K embeddings", len(missing))

    if not missing:
        logger.info("No missing tickers!")
        return

    logger.info("Missing tickers: %s", ", ".join(missing[:20]))
    if len(missing) > 20:
        logger.info("  ... and %d more", len(missing) - 20)

    # 2. Build CIK lookup from cik_mapping.json
    cik_path = DATA_DIR / "cik_mapping.json"
    if not cik_path.exists():
        logger.error("No cik_mapping.json found. Run download_data.py first.")
        return

    with open(cik_path) as f:
        cik_mapping = json.load(f)

    # Map CIK (stripped) -> ticker for just the missing tickers
    cik_to_ticker: dict[str, str] = {}
    missing_no_cik = []
    for ticker in missing:
        info = cik_mapping.get(ticker)
        if info:
            cik_stripped = info["cik"].lstrip("0")
            cik_to_ticker[cik_stripped] = ticker
        else:
            missing_no_cik.append(ticker)

    if missing_no_cik:
        logger.warning(
            "%d tickers have no CIK mapping: %s",
            len(missing_no_cik), ", ".join(missing_no_cik),
        )

    logger.info("Targeting %d CIKs for download", len(cik_to_ticker))

    if dry_run:
        logger.info("DRY RUN — would download for: %s", ", ".join(sorted(cik_to_ticker.values())))
        return

    # 3. Stream PleIAs/SEC and save matching filings
    from datasets import load_dataset

    logger.info("Streaming PleIAs/SEC dataset (this will take a while)...")
    ds = load_dataset("PleIAs/SEC", split="train", streaming=True)

    filings_dir = DATA_DIR / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)

    n_matched = 0
    n_scanned = 0
    records = []

    t0 = time.time()
    for row in ds:
        n_scanned += 1
        row_cik = str(row.get("cik", "")).lstrip("0")

        if row_cik not in cik_to_ticker:
            if n_scanned % 500_000 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  Scanned %dK rows (%.1f min), %d filings matched for %d tickers",
                    n_scanned // 1000, elapsed / 60, n_matched,
                    len({r["ticker"] for r in records}),
                )
            continue

        year_str = str(row.get("year", ""))
        if not year_str.isdigit():
            continue
        year = int(year_str)

        if year < 2000:
            continue

        text = row.get("text", "")
        if not text or len(text) < 500:
            continue

        ticker = cik_to_ticker[row_cik]
        n_matched += 1

        records.append({
            "ticker": ticker,
            "cik": row_cik,
            "year": year,
            "word_count": row.get("word_count", 0),
            "char_count": len(text),
        })

        # Save text
        ticker_dir = filings_dir / ticker
        ticker_dir.mkdir(exist_ok=True)
        text_path = ticker_dir / f"{year}.txt"
        text_path.write_text(text[:500_000])

        if n_matched % 50 == 0:
            n_tickers = len({r["ticker"] for r in records})
            logger.info(
                "  %d filings matched (%d/%d tickers), scanned %dK rows",
                n_matched, n_tickers, len(cik_to_ticker), n_scanned // 1000,
            )

    elapsed = time.time() - t0
    logger.info(
        "Streaming complete: scanned %d rows in %.1f min",
        n_scanned, elapsed / 60,
    )

    # 4. Update filings_index.parquet
    if records:
        new_df = pd.DataFrame(records)
        n_tickers = new_df["ticker"].nunique()
        logger.info("Downloaded %d filings for %d tickers", len(new_df), n_tickers)

        # Merge with existing index
        index_path = DATA_DIR / "filings_index.parquet"
        if index_path.exists():
            existing = pd.read_parquet(index_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            # Drop duplicates (same ticker+year)
            combined = combined.drop_duplicates(subset=["ticker", "year"], keep="last")
            combined.to_parquet(index_path)
            logger.info(
                "Updated filings_index.parquet: %d total filings (%d new)",
                len(combined), len(combined) - len(existing),
            )
        else:
            new_df.to_parquet(index_path)
            logger.info("Created filings_index.parquet with %d filings", len(new_df))

        logger.info(
            "Next steps:\n"
            "  1. uv run python scripts/fetch_filing_dates.py  (get SEC filing dates)\n"
            "  2. uv run python scripts/embed_filings.py        (embed new tickers)"
        )
    else:
        not_found = set(cik_to_ticker.values())
        logger.warning(
            "No filings matched! %d tickers not found in PleIAs/SEC: %s",
            len(not_found), ", ".join(sorted(not_found)[:20]),
        )


if __name__ == "__main__":
    main()
