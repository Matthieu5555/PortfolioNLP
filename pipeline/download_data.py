"""
Download all data for the PortfolioNLP project.

This script:
1. Downloads the CIK mapping from SEC EDGAR
2. Streams PleIAs/SEC 10-K filings and filters to our universe
3. Downloads earnings call transcripts
4. Downloads per-firm stock prices via Tiingo API

Usage:
    uv run python pipeline/download_data.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download")


def step_1_cik_mapping() -> dict[str, dict]:
    """Download CIK mapping from SEC EDGAR."""
    from pnlp.data.universe import UniverseLoader

    logger.info("=== Step 1: CIK Mapping ===")
    loader = UniverseLoader()
    mapping = loader.load_cik_mapping()
    logger.info("CIK mapping: %d tickers", len(mapping))
    return mapping


def step_2_filings(cik_mapping: dict[str, dict]) -> None:
    """Stream PleIAs/SEC and save filings for our universe."""
    from datasets import load_dataset

    logger.info("=== Step 2: 10-K Filings (PleIAs/SEC) ===")

    # Build CIK lookup (strip leading zeros for matching)
    cik_to_ticker: dict[str, str] = {}
    for ticker, info in cik_mapping.items():
        cik_stripped = info["cik"].lstrip("0")
        cik_to_ticker[cik_stripped] = ticker

    logger.info("Looking for %d CIKs in PleIAs/SEC dataset...", len(cik_to_ticker))

    # Stream the dataset to avoid downloading everything
    ds = load_dataset("PleIAs/SEC", split="train", streaming=True)

    filings_dir = DATA_DIR / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)

    n_matched = 0
    n_scanned = 0
    records = []

    for row in ds:
        n_scanned += 1
        row_cik = str(row.get("cik", "")).lstrip("0")

        if row_cik not in cik_to_ticker:
            continue

        year_str = str(row.get("year", ""))
        if not year_str.isdigit():
            continue
        year = int(year_str)

        # Only keep 2000+ for tractability
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

        # Save text to per-ticker directory
        ticker_dir = filings_dir / ticker
        ticker_dir.mkdir(exist_ok=True)
        text_path = ticker_dir / f"{year}.txt"
        text_path.write_text(text[:500_000])

        if n_matched % 200 == 0:
            n_tickers = len({r["ticker"] for r in records})
            logger.info(
                "  %d filings matched (%d tickers), scanned %d rows",
                n_matched, n_tickers, n_scanned,
            )

    # Save index
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(DATA_DIR / "filings_index.parquet")
        n_tickers = df["ticker"].nunique()
        logger.info(
            "Saved %d filings for %d tickers (scanned %d total rows)",
            len(df), n_tickers, n_scanned,
        )
    else:
        logger.warning("No filings matched!")


def step_3_transcripts(cik_mapping: dict[str, dict]) -> None:
    """Download earnings call transcripts."""
    from datasets import load_dataset

    logger.info("=== Step 3: Earnings Call Transcripts ===")

    ticker_set = set(cik_mapping.keys())

    ds = load_dataset("kurry/sp500_earnings_transcripts", split="train")
    logger.info("Dataset loaded: %d rows", len(ds))

    transcripts_dir = DATA_DIR / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in ds:
        ticker = str(row.get("symbol", "")).upper()
        if ticker not in ticker_set:
            continue

        year = row.get("year")
        quarter = row.get("quarter")
        date_str = row.get("date", "")
        content = row.get("content", "")

        if not content or len(content) < 100:
            continue

        records.append({
            "ticker": ticker,
            "year": int(year) if year else 0,
            "quarter": int(quarter) if quarter else 0,
            "date": date_str[:10] if date_str else "",
            "char_count": len(content),
        })

        # Save transcript
        ticker_dir = transcripts_dir / ticker
        ticker_dir.mkdir(exist_ok=True)
        fname = f"Q{quarter}_{year}.txt" if quarter and year else f"{len(records)}.txt"
        (ticker_dir / fname).write_text(content[:500_000])

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(DATA_DIR / "transcripts_index.parquet")
        n_tickers = df["ticker"].nunique()
        logger.info("Saved %d transcripts for %d tickers", len(df), n_tickers)
    else:
        logger.warning("No transcripts matched!")


def step_4_prices(cik_mapping: dict[str, dict]) -> None:
    """Download stock prices for the universe via Tiingo API."""
    from pnlp.data.prices import PriceFetcher

    logger.info("=== Step 4: Stock Prices (Tiingo) ===")

    all_tickers = sorted(cik_mapping.keys())
    prices_dir = DATA_DIR / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    # Skip tickers we already have
    existing = {p.stem for p in prices_dir.glob("*.parquet")}
    to_download = [t for t in all_tickers if t not in existing]
    logger.info(
        "Total: %d tickers, already have %d, need %d",
        len(all_tickers), len(existing), len(to_download),
    )

    if not to_download:
        logger.info("All prices already downloaded!")
        return

    fetcher = PriceFetcher()
    start = date(2005, 1, 1)
    end = date(2025, 12, 31)

    n_fetched = 0
    n_failed = 0

    for i, ticker in enumerate(to_download):
        df = fetcher._fetch_one(ticker, start, end)
        if df is not None and len(df) > 100:
            df.to_parquet(prices_dir / f"{ticker}.parquet")
            n_fetched += 1
        else:
            n_failed += 1

        if (i + 1) % 50 == 0:
            logger.info(
                "  Progress: %d/%d (fetched=%d, skipped=%d)",
                i + 1, len(to_download), n_fetched, n_failed,
            )

        # Tiingo free tier: 500 requests/hour = ~8/min
        # Stay comfortably under with a small sleep
        if (i + 1) % 400 == 0:
            logger.info("  Pausing 120s for rate limiting...")
            time.sleep(120)

    logger.info(
        "Total: %d new tickers fetched (%d total with price data)",
        n_fetched, len(existing) + n_fetched,
    )


if __name__ == "__main__":
    logger.info("Starting PortfolioNLP data download")
    logger.info("Data directory: %s", DATA_DIR)

    cik_mapping = step_1_cik_mapping()

    # Run steps based on command-line args
    args = set(sys.argv[1:]) if len(sys.argv) > 1 else {"all"}

    if "all" in args or "filings" in args:
        step_2_filings(cik_mapping)

    if "all" in args or "transcripts" in args:
        step_3_transcripts(cik_mapping)

    if "all" in args or "prices" in args:
        step_4_prices(cik_mapping)

    logger.info("Done!")
