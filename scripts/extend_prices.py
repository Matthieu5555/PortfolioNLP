"""Extend existing price files backward to 2000-01-01 via Tiingo API.

For each data/prices/{TICKER}.parquet:
  1. Read file, get min(index).date()
  2. If min_date > 2000-01-04:
     - fetch_one(ticker, date(2000,1,1), min_date - timedelta(days=1))
     - If Tiingo returns data: concat [new, existing], sort, dedup, save
     - If empty/404: skip (ticker didn't exist before 2005)
  3. If min_date <= 2000-01-04: skip (already has full range)
  4. Log progress and stats

Resumable: tickers whose min_date is already <= 2000-01-04 are skipped.

Usage:
    uv run python scripts/extend_prices.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR
from pnlp.data.prices import PriceFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_START = date(2000, 1, 1)
# Tickers with min_date at or before this are considered already extended
CUTOFF = date(2000, 1, 4)


def main() -> None:
    prices_dir = DATA_DIR / "prices"
    if not prices_dir.exists():
        logger.error("Prices directory does not exist: %s", prices_dir)
        sys.exit(1)

    parquet_files = sorted(prices_dir.glob("*.parquet"))
    logger.info("Found %d price files in %s", len(parquet_files), prices_dir)

    fetcher = PriceFetcher()

    n_extended = 0
    n_skipped_already = 0
    n_no_new_data = 0
    n_failed = 0
    n_api_calls = 0
    n_total = len(parquet_files)

    for i, pf in enumerate(parquet_files):
        ticker = pf.stem

        try:
            existing = pd.read_parquet(pf)
        except Exception as e:
            logger.warning("Failed to read %s: %s", pf, e)
            n_failed += 1
            continue

        if len(existing) == 0:
            n_failed += 1
            continue

        min_date = existing.index.min()
        if hasattr(min_date, "date"):
            min_date = min_date.date()

        if min_date <= CUTOFF:
            n_skipped_already += 1
            continue

        # Fetch data before the existing start
        fetch_end = min_date - timedelta(days=1)
        new_df = fetcher._fetch_one(ticker, TARGET_START, fetch_end)
        n_api_calls += 1

        if new_df is not None and len(new_df) > 0:
            combined = pd.concat([new_df, existing]).sort_index()
            combined = combined[~combined.index.duplicated(keep="first")]
            combined.to_parquet(pf)
            n_extended += 1
            logger.debug(
                "Extended %s: %s -> %s (%d new rows)",
                ticker,
                min_date,
                combined.index.min().date(),
                len(new_df),
            )
        else:
            n_no_new_data += 1

        # Progress logging
        if (i + 1) % 100 == 0:
            logger.info(
                "Progress: %d/%d (extended=%d, already_done=%d, "
                "no_new_data=%d, failed=%d, api_calls=%d)",
                i + 1,
                n_total,
                n_extended,
                n_skipped_already,
                n_no_new_data,
                n_failed,
                n_api_calls,
            )

        # Rate limiting: sleep 120s every 400 API calls
        if n_api_calls > 0 and n_api_calls % 400 == 0:
            logger.info("Pausing 120s for rate limiting (after %d API calls)...", n_api_calls)
            time.sleep(120)

    logger.info(
        "Done! Extended: %d, Already had data: %d, "
        "No pre-2005 data: %d, Failed: %d, API calls: %d, Total files: %d",
        n_extended,
        n_skipped_already,
        n_no_new_data,
        n_failed,
        n_api_calls,
        n_total,
    )


if __name__ == "__main__":
    main()
