"""
Fetch actual SEC filing dates from EDGAR for all CIKs in our filings index.

Uses the SEC EDGAR submissions API:
    https://data.sec.gov/submissions/CIK{cik:010d}.json

Adds a `filed_date` column to filings_index.parquet so we can avoid
look-ahead bias (a 2023 10-K is typically filed in Feb/Mar 2024, NOT Jan 1 2023).

SEC EDGAR rate limit: 10 requests/sec with a proper User-Agent header.

Usage:
    uv run python scripts/fetch_filing_dates.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_dates")

# SEC requires a descriptive User-Agent.
_HEADERS = {
    "User-Agent": "PortfolioNLP Research academic-research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

_BASE_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _fetch_10k_dates(cik_str: str) -> dict[int, str]:
    """Fetch all 10-K filing dates for a CIK.

    Returns
    -------
    Dict mapping fiscal year-end year -> filing date string (YYYY-MM-DD).
    """
    # Pad CIK to 10 digits
    cik_padded = cik_str.zfill(10)
    url = _BASE_URL.format(cik=cik_padded)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code == 429:
            logger.warning("Rate limited. Sleeping 10s...")
            time.sleep(10)
            resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch CIK %s: %s", cik_str, e)
        return {}

    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    result: dict[int, str] = {}

    for form, filed, report_date in zip(forms, filing_dates, report_dates):
        # Match 10-K and 10-K/A (amended)
        if form not in ("10-K", "10-K/A", "10-KSB", "10-KSB/A"):
            continue

        # report_date is the period-of-report end date (e.g. "2023-09-30" for AAPL FY2023)
        # We map this to the fiscal year in our index
        if not report_date:
            continue

        try:
            year = int(report_date[:4])
        except (ValueError, IndexError):
            continue

        # If we already have a 10-K for this year, prefer the original (not amendment)
        if year not in result or form == "10-K":
            result[year] = filed

    # Also check older filings if paginated
    older_files = data.get("filings", {}).get("files", [])
    for older_file_info in older_files:
        older_url = f"https://data.sec.gov/submissions/{older_file_info['name']}"
        try:
            time.sleep(0.1)  # Stay under rate limit
            resp2 = requests.get(older_url, headers=_HEADERS, timeout=15)
            resp2.raise_for_status()
            older = resp2.json()
        except requests.RequestException:
            continue

        forms_old = older.get("form", [])
        dates_old = older.get("filingDate", [])
        reports_old = older.get("reportDate", [])

        for form, filed, report_date in zip(forms_old, dates_old, reports_old):
            if form not in ("10-K", "10-K/A", "10-KSB", "10-KSB/A"):
                continue
            if not report_date:
                continue
            try:
                year = int(report_date[:4])
            except (ValueError, IndexError):
                continue
            if year not in result or form == "10-K":
                result[year] = filed

    return result


def main() -> None:
    index_path = DATA_DIR / "filings_index.parquet"
    if not index_path.exists():
        logger.error("No filings_index.parquet found. Run download_data.py first.")
        return

    index = pd.read_parquet(index_path)
    logger.info("Index: %d rows, %d unique CIKs", len(index), index["cik"].nunique())

    # Check if filed_date column already exists with data
    if "filed_date" in index.columns:
        n_have = index["filed_date"].notna().sum()
        logger.info("Already have %d/%d filing dates", n_have, len(index))
        if n_have == len(index):
            logger.info("All filing dates already fetched!")
            return

    # Build CIK -> list of years we need dates for
    cik_years: dict[str, set[int]] = {}
    for _, row in index.iterrows():
        cik_years.setdefault(row["cik"], set()).add(row["year"])

    # Fetch dates for each CIK
    cik_list = sorted(cik_years.keys())
    logger.info("Fetching filing dates for %d CIKs...", len(cik_list))

    # filed_date lookup: (cik, year) -> date string
    date_lookup: dict[tuple[str, int], str] = {}

    cache_path = DATA_DIR / "filing_dates_cache.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        for key, val in cached.items():
            cik, year_str = key.split("|")
            date_lookup[(cik, int(year_str))] = val
        logger.info("Loaded %d cached dates", len(date_lookup))

    t0 = time.time()
    fetched = 0
    skipped = 0

    for i, cik in enumerate(cik_list):
        # Check if we already have all dates for this CIK
        needed_years = cik_years[cik]
        if all((cik, y) in date_lookup for y in needed_years):
            skipped += 1
            continue

        dates = _fetch_10k_dates(cik)
        for year, filed in dates.items():
            date_lookup[(cik, year)] = filed
        fetched += 1

        # Rate limiting: ~8 requests/sec to stay under SEC limit
        time.sleep(0.12)

        if (fetched) % 200 == 0:
            elapsed = time.time() - t0
            rate = fetched / elapsed
            remaining = (len(cik_list) - skipped - fetched) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] fetched=%d, skipped=%d, rate=%.1f/sec, ~%.0fs remaining",
                i + 1, len(cik_list), fetched, skipped, rate, remaining,
            )
            # Save cache periodically
            cache_data = {f"{cik}|{year}": d for (cik, year), d in date_lookup.items()}
            cache_path.write_text(json.dumps(cache_data))

    # Save final cache
    cache_data = {f"{cik}|{year}": d for (cik, year), d in date_lookup.items()}
    cache_path.write_text(json.dumps(cache_data))
    logger.info("Saved %d dates to cache", len(date_lookup))

    # Merge into index
    index["filed_date"] = index.apply(
        lambda row: date_lookup.get((row["cik"], row["year"])), axis=1
    )

    n_matched = index["filed_date"].notna().sum()
    n_missing = index["filed_date"].isna().sum()
    logger.info("Matched: %d/%d (%.1f%%). Missing: %d", n_matched, len(index), 100 * n_matched / len(index), n_missing)

    # For filings where we couldn't get the exact date, use a conservative estimate:
    # fiscal year end + 75 days (typical 10-K filing deadline is 60 days for large accelerated filers)
    index["filed_date"] = index.apply(
        lambda row: row["filed_date"] if pd.notna(row["filed_date"])
        else f"{row['year'] + 1}-03-15",  # Conservative: March 15 of next year
        axis=1,
    )

    index.to_parquet(index_path)
    elapsed = time.time() - t0
    logger.info(
        "Done: fetched %d CIKs in %.1f min. %d dates matched, %d used conservative estimate.",
        fetched, elapsed / 60, n_matched, n_missing,
    )


if __name__ == "__main__":
    main()
