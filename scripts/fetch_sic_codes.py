"""
Fetch SIC codes from SEC EDGAR submissions API.

Downloads SIC code and description for each ticker in our CIK mapping.
Caches results to data/sic_codes.json. Skips already-cached entries on
re-runs.

SEC EDGAR rate limit: 10 req/sec with proper User-Agent header.

Usage:
    uv run python scripts/fetch_sic_codes.py
    uv run python scripts/fetch_sic_codes.py --max-tickers 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
USER_AGENT = "PortfolioNLP Research ms@research.edu"

SIC_CACHE_PATH = DATA_DIR / "sic_codes.json"


def load_cik_mapping() -> dict[str, dict]:
    path = DATA_DIR / "cik_mapping.json"
    with open(path) as f:
        return json.load(f)


def load_sic_cache() -> dict[str, dict]:
    if SIC_CACHE_PATH.exists():
        with open(SIC_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_sic_cache(cache: dict[str, dict]) -> None:
    with open(SIC_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_sic_for_cik(cik: str, session: requests.Session) -> tuple[str | None, str | None]:
    """Fetch SIC code and description from SEC EDGAR for a single CIK."""
    url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            sic = data.get("sic")
            sic_desc = data.get("sicDescription")
            return sic, sic_desc
        elif resp.status_code == 404:
            return None, None
        else:
            logger.warning("HTTP %d for CIK %s", resp.status_code, cik)
            return None, None
    except Exception as e:
        logger.warning("Error fetching CIK %s: %s", cik, e)
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SIC codes from SEC EDGAR")
    parser.add_argument("--max-tickers", type=int, default=None,
                        help="Max tickers to fetch (for testing)")
    args = parser.parse_args()

    cik_mapping = load_cik_mapping()
    sic_cache = load_sic_cache()

    # Determine which tickers still need SIC codes
    tickers_to_fetch = [t for t in cik_mapping if t not in sic_cache]
    if args.max_tickers:
        tickers_to_fetch = tickers_to_fetch[:args.max_tickers]

    logger.info("CIK mapping: %d tickers. SIC cache: %d. To fetch: %d",
                len(cik_mapping), len(sic_cache), len(tickers_to_fetch))

    if not tickers_to_fetch:
        logger.info("All tickers already cached. Nothing to do.")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    fetched = 0
    errors = 0
    batch_save_interval = 100

    for i, ticker in enumerate(tickers_to_fetch):
        cik = cik_mapping[ticker]["cik"]
        sic, sic_desc = fetch_sic_for_cik(cik, session)

        if sic is not None:
            sic_cache[ticker] = {
                "sic": sic,
                "sic_description": sic_desc,
                "cik": cik,
                "name": cik_mapping[ticker].get("name", ""),
            }
            fetched += 1
        else:
            sic_cache[ticker] = {
                "sic": None,
                "sic_description": None,
                "cik": cik,
                "name": cik_mapping[ticker].get("name", ""),
            }
            errors += 1

        # Rate limit: ~8 req/sec to stay under 10/sec limit
        time.sleep(0.125)

        # Periodic save
        if (i + 1) % batch_save_interval == 0:
            save_sic_cache(sic_cache)
            logger.info("Progress: %d/%d fetched, %d errors, saved cache",
                        i + 1, len(tickers_to_fetch), errors)

    # Final save
    save_sic_cache(sic_cache)
    logger.info("Done. Fetched: %d, Errors: %d, Total cached: %d",
                fetched, errors, len(sic_cache))

    # Print SIC distribution
    sic_counts: dict[str, int] = {}
    for entry in sic_cache.values():
        sic = entry.get("sic")
        if sic:
            sic_2digit = sic[:2]
            sic_counts[sic_2digit] = sic_counts.get(sic_2digit, 0) + 1

    print(f"\nSIC 2-digit distribution ({len(sic_counts)} sectors):")
    for sic, count in sorted(sic_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  SIC {sic}: {count} firms")


if __name__ == "__main__":
    main()
