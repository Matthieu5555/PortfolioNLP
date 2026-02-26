"""
Load pre-parsed 10-K filings from HuggingFace PleIAs/SEC dataset.

This avoids building a custom EDGAR parser — the dataset already has
30+ years of 10-K filings sectioned and cleaned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from pnlp.data.universe import Firm

logger = logging.getLogger(__name__)

__all__ = ["FilingDocument", "FilingsLoader"]


@dataclass(frozen=True)
class FilingDocument:
    """A single SEC filing with extracted text."""

    ticker: str
    cik: str
    filing_date: date
    filing_type: str  # "10-K", "10-Q", etc.
    text: str  # Full filing text or specific section
    section: str  # "full", "item_1", "item_1a", "item_7", etc.


class FilingsLoader:
    """Load 10-K filings from HuggingFace PleIAs/SEC dataset.

    The dataset contains pre-parsed SEC filings from 1993-2024 for all
    public US companies. We filter by CIK to get filings for our universe.
    """

    def __init__(self, dataset_name: str = "PleIAs/SEC") -> None:
        self.dataset_name = dataset_name
        self._dataset = None

    def _load_dataset(self):
        """Lazy-load the HuggingFace dataset."""
        if self._dataset is None:
            from datasets import load_dataset

            logger.info("Loading %s from HuggingFace (this may take a while)...", self.dataset_name)
            self._dataset = load_dataset(self.dataset_name, split="train", streaming=True)
        return self._dataset

    def load_filings_for_firms(
        self,
        firms: list[Firm],
        start_year: int | None = None,
        end_year: int | None = None,
        max_per_firm: int | None = None,
    ) -> dict[str, list[FilingDocument]]:
        """Load filings for a set of firms, keyed by ticker.

        PleIAs/SEC schema:
            filename, id, year, cik, text, word_count, character_count

        The dataset contains 10-K filings (Item 1 Business Description)
        indexed by CIK and year. We match by stripping leading zeros
        from both sides.

        Args:
            firms: list of Firm objects with CIK identifiers.
            start_year: earliest filing year to include.
            end_year: latest filing year to include.
            max_per_firm: cap on filings per firm (most recent first).

        Returns:
            dict mapping ticker -> list of FilingDocument, sorted by date.
        """
        cik_to_ticker = {f.cik.lstrip("0"): f.ticker for f in firms}
        cik_set = set(cik_to_ticker.keys())

        result: dict[str, list[FilingDocument]] = {f.ticker: [] for f in firms}
        dataset = self._load_dataset()

        n_matched = 0
        n_scanned = 0
        for row in dataset:
            n_scanned += 1
            row_cik = str(row.get("cik", "")).lstrip("0")
            if row_cik not in cik_set:
                continue

            # Year field (string in PleIAs/SEC)
            year_str = str(row.get("year", ""))
            if not year_str.isdigit():
                continue
            year = int(year_str)

            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue

            text = row.get("text", "")
            if not text or len(text) < 100:
                continue

            # Use Jan 1 of the filing year as approximate date
            filing_date = date(year, 1, 1)

            ticker = cik_to_ticker[row_cik]
            doc = FilingDocument(
                ticker=ticker,
                cik=row_cik,
                filing_date=filing_date,
                filing_type="10-K",
                text=text[:500_000],  # Cap at 500K chars
                section="full",
            )
            result[ticker].append(doc)
            n_matched += 1

            if n_matched % 500 == 0:
                logger.info("Matched %d filings (scanned %d rows)...", n_matched, n_scanned)

        # Sort by date and optionally cap
        for ticker in result:
            result[ticker].sort(key=lambda d: d.filing_date)
            if max_per_firm:
                result[ticker] = result[ticker][-max_per_firm:]

        total = sum(len(v) for v in result.values())
        firms_with_data = sum(1 for v in result.values() if v)
        logger.info(
            "Loaded %d filings for %d/%d firms (scanned %d total rows)",
            total,
            firms_with_data,
            len(firms),
            n_scanned,
        )
        return result
