"""
Load earnings call transcripts from HuggingFace kurry/sp500_earnings_transcripts.

Provides quarterly forward-looking management commentary to complement
annual 10-K filings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from pnlp.data.universe import Firm

logger = logging.getLogger(__name__)

__all__ = ["Transcript", "TranscriptsLoader"]


@dataclass(frozen=True)
class Transcript:
    """A single earnings call transcript."""

    ticker: str
    date: date
    quarter: str  # e.g., "Q1 2024"
    text: str


class TranscriptsLoader:
    """Load earnings call transcripts from HuggingFace."""

    def __init__(self, dataset_name: str = "kurry/sp500_earnings_transcripts") -> None:
        self.dataset_name = dataset_name
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is None:
            from datasets import load_dataset

            logger.info("Loading %s from HuggingFace...", self.dataset_name)
            self._dataset = load_dataset(self.dataset_name, split="train")
        return self._dataset

    def load_transcripts_for_firms(
        self,
        firms: list[Firm],
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, list[Transcript]]:
        """Load transcripts for firms, keyed by ticker.

        kurry/sp500_earnings_transcripts schema:
            symbol, quarter, year, date, content, structured_content,
            company_name, company_id

        Returns dict mapping ticker -> list of Transcript, sorted by date.
        """
        ticker_set = {f.ticker.upper() for f in firms}
        result: dict[str, list[Transcript]] = {f.ticker: [] for f in firms}

        dataset = self._load_dataset()

        for row in dataset:
            ticker = str(row.get("symbol", "")).upper()
            if ticker not in ticker_set:
                continue

            # Year filtering
            year = row.get("year")
            if year is not None:
                if start_year and int(year) < start_year:
                    continue
                if end_year and int(year) > end_year:
                    continue

            # Parse date
            date_str = row.get("date", "")
            if not date_str:
                continue

            try:
                if isinstance(date_str, str):
                    transcript_date = date.fromisoformat(date_str[:10])
                else:
                    transcript_date = date_str
            except (ValueError, TypeError):
                continue

            # Content field (full transcript text)
            text = row.get("content", "")
            if not text or len(text) < 100:
                continue

            quarter_num = row.get("quarter", "")
            year_val = row.get("year", "")
            quarter_str = f"Q{quarter_num} {year_val}" if quarter_num and year_val else ""

            t = Transcript(
                ticker=ticker,
                date=transcript_date,
                quarter=quarter_str,
                text=text[:500_000],
            )
            result[ticker].append(t)

        for ticker in result:
            result[ticker].sort(key=lambda t: t.date)

        total = sum(len(v) for v in result.values())
        logger.info("Loaded %d transcripts for %d firms", total, sum(1 for v in result.values() if v))
        return result
