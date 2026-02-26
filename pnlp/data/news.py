"""
Firm-level news from HuggingFace financial-news-multisource.

Filters the 57M-article corpus to firm-specific news by ticker matching.
Can also load from the MCF project's cached articles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from pnlp.data.universe import Firm

logger = logging.getLogger(__name__)

__all__ = ["NewsArticle", "NewsLoader"]


@dataclass(frozen=True)
class NewsArticle:
    """A firm-level news article."""

    ticker: str
    title: str
    description: str
    published_date: date
    source: str = ""


class NewsLoader:
    """Load firm-level news articles from HuggingFace or local cache."""

    def __init__(
        self,
        dataset_name: str = "Brianferrell787/financial-news-multisource",
        subsets: tuple[str, ...] = (
            "nyt_articles_2000_present",
            "headlines_10sites_2007_2022",
            "finsen_us_2007_2023",
            "cnbc_headlines",
            "yahoo_finance_felixdrinkall",
            "sp500_daily_headlines",
            "reuters_bloomberg",
        ),
    ) -> None:
        self.dataset_name = dataset_name
        self.subsets = subsets

    def load_news_for_firms(
        self,
        firms: list[Firm],
        start: date | None = None,
        end: date | None = None,
        max_per_firm: int = 500,
    ) -> dict[str, list[NewsArticle]]:
        """Load news articles mentioning firms in our universe.

        Matches articles to firms by ticker mention in title/description
        or via the tickers field if available in the dataset.

        Returns dict mapping ticker -> list of NewsArticle, sorted by date.
        """
        from datasets import load_dataset

        ticker_set = {f.ticker.upper() for f in firms}
        result: dict[str, list[NewsArticle]] = {f.ticker: [] for f in firms}

        for subset in self.subsets:
            logger.info("Loading news subset: %s", subset)
            try:
                ds = load_dataset(self.dataset_name, subset, split="train", streaming=True)
            except Exception:
                logger.warning("Failed to load subset %s, skipping", subset)
                continue

            for row in ds:
                title = row.get("title", row.get("headline", ""))
                if not title or len(title) < 10:
                    continue

                # Try to match by tickers field first
                row_tickers = row.get("tickers", [])
                if isinstance(row_tickers, str):
                    row_tickers = [t.strip() for t in row_tickers.split(",")]

                matched_tickers = set()
                for t in row_tickers:
                    t_upper = t.upper().strip("$")
                    if t_upper in ticker_set:
                        matched_tickers.add(t_upper)

                # Fallback: check title for ticker mentions (less reliable)
                if not matched_tickers:
                    title_upper = title.upper()
                    for ticker in ticker_set:
                        if len(ticker) >= 3 and f" {ticker} " in f" {title_upper} ":
                            matched_tickers.add(ticker)
                            break  # One match per article is enough

                if not matched_tickers:
                    continue

                date_str = row.get("published_date", row.get("date", row.get("publishedAt", "")))
                if not date_str:
                    continue

                try:
                    if isinstance(date_str, str):
                        pub_date = date.fromisoformat(date_str[:10])
                    else:
                        pub_date = date_str
                except (ValueError, TypeError):
                    continue

                if start and pub_date < start:
                    continue
                if end and pub_date > end:
                    continue

                description = row.get("description", row.get("text", ""))
                source = row.get("source", subset)

                for ticker in matched_tickers:
                    if len(result[ticker]) < max_per_firm:
                        result[ticker].append(
                            NewsArticle(
                                ticker=ticker,
                                title=title,
                                description=str(description)[:5000] if description else "",
                                published_date=pub_date,
                                source=str(source),
                            )
                        )

        for ticker in result:
            result[ticker].sort(key=lambda a: a.published_date)

        total = sum(len(v) for v in result.values())
        logger.info("Loaded %d news articles for %d firms", total, sum(1 for v in result.values() if v))
        return result
