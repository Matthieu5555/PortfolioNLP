"""
Embed firm-level news articles from HuggingFace financial-news-multisource.

For each (ticker, quarter), concatenates all matched articles into one
document, embeds it via DocumentEmbedder, and saves per-firm embedding
arrays to data/embeddings_news/{ticker}.npz.

Targets financial-specific subsets that carry structured ticker tags in
their extra_fields JSON. Uses data_files to stream only relevant parquet
shards rather than the full 57M-article corpus.

Usage:
    uv run python scripts/embed_news.py
    uv run python scripts/embed_news.py --force   # re-embed everything
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR
from pnlp.data.store import DocumentStore, NEWS_EMBEDDINGS_DIR
from pnlp.embeddings.document_embedder import DocumentEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embed_news")

DATASET_NAME = "Brianferrell787/financial-news-multisource"

# Financial subsets with structured `stocks` tags in extra_fields.
# Ordered roughly by coverage breadth.
TAGGED_SUBSETS = (
    "benzinga_6000stocks",       # 2009+, 6000 stocks, best coverage
    "sentarl_combined",          # 1997+, sentiment data
    "fnspid_news",               # 2020+, financial news
    "yahoo_finance_felixdrinkall",  # 2017+
    "yahoo_finance_articles",    # 2025+
)

# Subsets with no ticker tags — use title-mention matching only.
UNTAGGED_SUBSETS = (
    "sp500_daily_headlines",     # S&P 500 focused headlines
)

# Index/non-equity tickers to skip
INDEX_TICKERS = {"$INDU", "$SPX", "$COMP", "$DJI", "$NDX", "$VIX", "INDU", "SPX", "COMP", "DJI", "NDX", "VIX"}

# Date bounds for valid articles
MIN_DATE = date(2000, 1, 1)
MAX_DATE = date(2025, 12, 31)


def quarter_start(d: date) -> date:
    """Return the first day of the quarter containing date d."""
    q = (d.month - 1) // 3
    return date(d.year, q * 3 + 1, 1)


def quarter_end(d: date) -> date:
    """Return the last day of the quarter containing date d."""
    q = (d.month - 1) // 3
    end_month = (q + 1) * 3
    if end_month == 12:
        return date(d.year, 12, 31)
    next_q_start = date(d.year + (1 if end_month >= 12 else 0),
                        (end_month % 12) + 1, 1)
    return next_q_start - timedelta(days=1)


def dedup_key(text_start: str, pub_date: date) -> str:
    """Hash (text[:80], date) for deduplication across subsets."""
    raw = f"{text_start[:80].lower()}|{pub_date.isoformat()}"
    return hashlib.md5(raw.encode()).hexdigest()


def parse_date_field(date_str: str) -> date | None:
    """Parse ISO date string (possibly with timezone) to date."""
    if not date_str:
        return None
    try:
        d = date.fromisoformat(date_str[:10])
        if MIN_DATE <= d <= MAX_DATE:
            return d
        return None
    except (ValueError, TypeError):
        return None


def extract_tickers_from_extra(extra_fields: str, ticker_set: set[str]) -> set[str]:
    """Extract ticker matches from extra_fields JSON `stocks` list."""
    if not extra_fields:
        return set()
    try:
        ef = json.loads(extra_fields)
    except (json.JSONDecodeError, TypeError):
        return set()

    stocks = ef.get("stocks", [])
    if not stocks or not isinstance(stocks, list):
        return set()

    matched = set()
    for s in stocks:
        if not isinstance(s, str):
            continue
        cleaned = s.upper().strip("$").strip()
        if cleaned in INDEX_TICKERS:
            continue
        # Skip entries that look like company names (len > 6, contains space)
        if len(cleaned) > 6 or " " in cleaned:
            continue
        if cleaned in ticker_set:
            matched.add(cleaned)
    return matched


def extract_tickers_from_title(text: str, ticker_set: set[str]) -> set[str]:
    """Fallback: match tickers by word-boundary in first line of text."""
    # Title is the first line (before first \n\n)
    title = text.split("\n\n")[0] if "\n\n" in text else text[:200]
    title_upper = title.upper()

    for ticker in ticker_set:
        if len(ticker) >= 3 and f" {ticker} " in f" {title_upper} ":
            return {ticker}  # one match per article
    return set()


def main() -> None:
    force = "--force" in sys.argv

    # Build ticker universe from existing 10-K + transcript embeddings
    emb_10k_dir = DATA_DIR / "embeddings"
    emb_transcript_dir = DATA_DIR / "embeddings_transcripts"
    ticker_set: set[str] = set()

    for d in (emb_10k_dir, emb_transcript_dir):
        if d.exists():
            for f in d.glob("*.npz"):
                ticker_set.add(f.stem)

    logger.info("Ticker universe: %d tickers from 10-K + transcript embeddings", len(ticker_set))

    if not ticker_set:
        logger.error("No existing embeddings found. Run embed_filings.py first.")
        return

    # Phase 1: Stream financial subsets, accumulate text per (ticker, quarter)
    from datasets import load_dataset

    docs: dict[tuple[str, date], list[str]] = {}  # (ticker, q_start) -> [article_texts]
    seen: set[str] = set()  # dedup keys
    n_articles = 0
    n_matched = 0
    n_dupes = 0

    all_subsets = [(s, True) for s in TAGGED_SUBSETS] + [(s, False) for s in UNTAGGED_SUBSETS]

    for subset, has_tags in all_subsets:
        logger.info("Streaming subset: %s (tagged=%s)", subset, has_tags)
        try:
            ds = load_dataset(
                DATASET_NAME,
                data_files=f"data/{subset}/*.parquet",
                split="train",
                streaming=True,
            )
        except Exception as e:
            logger.warning("Failed to load subset %s: %s, skipping", subset, e)
            continue

        subset_matched = 0
        subset_total = 0
        for row in ds:
            subset_total += 1
            n_articles += 1

            text = row.get("text", "")
            if not text or len(text) < 20:
                continue

            pub_date = parse_date_field(row.get("date", ""))
            if pub_date is None:
                continue

            # Dedup
            dk = dedup_key(text, pub_date)
            if dk in seen:
                n_dupes += 1
                continue
            seen.add(dk)

            # Match to tickers
            if has_tags:
                matched = extract_tickers_from_extra(row.get("extra_fields", ""), ticker_set)
            else:
                matched = set()

            # Fallback: title-mention matching (for untagged or if no tag match)
            if not matched:
                matched = extract_tickers_from_title(text, ticker_set)

            if not matched:
                continue

            # Truncate article text for quarterly document
            article_text = text[:5000] + "\n\n"

            q_start = quarter_start(pub_date)
            for ticker in matched:
                key = (ticker, q_start)
                if key not in docs:
                    docs[key] = []
                docs[key].append(article_text)
                subset_matched += 1

            n_matched += 1

            if n_articles % 500_000 == 0:
                logger.info(
                    "  Streamed %dK articles, %d matched, %d dupes, %d (ticker,quarter) pairs",
                    n_articles // 1000, n_matched, n_dupes, len(docs),
                )

        logger.info(
            "  Subset %s: %d rows, %d article-ticker matches",
            subset, subset_total, subset_matched,
        )

    logger.info(
        "Phase 1 done: %d articles streamed, %d matched, %d dupes, %d (ticker,quarter) pairs",
        n_articles, n_matched, n_dupes, len(docs),
    )

    if not docs:
        logger.warning("No articles matched any tickers!")
        return

    # Phase 2: Embed each (ticker, quarter) document
    store = DocumentStore(embeddings_dir=NEWS_EMBEDDINGS_DIR)
    embedder = DocumentEmbedder()

    # Group by ticker
    ticker_docs: dict[str, list[tuple[date, str]]] = {}
    for (ticker, q_start), articles in docs.items():
        combined = "".join(articles)
        if ticker not in ticker_docs:
            ticker_docs[ticker] = []
        q_end = quarter_end(q_start)
        ticker_docs[ticker].append((q_end, combined))

    del docs, seen  # free memory

    tickers = sorted(ticker_docs.keys())
    logger.info("Phase 2: embedding %d tickers", len(tickers))

    if not force:
        tickers = [t for t in tickers if not store.has_embeddings(t)]
        logger.info("To embed: %d tickers (skipping already done)", len(tickers))

    if not tickers:
        logger.info("All tickers already embedded!")
        return

    t0 = time.time()
    n_docs_total = 0
    n_chunks_total = 0

    for i, ticker in enumerate(tickers):
        items = sorted(ticker_docs[ticker])  # sort by date
        if not items:
            continue

        embeddings_list = []
        dates_list = []
        norms_list = []
        n_chunks_list = []

        for q_date, text in items:
            # Skip very short quarterly documents (< 100 words)
            if len(text.split()) < 100:
                continue

            embedding, norm, n_chunks = embedder.embed_text(text)
            embeddings_list.append(embedding)
            norms_list.append(norm)
            dates_list.append(q_date)
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
                i + 1, len(tickers), n_docs_total, n_chunks_total, rate,
            )

    elapsed = time.time() - t0
    logger.info(
        "Done: %d tickers, %d quarterly docs, %d chunks in %.1f min",
        len(tickers), n_docs_total, n_chunks_total, elapsed / 60,
    )


if __name__ == "__main__":
    main()
