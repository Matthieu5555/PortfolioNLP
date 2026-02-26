"""
Synthetic data factories for testing.

Generates fake firms, documents, embeddings, and prices so tests can
run without network access or ML model loading.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pnlp.data.universe import Firm
from pnlp.embeddings.document_embedder import DocumentEmbedding
from pnlp.embeddings.firm_aggregator import FirmEmbedding
from pnlp.portfolio.optimizer import PortfolioWeights


# ---------------------------------------------------------------------------
# Firms
# ---------------------------------------------------------------------------
SYNTHETIC_FIRMS = [
    Firm(ticker="AAPL", name="Apple Inc", cik="0000320193"),
    Firm(ticker="MSFT", name="Microsoft Corp", cik="0000789019"),
    Firm(ticker="GOOGL", name="Alphabet Inc", cik="0001652044"),
    Firm(ticker="AMZN", name="Amazon.com Inc", cik="0001018724"),
    Firm(ticker="META", name="Meta Platforms Inc", cik="0001326801"),
    Firm(ticker="JPM", name="JPMorgan Chase", cik="0000019617"),
    Firm(ticker="XOM", name="Exxon Mobil Corp", cik="0000034088"),
    Firm(ticker="JNJ", name="Johnson & Johnson", cik="0000200406"),
    Firm(ticker="PG", name="Procter & Gamble", cik="0000080424"),
    Firm(ticker="UNH", name="UnitedHealth Group", cik="0000731766"),
]


@pytest.fixture
def synthetic_firms() -> list[Firm]:
    return list(SYNTHETIC_FIRMS)


# ---------------------------------------------------------------------------
# Document Embeddings
# ---------------------------------------------------------------------------
def make_doc_embeddings(
    firms: list[Firm],
    n_docs_per_firm: int = 5,
    embed_dim: int = 768,
    seed: int = 42,
    start_date: date = date(2020, 1, 1),
) -> dict[str, list[DocumentEmbedding]]:
    """Generate synthetic document embeddings for testing.

    Each firm gets a cluster of document embeddings centered around
    a random centroid, with some noise. Tech firms are closer together,
    banks are closer together, etc.
    """
    rng = np.random.RandomState(seed)
    result: dict[str, list[DocumentEmbedding]] = {}

    for i, firm in enumerate(firms):
        # Create a centroid for this firm
        centroid = rng.randn(embed_dim).astype(np.float32)
        centroid = centroid / np.linalg.norm(centroid)

        docs = []
        for j in range(n_docs_per_firm):
            # Add noise to centroid
            noise = rng.randn(embed_dim).astype(np.float32) * 0.1
            embedding = centroid + noise
            embedding = embedding / np.linalg.norm(embedding)

            doc_date = start_date + timedelta(days=j * 90)
            docs.append(
                DocumentEmbedding(
                    ticker=firm.ticker,
                    doc_date=doc_date,
                    doc_type="10-K",
                    embedding=embedding,
                    norm=float(np.linalg.norm(embedding)),
                    n_chunks=3,
                )
            )

        result[firm.ticker] = docs

    return result


@pytest.fixture
def doc_embeddings(synthetic_firms) -> dict[str, list[DocumentEmbedding]]:
    return make_doc_embeddings(synthetic_firms)


# ---------------------------------------------------------------------------
# Firm Embeddings
# ---------------------------------------------------------------------------
def make_firm_embeddings(
    firms: list[Firm],
    embed_dim: int = 768,
    seed: int = 42,
    as_of_date: date = date(2024, 1, 1),
) -> dict[str, FirmEmbedding]:
    """Generate synthetic firm embeddings."""
    rng = np.random.RandomState(seed)
    result: dict[str, FirmEmbedding] = {}

    for firm in firms:
        embedding = rng.randn(embed_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        result[firm.ticker] = FirmEmbedding(
            ticker=firm.ticker,
            as_of_date=as_of_date,
            embedding=embedding,
            n_documents=5,
        )

    return result


@pytest.fixture
def firm_embeddings(synthetic_firms) -> dict[str, FirmEmbedding]:
    return make_firm_embeddings(synthetic_firms)


# ---------------------------------------------------------------------------
# Prices and Returns
# ---------------------------------------------------------------------------
def make_returns(
    tickers: list[str],
    n_days: int = 500,
    seed: int = 42,
    start_date: date = date(2022, 1, 1),
) -> pd.DataFrame:
    """Generate synthetic daily log returns via GBM.

    Returns DataFrame with DatetimeIndex and ticker columns.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    returns = rng.randn(n_days, len(tickers)) * 0.015  # ~1.5% daily vol
    return pd.DataFrame(returns, index=dates, columns=tickers)


def make_dollar_volume(
    tickers: list[str],
    n_days: int = 500,
    seed: int = 42,
    start_date: date = date(2022, 1, 1),
    base_adv: float = 50_000_000.0,
) -> pd.DataFrame:
    """Generate synthetic daily dollar volume.

    Each ticker gets a base ADV drawn from a log-normal, then daily
    variation layered on top. Index matches make_returns().
    """
    rng = np.random.RandomState(seed + 999)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    # Per-ticker base ADV (log-normal centered around base_adv)
    ticker_adv = rng.lognormal(
        mean=np.log(base_adv), sigma=1.0, size=len(tickers),
    )
    # Daily variation: multiply by a uniform in [0.5, 1.5]
    daily_mult = rng.uniform(0.5, 1.5, size=(n_days, len(tickers)))
    dv = daily_mult * ticker_adv[np.newaxis, :]
    return pd.DataFrame(dv, index=dates, columns=tickers)


@pytest.fixture
def daily_returns(synthetic_firms) -> pd.DataFrame:
    tickers = [f.ticker for f in synthetic_firms]
    return make_returns(tickers)


@pytest.fixture
def dollar_volume(synthetic_firms) -> pd.DataFrame:
    tickers = [f.ticker for f in synthetic_firms]
    return make_dollar_volume(tickers)
