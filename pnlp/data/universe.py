"""
Investment universe: top liquid US equities with CIK mapping.

Uses SEC EDGAR company_tickers.json for ticker-to-CIK mapping,
and yfinance for liquidity screening.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pnlp.config import DATA_DIR, UniverseConfig

logger = logging.getLogger(__name__)

__all__ = ["Firm", "UniverseLoader"]


@dataclass(frozen=True)
class Firm:
    """A firm in the investment universe."""

    ticker: str
    name: str
    cik: str
    exchange: str = ""


class UniverseLoader:
    """Load and filter the investment universe.

    The CIK mapping comes from SEC EDGAR's company_tickers.json.
    Liquidity filtering uses yfinance price/volume data.
    """

    def __init__(self, config: UniverseConfig | None = None) -> None:
        self.config = config or UniverseConfig()
        self._cik_cache_path = DATA_DIR / "cik_mapping.json"

    def load_cik_mapping(self) -> dict[str, dict]:
        """Load ticker -> {cik, name} mapping from SEC EDGAR.

        Downloads and caches company_tickers.json from SEC.
        Returns dict keyed by uppercase ticker.
        """
        if self._cik_cache_path.exists():
            with open(self._cik_cache_path) as f:
                return json.load(f)

        import requests

        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "PortfolioNLP research@example.com"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        raw = resp.json()

        # Raw format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"}, ...}
        mapping: dict[str, dict] = {}
        for entry in raw.values():
            ticker = entry["ticker"].upper()
            mapping[ticker] = {
                "cik": str(entry["cik_str"]).zfill(10),
                "name": entry["title"],
            }

        self._cik_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cik_cache_path, "w") as f:
            json.dump(mapping, f)

        logger.info("Cached CIK mapping for %d tickers", len(mapping))
        return mapping

    def screen_liquid_universe(
        self,
        as_of_date: date | None = None,
    ) -> list[Firm]:
        """Screen for the top N most-liquid US equities.

        Downloads recent price/volume data from yfinance, computes average
        daily dollar volume, and returns the top_n firms with CIK mappings.
        """
        import yfinance as yf

        as_of = as_of_date or self.config.end_date
        lookback = self.config.liquidity_lookback_days

        cik_mapping = self.load_cik_mapping()
        tickers = list(cik_mapping.keys())

        # Download in batches — yfinance handles batching internally
        logger.info("Downloading price data for %d tickers to screen liquidity...", len(tickers))

        # Use a broad download to get volume * close for dollar volume
        start_str = str(date(as_of.year - 2, as_of.month, as_of.day))
        end_str = str(as_of)

        try:
            data = yf.download(
                tickers[:5000],  # yfinance limit
                start=start_str,
                end=end_str,
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception:
            logger.exception("Failed to download screening data")
            return []

        # Compute average daily dollar volume per ticker
        adv: dict[str, float] = {}
        for ticker in tickers[:5000]:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    close = data[(ticker, "Close")].dropna()
                    volume = data[(ticker, "Volume")].dropna()
                else:
                    continue
                if len(close) < lookback // 2:
                    continue
                dollar_vol = (close * volume).tail(lookback).mean()
                if np.isfinite(dollar_vol) and dollar_vol > 0:
                    adv[ticker] = float(dollar_vol)
            except (KeyError, TypeError):
                continue

        # Sort by dollar volume, take top N
        sorted_tickers = sorted(adv, key=adv.get, reverse=True)  # type: ignore[arg-type]
        top_tickers = sorted_tickers[: self.config.top_n]

        firms = []
        for ticker in top_tickers:
            info = cik_mapping.get(ticker)
            if info:
                firms.append(
                    Firm(
                        ticker=ticker,
                        name=info["name"],
                        cik=info["cik"],
                    )
                )

        logger.info(
            "Screened universe: %d firms (from %d with volume data)",
            len(firms),
            len(adv),
        )
        return firms
