"""
Per-firm stock price fetcher with parquet caching.

Uses Tiingo REST API for daily adjusted OHLCV data. Falls back to
cached parquet files if already downloaded. Caches locally to avoid
redundant API calls.

Tiingo free tier: 500 requests/hour, generous daily limits.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pnlp.config import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["PriceFetcher", "load_price_data"]

PRICES_DIR = DATA_DIR / "prices"

# Load API key from environment or .env
def _get_api_key() -> str:
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        from dotenv import load_dotenv
        load_dotenv(DATA_DIR.parent / ".env")
        key = os.environ.get("TIINGO_API_KEY", "")
    return key


class PriceFetcher:
    """Fetch and cache per-firm daily stock prices via Tiingo."""

    TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, cache_dir: Path | None = None, api_key: str | None = None) -> None:
        self.cache_dir = cache_dir or PRICES_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or _get_api_key()
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        })

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.parquet"

    def _fetch_one(self, ticker: str, start: date, end: date) -> pd.DataFrame | None:
        """Fetch daily adjusted prices for a single ticker from Tiingo."""
        url = f"{self.TIINGO_BASE}/{ticker}/prices"
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
        }
        try:
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                logger.warning("Rate limited on %s, sleeping 60s...", ticker)
                time.sleep(60)
                resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug("Failed to fetch %s: %s", ticker, e)
            return None

        data = resp.json()
        if not data:
            return None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        # Use adjusted columns
        col_map = {
            "adjOpen": "Open",
            "adjHigh": "High",
            "adjLow": "Low",
            "adjClose": "Close",
            "adjVolume": "Volume",
        }
        out = df.rename(columns=col_map)[list(col_map.values())]
        return out.dropna(how="all")

    def fetch(
        self,
        tickers: list[str],
        start: date,
        end: date,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Fetch daily OHLCV for a list of tickers.

        Returns dict mapping ticker -> DataFrame with DatetimeIndex and
        columns [Open, High, Low, Close, Volume]. Uses adjusted prices.
        """
        result: dict[str, pd.DataFrame] = {}
        to_download: list[str] = []

        # Check cache first
        if not force_refresh:
            for ticker in tickers:
                path = self._cache_path(ticker)
                if path.exists():
                    df = pd.read_parquet(path)
                    if len(df) > 0:
                        assert isinstance(df.index, pd.DatetimeIndex), (
                            f"Price cache for {ticker}: index must be DatetimeIndex, got {type(df.index)}"
                        )
                        assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns), (
                            f"Price cache for {ticker}: missing columns, got {list(df.columns)}"
                        )
                        result[ticker] = df
                    else:
                        to_download.append(ticker)
                else:
                    to_download.append(ticker)
        else:
            to_download = list(tickers)

        if to_download:
            logger.info("Downloading prices for %d tickers from Tiingo...", len(to_download))
            n_fetched = 0
            for i, ticker in enumerate(to_download):
                df = self._fetch_one(ticker, start, end)
                if df is not None and len(df) > 100:
                    result[ticker] = df
                    df.to_parquet(self._cache_path(ticker))
                    n_fetched += 1

                if (i + 1) % 100 == 0:
                    logger.info("  Progress: %d/%d tickers (%d with data)", i + 1, len(to_download), n_fetched)

                # Rate limiting: ~8 requests/sec to stay under 500/hour comfortably
                if (i + 1) % 400 == 0:
                    logger.info("  Pausing 60s for rate limiting...")
                    time.sleep(60)

            logger.info("Downloaded %d/%d tickers from Tiingo", n_fetched, len(to_download))

        return result

    @staticmethod
    def compute_returns(
        prices: dict[str, pd.DataFrame],
        frequency: str = "daily",
        clip_bound: float = 0.5,
    ) -> pd.DataFrame:
        """Compute log returns from price data.

        Args:
            prices: dict mapping ticker -> OHLCV DataFrame.
            frequency: 'daily', 'weekly', or 'monthly'.
            clip_bound: max absolute daily log return. Clips extreme values
                caused by data errors or penny stock noise. Default 0.5
                corresponds to roughly [-39%, +65%] in simple returns.
                Set to None to disable clipping.

        Returns:
            DataFrame with DatetimeIndex, columns = tickers, values = log returns.
        """
        close_series: dict[str, pd.Series] = {}
        for ticker, df in prices.items():
            close_col = "Close" if "Close" in df.columns else "Adj Close"
            if close_col in df.columns:
                close_series[ticker] = df[close_col]

        if not close_series:
            return pd.DataFrame()

        close_df = pd.DataFrame(close_series)

        if frequency == "weekly":
            close_df = close_df.resample("W-FRI").last()
        elif frequency == "monthly":
            close_df = close_df.resample("ME").last()

        returns = np.log(close_df / close_df.shift(1)).iloc[1:]

        if clip_bound is not None:
            n_clipped = int((returns.abs() > clip_bound).sum().sum())
            if n_clipped > 0:
                logger.info(
                    "Clipping %d extreme log returns (|r| > %.2f) across %d tickers",
                    n_clipped, clip_bound, int((returns.abs() > clip_bound).any().sum()),
                )
                returns = returns.clip(-clip_bound, clip_bound)

        return returns

    @staticmethod
    def compute_dollar_volume(
        prices: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute daily dollar volume (Close * Volume) from price data.

        Args:
            prices: dict mapping ticker -> OHLCV DataFrame (same as fetch()).

        Returns:
            DataFrame with DatetimeIndex, columns = tickers, values = daily
            dollar volume. Days with zero volume produce NaN (not zero) to
            avoid treating non-trading/halted days as "liquid".
        """
        dv_series: dict[str, pd.Series] = {}
        for ticker, df in prices.items():
            close_col = "Close" if "Close" in df.columns else "Adj Close"
            if close_col not in df.columns or "Volume" not in df.columns:
                continue
            close = df[close_col]
            volume = df["Volume"].replace(0, np.nan)
            dv_series[ticker] = close * volume

        if not dv_series:
            return pd.DataFrame()

        return pd.DataFrame(dv_series)


def load_price_data(
    tickers: list[str],
    start: date = date(2000, 1, 1),
    end: date = date(2025, 12, 31),
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Load prices, compute returns and dollar volume.

    Convenience function for backtest scripts. Loads once, returns all
    three data products needed for the backtest pipeline.

    Returns:
        (prices_dict, daily_returns, dollar_volume)
    """
    fetcher = PriceFetcher()
    prices = fetcher.fetch(tickers, start=start, end=end)
    logger.info("Loaded prices for %d / %d tickers", len(prices), len(tickers))
    returns = PriceFetcher.compute_returns(prices, frequency="daily")
    dollar_volume = PriceFetcher.compute_dollar_volume(prices)
    logger.info("Returns: %d days x %d tickers", *returns.shape)
    logger.info("Dollar volume: %d days x %d tickers", *dollar_volume.shape)
    return prices, returns, dollar_volume
