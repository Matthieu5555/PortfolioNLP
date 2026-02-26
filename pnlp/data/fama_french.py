"""
Download and cache Fama-French factor data from Kenneth French's data library.

Provides FF5 + Momentum daily factors for factor regression analysis.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from pnlp.config import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["load_ff_factors"]

_FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

_CACHE_DIR = DATA_DIR / "ff_factors"


def _download_and_parse_ff_csv(url: str, cache_name: str) -> pd.DataFrame:
    """Download a zipped CSV from French's library, cache locally, and parse."""
    cache_path = _CACHE_DIR / f"{cache_name}.parquet"
    if cache_path.exists():
        logger.info("Loading cached FF data from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Downloading FF data from %s", url)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    response = urlopen(url)  # noqa: S310
    zf = zipfile.ZipFile(io.BytesIO(response.read()))

    # The zip contains a single CSV
    csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
    raw = zf.read(csv_name).decode("utf-8")

    # French's CSVs have header text before the data. Find the first line
    # that starts with a date (YYYYMMDD = 8 digits).
    lines = raw.strip().split("\n")
    data_start = None
    data_end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and len(stripped.split(",")[0].strip()) == 8:
            if data_start is None:
                data_start = i
            data_end = i
        elif data_start is not None and not (stripped and stripped[0].isdigit()):
            # Hit non-data line after data started (monthly section follows daily)
            break

    if data_start is None:
        raise ValueError(f"Could not find data start in {csv_name}")

    # Find header line (the line just before data_start that has commas)
    header_line = None
    for i in range(data_start - 1, -1, -1):
        if "," in lines[i]:
            header_line = i
            break

    if header_line is None:
        raise ValueError(f"Could not find header line before data in {csv_name}")

    # Parse the daily data section
    csv_section = "\n".join(lines[header_line : data_end + 1])
    df = pd.read_csv(io.StringIO(csv_section), skipinitialspace=True)

    # First column is the date in YYYYMMDD format
    date_col = df.columns[0]
    df[date_col] = df[date_col].astype(str).str.strip()
    df.index = pd.to_datetime(df[date_col], format="%Y%m%d")
    df.index.name = "date"
    df = df.drop(columns=[date_col])

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Convert from percentage to decimal
    df = df.astype(float) / 100.0

    # Cache
    df.to_parquet(cache_path)
    logger.info("Cached FF data to %s (%d rows)", cache_path, len(df))

    return df


def load_ff_factors(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load FF5 + Momentum daily factors.

    Returns DataFrame with columns:
        Mkt-RF, SMB, HML, RMW, CMA, Mom, RF

    All values are in decimal (not percent): 0.01 = 1%.

    Args:
        start: start date (inclusive), e.g. '2013-01-01'.
        end: end date (inclusive), e.g. '2024-06-30'.

    Returns:
        DataFrame indexed by date.
    """
    ff5 = _download_and_parse_ff_csv(_FF5_URL, "ff5_daily")
    mom = _download_and_parse_ff_csv(_MOM_URL, "mom_daily")

    # Rename Mom column (it's named 'Mom   ' or similar)
    mom_cols = [c for c in mom.columns if c.strip().lower().startswith("mom")]
    if mom_cols:
        mom = mom.rename(columns={mom_cols[0]: "Mom"})
    else:
        # Fallback: use the first non-date column
        mom.columns = ["Mom"]

    # Merge on date
    factors = ff5.join(mom[["Mom"]], how="inner")

    if start:
        factors = factors.loc[start:]
    if end:
        factors = factors.loc[:end]

    logger.info(
        "FF5+Mom factors: %d days, %s to %s",
        len(factors),
        factors.index.min().date(),
        factors.index.max().date(),
    )
    return factors
