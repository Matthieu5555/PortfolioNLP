"""
Price data quality audit.

Validates that Tiingo adjusted prices are correctly handling
stock splits and dividends. Checks for extreme return artifacts
and reports distribution statistics.

Usage:
    uv run python pipeline/audit_price_data.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PRICES_DIR = DATA_DIR / "prices"
RESULTS_DIR = DATA_DIR / "results"


def load_all_returns(clip: bool = False) -> pd.DataFrame:
    """Load returns from all cached price files."""
    close_series: dict[str, pd.Series] = {}
    files = sorted(PRICES_DIR.glob("*.parquet"))
    logger.info("Loading %d price files...", len(files))
    for f in files:
        ticker = f.stem
        df = pd.read_parquet(f)
        if "Close" in df.columns and len(df) > 100:
            close_series[ticker] = df["Close"]
    close_df = pd.DataFrame(close_series)
    returns = np.log(close_df / close_df.shift(1)).iloc[1:]
    if clip:
        returns = returns.clip(-0.5, 0.5)
    return returns


def audit_return_distribution(returns: pd.DataFrame) -> dict:
    """Compute distribution statistics for all returns."""
    flat = returns.values.flatten()
    n_inf = int(np.sum(np.isinf(flat)))
    flat = flat[np.isfinite(flat)]  # Remove NaN and inf

    thresholds = [0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
    extreme_counts = {}
    for t in thresholds:
        n = int(np.sum(np.abs(flat) > t))
        extreme_counts[f"abs_gt_{t:.2f}"] = n
        extreme_counts[f"abs_gt_{t:.2f}_pct"] = round(100 * n / len(flat), 6)

    return {
        "n_observations": len(flat),
        "n_inf_removed": n_inf,
        "n_tickers": returns.shape[1],
        "n_days": returns.shape[0],
        "mean": round(float(np.mean(flat)), 8),
        "std": round(float(np.std(flat)), 6),
        "min": round(float(np.min(flat)), 6),
        "max": round(float(np.max(flat)), 6),
        "quantiles": {
            f"p{p}": round(float(np.percentile(flat, p)), 6)
            for p in [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
        },
        "skewness": round(float(pd.Series(flat).skew()), 4),
        "kurtosis": round(float(pd.Series(flat).kurtosis()), 4),
        "extreme_counts": extreme_counts,
    }


def audit_stock_splits(returns: pd.DataFrame) -> dict:
    """Verify known stock splits produce no return spikes in adjusted data.

    A properly adjusted price series should show NO discontinuity on the
    split date. We check that the return on the split ex-date is within
    normal range (|log r| < 0.15, i.e., roughly < 16%).
    """
    # Known splits: (ticker, ex-date, ratio description)
    known_splits = [
        ("AAPL", "2020-08-31", "4:1"),
        ("TSLA", "2020-08-31", "5:1"),
        ("AMZN", "2022-06-06", "20:1"),
        ("GOOG", "2022-07-18", "20:1"),
        ("GOOGL", "2022-07-18", "20:1"),
        ("NVDA", "2021-07-20", "4:1"),
        ("NVDA", "2024-06-10", "10:1"),
        ("TSLA", "2022-08-25", "3:1"),
        ("SHW", "2024-09-17", "4:1"),
    ]

    results = []
    for ticker, split_date, ratio in known_splits:
        if ticker not in returns.columns:
            results.append({
                "ticker": ticker,
                "split_date": split_date,
                "ratio": ratio,
                "status": "NO_DATA",
                "return_on_date": None,
            })
            continue

        ts = returns[ticker]
        if split_date not in ts.index.astype(str).values:
            # Find closest trading day
            idx = ts.index.astype(str)
            close_dates = idx[(idx >= split_date) & (idx <= str(pd.Timestamp(split_date) + pd.Timedelta(days=5)))]
            if len(close_dates) > 0:
                actual_date = close_dates[0]
                r = float(ts.loc[actual_date])
            else:
                results.append({
                    "ticker": ticker,
                    "split_date": split_date,
                    "ratio": ratio,
                    "status": "DATE_NOT_FOUND",
                    "return_on_date": None,
                })
                continue
        else:
            actual_date = split_date
            r = float(ts.loc[ts.index.astype(str) == split_date].iloc[0])

        is_normal = abs(r) < 0.15
        results.append({
            "ticker": ticker,
            "split_date": split_date,
            "actual_date": str(actual_date),
            "ratio": ratio,
            "return_on_date": round(r, 6),
            "abs_return": round(abs(r), 6),
            "status": "PASS" if is_normal else "FAIL_SPIKE",
        })

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_checked = sum(1 for r in results if r["status"] in ("PASS", "FAIL_SPIKE"))

    return {
        "splits_checked": len(known_splits),
        "data_available": n_checked,
        "passed": n_pass,
        "failed": n_checked - n_pass,
        "verdict": "PASS" if n_pass == n_checked and n_checked > 0 else "FAIL",
        "details": results,
    }


def audit_consecutive_extremes(returns: pd.DataFrame, threshold: float = 0.30) -> dict:
    """Check for suspicious consecutive extreme returns (data artifacts)."""
    suspicious = []
    for ticker in returns.columns:
        ts = returns[ticker].dropna()
        extreme = ts.abs() > threshold
        if extreme.sum() < 2:
            continue
        # Check for consecutive extremes
        for i in range(1, len(extreme)):
            if extreme.iloc[i] and extreme.iloc[i - 1]:
                suspicious.append({
                    "ticker": ticker,
                    "date1": str(ts.index[i - 1].date()),
                    "date2": str(ts.index[i].date()),
                    "return1": round(float(ts.iloc[i - 1]), 4),
                    "return2": round(float(ts.iloc[i]), 4),
                })
    return {
        "threshold": threshold,
        "n_consecutive_extreme_pairs": len(suspicious),
        "examples": suspicious[:20],
    }


def audit_clipping_impact(returns_raw: pd.DataFrame) -> dict:
    """Quantify impact of the 0.50 log return clipping."""
    flat_raw = returns_raw.values.flatten()
    flat_raw = flat_raw[np.isfinite(flat_raw)]  # Remove NaN and inf

    n_clipped = int(np.sum(np.abs(flat_raw) > 0.5))
    clipped = np.clip(flat_raw, -0.5, 0.5)

    return {
        "n_returns_clipped": n_clipped,
        "pct_clipped": round(100 * n_clipped / len(flat_raw), 6),
        "n_tickers_affected": int((returns_raw.abs() > 0.5).any().sum()),
        "mean_before_clip": round(float(np.mean(flat_raw)), 8),
        "mean_after_clip": round(float(np.mean(clipped)), 8),
        "std_before_clip": round(float(np.std(flat_raw)), 6),
        "std_after_clip": round(float(np.std(clipped)), 6),
    }


def main() -> None:
    logger.info("=" * 60)
    logger.info("PRICE DATA QUALITY AUDIT")
    logger.info("=" * 60)

    # Load raw returns (no clipping)
    returns_raw = load_all_returns(clip=False)
    logger.info("Loaded returns: %d days x %d tickers", *returns_raw.shape)

    # 1. Distribution statistics
    logger.info("Computing distribution statistics...")
    dist_stats = audit_return_distribution(returns_raw)
    logger.info(
        "  Mean=%.6f, Std=%.4f, Min=%.4f, Max=%.4f",
        dist_stats["mean"], dist_stats["std"], dist_stats["min"], dist_stats["max"],
    )
    logger.info(
        "  |r|>0.20: %d (%.4f%%), |r|>0.50: %d (%.4f%%)",
        dist_stats["extreme_counts"]["abs_gt_0.20"],
        dist_stats["extreme_counts"]["abs_gt_0.20_pct"],
        dist_stats["extreme_counts"]["abs_gt_0.50"],
        dist_stats["extreme_counts"]["abs_gt_0.50_pct"],
    )

    # 2. Stock split verification
    logger.info("Verifying stock split adjustments...")
    split_audit = audit_stock_splits(returns_raw)
    logger.info(
        "  Splits checked: %d, Passed: %d, Failed: %d → %s",
        split_audit["data_available"], split_audit["passed"],
        split_audit["failed"], split_audit["verdict"],
    )
    for detail in split_audit["details"]:
        if detail["status"] in ("PASS", "FAIL_SPIKE"):
            logger.info(
                "    %s %s (%s): return=%.4f → %s",
                detail["ticker"], detail["split_date"], detail["ratio"],
                detail["return_on_date"], detail["status"],
            )

    # 3. Consecutive extreme returns
    logger.info("Checking for consecutive extreme returns...")
    consec = audit_consecutive_extremes(returns_raw)
    logger.info(
        "  Found %d consecutive extreme pairs (|r|>0.30)",
        consec["n_consecutive_extreme_pairs"],
    )

    # 4. Clipping impact
    logger.info("Assessing clipping impact...")
    clip_impact = audit_clipping_impact(returns_raw)
    logger.info(
        "  Returns clipped: %d (%.4f%%), affecting %d tickers",
        clip_impact["n_returns_clipped"], clip_impact["pct_clipped"],
        clip_impact["n_tickers_affected"],
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "audit_date": str(date.today()),
        "data_source": "Tiingo adjClose (total-return adjusted)",
        "return_type": "log returns (unclipped for this audit)",
        "distribution": dist_stats,
        "stock_split_verification": split_audit,
        "consecutive_extremes": consec,
        "clipping_impact": clip_impact,
        "methodology_note": (
            "Tiingo adjClose provides retroactively adjusted prices accounting for "
            "stock splits, dividends, and corporate actions. For return-based covariance "
            "estimation, adjusted returns are mathematically equivalent to total returns: "
            "r_t = adjClose_t / adjClose_{t-1} - 1. Using unadjusted prices would "
            "create spurious returns on split dates (e.g., a 4:1 split would appear "
            "as a -75% return). The extreme return clipping at |log r| > 0.50 "
            "(~[-39%, +65%] in simple returns) provides an additional safety net "
            "against any remaining data artifacts."
        ),
    }

    out_path = RESULTS_DIR / "price_audit.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Audit saved to %s", out_path)


if __name__ == "__main__":
    main()
