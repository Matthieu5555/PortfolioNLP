"""
Research study: does embedding cosine similarity predict return correlation?

This is the foundational validation for the entire project. If cosine
similarity of firm embeddings does NOT correlate with realized return
correlations, the embedding-based covariance matrix has no value.

Uses strictly FORWARD returns (after the as_of_date of embeddings) to
prevent look-ahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pnlp.embeddings.firm_aggregator import FirmEmbedding

logger = logging.getLogger(__name__)

__all__ = ["CorrelationStudyResult", "run_correlation_study"]


@dataclass
class CorrelationStudyResult:
    """Result of the embedding vs return correlation study."""

    spearman_rho: float
    spearman_pvalue: float
    n_pairs: int
    n_firms: int
    # Binned analysis: cosine_sim decile -> avg realized correlation
    binned_analysis: pd.DataFrame


def run_correlation_study(
    firm_embeddings: dict[str, FirmEmbedding],
    returns: pd.DataFrame,
    horizon_days: int = 252,
    gap_days: int = 5,
) -> CorrelationStudyResult:
    """Compare embedding cosine similarity to realized return correlations.

    For each pair of firms (i, j):
    - Compute cosine_sim(e_i, e_j)  [from embeddings as of as_of_date]
    - Compute realized return correlation over FORWARD window
      [as_of_date + gap_days, as_of_date + gap_days + horizon_days]

    Uses strictly forward returns to prevent look-ahead bias.
    gap_days provides a buffer for information dissemination.

    Args:
        firm_embeddings: ticker -> FirmEmbedding.
        returns: DataFrame with tickers as columns, dates as index.
        horizon_days: forward window for computing realized return correlations.
        gap_days: gap between as_of_date and start of return window.

    Returns:
        CorrelationStudyResult with rank correlation and binned analysis.
    """
    # Determine the as_of_date from embeddings
    as_of_dates = [fe.as_of_date for fe in firm_embeddings.values()]
    as_of_date = max(as_of_dates)

    # Forward return window: strictly after as_of_date + gap
    forward_start = as_of_date + timedelta(days=gap_days)
    forward_end = as_of_date + timedelta(days=gap_days + int(horizon_days * 365 / 252))

    # Align tickers: must have both embeddings and return data in the forward window
    tickers = sorted(set(firm_embeddings.keys()) & set(returns.columns))

    # Filter returns to forward window
    returns_idx = pd.to_datetime(returns.index)
    fwd_mask = (returns_idx >= str(forward_start)) & (returns_idx <= str(forward_end))
    fwd_returns = returns.loc[fwd_mask, tickers]

    # Drop tickers with any NaN in the forward window (survivorship)
    valid_cols = fwd_returns.columns[fwd_returns.notna().all()]
    fwd_returns = fwd_returns[valid_cols]
    tickers = list(valid_cols)

    if len(tickers) < 10:
        logger.warning("Too few common tickers (%d) for correlation study", len(tickers))
        return CorrelationStudyResult(
            spearman_rho=0.0,
            spearman_pvalue=1.0,
            n_pairs=0,
            n_firms=len(tickers),
            binned_analysis=pd.DataFrame(),
        )

    if len(fwd_returns) < 30:
        logger.warning("Only %d forward trading days available (need >= 30)", len(fwd_returns))
        return CorrelationStudyResult(
            spearman_rho=0.0,
            spearman_pvalue=1.0,
            n_pairs=0,
            n_firms=len(tickers),
            binned_analysis=pd.DataFrame(),
        )

    logger.info(
        "Correlation study: %d tickers, forward window %s to %s (%d days)",
        len(tickers), forward_start, forward_end, len(fwd_returns),
    )

    # Build embedding similarity matrix
    E = np.array([firm_embeddings[t].embedding for t in tickers])
    cosine_sim = E @ E.T  # (n, n)

    # Compute realized return correlation in FORWARD window
    realized_corr = fwd_returns.corr().values

    # Extract upper triangle pairs
    n = len(tickers)
    cosine_pairs = []
    corr_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            cs = cosine_sim[i, j]
            rc = realized_corr[i, j]
            if np.isfinite(cs) and np.isfinite(rc):
                cosine_pairs.append(cs)
                corr_pairs.append(rc)

    cosine_arr = np.array(cosine_pairs)
    corr_arr = np.array(corr_pairs)

    # Spearman rank correlation
    rho, pval = spearmanr(cosine_arr, corr_arr)

    # Binned analysis: deciles of cosine similarity
    df = pd.DataFrame({"cosine_sim": cosine_arr, "realized_corr": corr_arr})
    df["cosine_decile"] = pd.qcut(df["cosine_sim"], 10, labels=False, duplicates="drop")
    binned = df.groupby("cosine_decile").agg(
        avg_cosine_sim=("cosine_sim", "mean"),
        avg_realized_corr=("realized_corr", "mean"),
        n_pairs=("realized_corr", "count"),
    ).reset_index()

    logger.info(
        "Correlation study: Spearman rho=%.4f, p=%.6f, n_pairs=%d, n_firms=%d",
        rho, pval, len(cosine_arr), n,
    )

    return CorrelationStudyResult(
        spearman_rho=float(rho),
        spearman_pvalue=float(pval),
        n_pairs=len(cosine_arr),
        n_firms=n,
        binned_analysis=binned,
    )
