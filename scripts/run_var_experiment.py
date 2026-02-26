"""
Text-Based VaR Experiment.

Tests whether parametric Value-at-Risk computed from text-based covariance
is better calibrated than VaR from sample covariance, especially at high p/n.

Uses EQUAL WEIGHT portfolios throughout so the covariance matrix is the ONLY
variable — we're testing matrix quality, not optimization quality.

Key questions:
- At high p/n, does text-based VaR have fewer violations than sample-based?
- Does text provide well-calibrated risk estimates where returns fail?
- How does VaR calibration degrade as p/n increases for each method?

Usage:
    uv run python scripts/run_var_experiment.py
    uv run python scripts/run_var_experiment.py --p-grid "200,500,1000,2000"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.data.dates import generate_quarterly_dates
from pnlp.config import (
    DATA_DIR,
    CovarianceConfig,
    EmbeddingConfig,
)
from pnlp.data.embeddings_loader import load_doc_embeddings
from pnlp.data.prices import load_price_data
from pnlp.data.universe_filter import LiquidityConfig, filter_investable_universe
from pnlp.embeddings.firm_aggregator import FirmAggregator
from pnlp.primitives.covariance import CosineSimilarityCovariance, SemanticShrinkageCovariance
from pnlp.validation.var_tests import christoffersen_test, kupiec_test, parametric_var

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_portfolio_vol(
    cov_matrix: pd.DataFrame,
    tickers: list[str],
) -> float:
    """Compute EW portfolio volatility from a covariance matrix."""
    aligned = cov_matrix.loc[tickers, tickers]
    n = len(tickers)
    w = np.ones(n) / n
    port_var = w @ aligned.values @ w
    return float(np.sqrt(max(port_var, 0.0)))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_var_experiment(
    p_grid: list[int],
    text_source: str = "10k",
    start: str = "2013-04-01",
    end: str = "2024-04-01",
    lookback_days: int = 504,
    forward_days: int = 63,
) -> dict:
    """Run VaR calibration experiment across universe sizes."""

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    # Load data
    doc_embeddings = load_doc_embeddings(text_source)
    _, daily_returns, dollar_volume = load_price_data(list(doc_embeddings.keys()))

    # Convert to simple returns for VaR
    simple_returns = np.exp(daily_returns) - 1.0

    rebal_dates = generate_quarterly_dates(start_date, end_date)
    logger.info("%d quarterly rebalance dates", len(rebal_dates))

    aggregator = FirmAggregator(EmbeddingConfig())

    all_results = []

    for p in p_grid:
        logger.info("=" * 60)
        logger.info("VaR EXPERIMENT: p=%d", p)
        logger.info("=" * 60)

        liq_config = LiquidityConfig(max_firms=p)

        # Collect violations per (method, alpha)
        violations: dict[str, dict[float, list[bool]]] = {
            method: {alpha: [] for alpha in VAR_ALPHAS}
            for method in COV_METHODS
        }
        n_obs_total = 0
        n_rebal = 0

        for rebal_date in tqdm(rebal_dates[:-1], desc=f"p={p} VaR"):
            rebal_str = str(rebal_date)

            # Need forward returns
            forward_rets = simple_returns.loc[simple_returns.index > rebal_str]
            if len(forward_rets) < forward_days:
                continue

            # Aggregate embeddings and filter universe
            firm_embeddings = aggregator.aggregate_universe(
                doc_embeddings, as_of_date=rebal_date, min_documents=1,
            )
            eligible = filter_investable_universe(
                firm_embeddings, daily_returns, dollar_volume,
                rebal_date, config=liq_config,
            )
            if len(eligible) < 20:
                continue

            eligible = [t for t in eligible if t in daily_returns.columns]
            returns_before = daily_returns.loc[daily_returns.index < rebal_str]
            lookback_rets = returns_before[eligible].iloc[-lookback_days:]
            completeness = lookback_rets.notna().sum() / len(lookback_rets)
            eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]
            if len(eligible) < 20:
                continue

            sub_embeddings = {t: firm_embeddings[t] for t in eligible}
            lookback_returns = lookback_rets[eligible].fillna(0.0)
            sample_stds = lookback_returns.std()

            # --- Compute covariance matrices ---
            cov_matrices = {}

            # Text-only (no returns)
            text_cov_est = CosineSimilarityCovariance(CovarianceConfig())
            cov_matrices["text_only"] = text_cov_est.estimate(
                sub_embeddings, sigma_estimates=sample_stds,
            )

            # Semantic shrinkage
            sem_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
            sem_est = SemanticShrinkageCovariance(sem_config, lookback_returns)
            cov_matrices["semantic"] = sem_est.estimate(sub_embeddings)

            # Sample raw (no shrinkage)
            S_raw = lookback_returns.cov()
            cov_matrices["sample_raw"] = S_raw

            # Ledoit-Wolf
            from pnlp.primitives.gpu_accel import ledoit_wolf_covariance
            lw_cov, _lw_shrinkage = ledoit_wolf_covariance(lookback_returns.values)
            cov_matrices["lw"] = pd.DataFrame(
                lw_cov, index=eligible, columns=eligible
            )

            # --- Compute EW portfolio returns for forward period ---
            fwd_simple = simple_returns.loc[simple_returns.index > rebal_str]
            fwd_simple = fwd_simple[eligible].iloc[:forward_days]

            n = len(eligible)
            ew_port_returns = fwd_simple.mean(axis=1)  # EW daily returns

            # --- Compute VaR and check violations ---
            for method in COV_METHODS:
                port_vol = compute_portfolio_vol(cov_matrices[method], eligible)

                for alpha in VAR_ALPHAS:
                    var_threshold = parametric_var(port_vol, alpha)

                    # Check each day: violation if loss > VaR
                    for daily_ret in ew_port_returns.values:
                        loss = -daily_ret  # loss is negative return
                        violation = loss > var_threshold
                        violations[method][alpha].append(bool(violation))

            n_obs_total += len(ew_port_returns)
            n_rebal += 1

        # --- Aggregate results for this p ---
        p_results = {"p": p, "p_over_n": p / lookback_days, "n_rebal": n_rebal}
        method_results = {}

        for method in COV_METHODS:
            method_results[method] = {}
            for alpha in VAR_ALPHAS:
                v = violations[method][alpha]
                v_array = np.array(v)
                n_total = len(v_array)
                n_viol = int(v_array.sum())

                kupiec = kupiec_test(n_total, n_viol, alpha)
                christoff = christoffersen_test(v_array, alpha)

                method_results[method][str(alpha)] = {
                    "kupiec": kupiec,
                    "christoffersen": christoff,
                }

        p_results["methods"] = method_results
        all_results.append(p_results)

        # Print summary for this p
        print(f"\n{'='*60}")
        print(f"VaR Results: p={p} (p/n={p/lookback_days:.2f})")
        print(f"{'='*60}")
        for alpha in VAR_ALPHAS:
            print(f"\n  VaR {alpha*100:.1f}% (expected violation rate: {alpha:.3f}):")
            for method in COV_METHODS:
                k = method_results[method][str(alpha)]["kupiec"]
                rate = k["violation_rate"]
                reject = "REJECT" if k["reject_5pct"] else "OK"
                ratio = rate / alpha if alpha > 0 else float("inf")
                print(f"    {method:15s}: rate={rate:.4f} (ratio={ratio:.2f}) "
                      f"p={k['p_value']:.3f} [{reject}]")

    return {
        "experiment": "var_calibration",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "p_grid": p_grid,
            "text_source": text_source,
            "start": start,
            "end": end,
            "lookback_days": lookback_days,
            "forward_days": forward_days,
            "var_alphas": VAR_ALPHAS,
        },
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Text-based VaR experiment")
    parser.add_argument("--p-grid", default="200,500,1000,2000")
    parser.add_argument("--text-source", default="10k",
                        choices=["10k", "transcript", "news", "combined", "all"])
    parser.add_argument("--start", default="2007-04-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    p_grid = [int(x) for x in args.p_grid.split(",")]

    t0 = time.time()
    results = run_var_experiment(
        p_grid=p_grid,
        text_source=args.text_source,
        start=args.start,
        end=args.end,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / "results" / f"var_experiment_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info("Done in %.1f seconds. Results saved to %s", elapsed, out_path)


if __name__ == "__main__":
    main()
