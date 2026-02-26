"""
Cold-Start Simulation: Text Handles IPOs Where Return-Based Methods Fail.

Simulates the actual new-listing scenario: firms enter the universe with
zero or minimal return history. Text-based covariance includes them from
day one; LW must exclude them.

Compares:
  A. Text-inclusive:  SemanticShrinkage over ALL firms (new + established)
  B. LW-exclusive:    LW over established firms ONLY (new excluded)
  C. SIC-substitute:  SIC block-diagonal for new, sample cov for established
  D. EW-inclusive:    1/N over ALL firms
  E. EW-established:  1/N over established firms ONLY

"New entrant" = firm with embeddings + ADV >= $1M but < 63 trading days
of returns (entered within last quarter).

Usage:
    uv run python experiments/run_cold_start_experiment.py
    uv run python experiments/run_cold_start_experiment.py --p 500 --new-threshold 63
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.baselines.equal_weight import equal_weight_portfolio
from pnlp.baselines.shrinkage import ledoit_wolf_portfolio
from pnlp.data.dates import generate_quarterly_dates
from pnlp.baselines.sic_sector import load_sic_codes, sic_sector_covariance
from pnlp.config import (
    DATA_DIR,
    BacktestConfig,
    CovarianceConfig,
    EmbeddingConfig,
    PortfolioConfig,
)
from pnlp.data.embeddings_loader import load_doc_embeddings
from pnlp.data.prices import load_price_data
from pnlp.data.universe_filter import LiquidityConfig, filter_investable_universe
from pnlp.embeddings.firm_aggregator import FirmAggregator
from pnlp.portfolio.optimizer import PortfolioOptimizer, PortfolioWeights
from pnlp.primitives.covariance import CosineSimilarityCovariance, SemanticShrinkageCovariance
from pnlp.validation.backtest import BacktestEngine, BacktestResult
from pnlp.validation.statistical_tests import block_bootstrap_ci, block_permutation_test
from pnlp.validation.transaction_costs import TransactionCostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_cold_start_experiment(
    p: int = 500,
    text_source: str = "10k",
    start: str = "2013-04-01",
    end: str = "2024-04-01",
    lookback_days: int = 504,
    new_threshold: int = 63,
    established_threshold: int = 252,
    max_weight: float = 0.05,
) -> dict:
    """Run the cold-start simulation experiment.

    Args:
        p: maximum universe size.
        new_threshold: firms with < this many return days are "new entrants".
        established_threshold: firms with >= this many return days are "established".
    """

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    # Load data
    doc_embeddings = load_doc_embeddings(text_source)
    _, daily_returns, dollar_volume = load_price_data(list(doc_embeddings.keys()))

    rebal_dates = generate_quarterly_dates(start_date, end_date)
    logger.info("%d quarterly rebalance dates", len(rebal_dates))

    aggregator = FirmAggregator(EmbeddingConfig())
    portfolio_config = PortfolioConfig(
        objective="min_variance", long_only=True, max_weight=max_weight,
    )
    cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
    backtest_config = BacktestConfig(start_date=start_date, end_date=end_date)
    optimizer = PortfolioOptimizer(portfolio_config)

    sic_codes = load_sic_codes()
    tc_model = TransactionCostModel()

    # Storage
    weights_history: dict[str, dict[date, PortfolioWeights]] = {
        s: {} for s in STRATEGIES
    }
    tc_rates_history: dict[date, pd.Series] = {}
    rebalance_log: list[dict] = []

    for rebal_date in tqdm(rebal_dates, desc="Cold-start"):
        rebal_str = str(rebal_date)

        # Aggregate embeddings
        firm_embeddings = aggregator.aggregate_universe(
            doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )

        # --- Step 1: Identify ALL liquid firms with embeddings ---
        # Relaxed filter: no return completeness requirement
        relaxed_config = LiquidityConfig(
            max_firms=p,
            min_return_completeness=0.0,  # Accept firms with no return history
            covariance_lookback_days=lookback_days,
        )
        all_eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=relaxed_config,
        )
        # Also need firms present in the returns DataFrame columns
        all_eligible = [t for t in all_eligible if t in daily_returns.columns]

        if len(all_eligible) < 20:
            continue

        # --- Step 2: Classify firms by return history ---
        returns_before_T = daily_returns.loc[daily_returns.index < rebal_str]
        lookback_rets = returns_before_T[all_eligible].iloc[-lookback_days:]
        non_nan_counts = lookback_rets.notna().sum()

        new_entrants = [t for t in all_eligible if non_nan_counts.get(t, 0) < new_threshold]
        established = [
            t for t in all_eligible
            if non_nan_counts.get(t, 0) >= established_threshold
            and non_nan_counts.get(t, 0) / max(len(lookback_rets), 1) >= 0.8
        ]

        # All firms regardless of history length
        all_firms = all_eligible

        if len(established) < 20:
            continue

        # TC rates
        dv_before = dollar_volume.loc[dollar_volume.index < rebal_str]
        adv_window = dv_before.iloc[-relaxed_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates_history[rebal_date] = tc_model.get_tc_rates(
            median_adv, daily_returns, rebal_str,
        )

        log_entry = {
            "date": rebal_str,
            "n_all": len(all_firms),
            "n_new_entrants": len(new_entrants),
            "n_established": len(established),
            "new_entrant_tickers": new_entrants[:20],  # Truncate for log size
        }

        # --- Strategy A: Text-inclusive (ALL firms via semantic shrinkage) ---
        try:
            sub_emb = {t: firm_embeddings[t] for t in all_firms}
            avail_returns = returns_before_T[all_firms].iloc[-lookback_days:].fillna(0.0)

            # Fix zero-fill bias: use median vol for new entrants lacking history
            sample_stds = avail_returns.std()
            median_vol = sample_stds[established].median() if established else sample_stds.median()
            for t in new_entrants:
                if sample_stds.get(t, 0.0) < 1e-8:
                    sample_stds[t] = median_vol

            sem_est = SemanticShrinkageCovariance(cov_config, avail_returns)
            cov = sem_est.estimate(sub_emb, sigma_estimates=sample_stds)
            w = optimizer.optimize(cov)
            w.metadata["method"] = "text_inclusive"
            weights_history["text_inclusive"][rebal_date] = w
            log_entry["text_inclusive_alpha"] = sem_est.last_alpha_
            # How much weight goes to new entrants?
            new_weight = sum(w.weights.get(t, 0.0) for t in new_entrants)
            log_entry["text_inclusive_new_weight"] = float(new_weight)
        except Exception as e:
            logger.error("text_inclusive failed at %s: %s", rebal_date, e)

        # --- Strategy A2: Text-exclusive (established firms ONLY via semantic shrinkage) ---
        # Isolates covariance quality: same universe as LW, different method
        try:
            est_emb = {t: firm_embeddings[t] for t in established if t in firm_embeddings}
            est_returns = returns_before_T[established].iloc[-lookback_days:].fillna(0.0)
            sem_est_excl = SemanticShrinkageCovariance(cov_config, est_returns)
            cov_excl = sem_est_excl.estimate(est_emb)
            w = optimizer.optimize(cov_excl)
            w.metadata["method"] = "text_exclusive"
            weights_history["text_exclusive"][rebal_date] = w
            log_entry["text_exclusive_alpha"] = sem_est_excl.last_alpha_
        except Exception as e:
            logger.error("text_exclusive failed at %s: %s", rebal_date, e)

        # --- Strategy B: LW-exclusive (established firms ONLY) ---
        try:
            est_returns = returns_before_T[established].iloc[-lookback_days:].fillna(0.0)
            w = ledoit_wolf_portfolio(
                est_returns,
                objective=portfolio_config.objective,
                long_only=portfolio_config.long_only,
                max_weight=portfolio_config.max_weight,
            )
            weights_history["lw_exclusive"][rebal_date] = w
            log_entry["lw_shrinkage"] = w.metadata.get("shrinkage")
        except Exception as e:
            logger.error("lw_exclusive failed at %s: %s", rebal_date, e)

        # --- Strategy C: SIC-substitute ---
        # New entrants use SIC-sector covariance; established use sample cov
        # Combine into a single covariance matrix
        try:
            sample_stds = lookback_rets[all_firms].std().fillna(
                lookback_rets[all_firms].std().median()  # Median vol for NaN firms
            )
            # Build full SIC-sector covariance as fallback structure
            sic_cov = sic_sector_covariance(
                all_firms, sic_codes, sigma_estimates=sample_stds,
            )

            # For established firms, use sample covariance; for mixed pairs, use SIC
            if len(established) >= 10:
                est_returns = returns_before_T[established].iloc[-lookback_days:].fillna(0.0)
                sample_cov = est_returns.cov()

                # Blend: established×established uses sample, rest uses SIC
                blended = sic_cov.copy()
                for i in established:
                    for j in established:
                        if i in sample_cov.index and j in sample_cov.columns:
                            blended.loc[i, j] = sample_cov.loc[i, j]

                # Ensure PSD
                from pnlp.primitives.covariance import _ensure_psd
                blended_vals = _ensure_psd(blended.values)
                blended = pd.DataFrame(blended_vals, index=all_firms, columns=all_firms)
            else:
                blended = sic_cov

            w = optimizer.optimize(blended)
            w.metadata["method"] = "sic_substitute"
            weights_history["sic_substitute"][rebal_date] = w
        except Exception as e:
            logger.error("sic_substitute failed at %s: %s", rebal_date, e)

        # --- Strategy D: EW-inclusive (ALL firms) ---
        w = equal_weight_portfolio(all_firms)
        weights_history["ew_inclusive"][rebal_date] = w

        # --- Strategy E: EW-established (established ONLY) ---
        w = equal_weight_portfolio(established)
        weights_history["ew_established"][rebal_date] = w

        rebalance_log.append(log_entry)

    # --- Run backtests ---
    engine = BacktestEngine(backtest_config)
    bt_results: list[BacktestResult] = []

    for strategy_name in STRATEGIES:
        wh = weights_history[strategy_name]
        if not wh:
            logger.warning("No weights for %s, skipping", strategy_name)
            continue
        try:
            bt = engine.run(
                wh, daily_returns,
                strategy_name=strategy_name,
                tc_rates_history=tc_rates_history,
            )
            bt_results.append(bt)
        except Exception as e:
            logger.error("Backtest failed for %s: %s", strategy_name, e)

    if len(bt_results) < 2:
        return {"error": "insufficient_strategies"}

    # Statistical tests
    metrics = {r.strategy_name: r.metrics for r in bt_results}
    stat_tests: dict = {}
    all_rets = {r.strategy_name: r.portfolio_returns for r in bt_results}
    common_idx = bt_results[0].portfolio_returns.index
    for r in bt_results[1:]:
        common_idx = common_idx.intersection(r.portfolio_returns.index)
    aligned = {name: rets.loc[common_idx].values for name, rets in all_rets.items()}

    # Key comparisons — structured to isolate effects:
    # text_exclusive vs lw_exclusive = covariance quality (same universe)
    # text_inclusive vs text_exclusive = universe breadth (same method)
    # text_inclusive vs lw_exclusive = combined effect
    for pair in [
        ("text_exclusive", "lw_exclusive"),       # Covariance quality (same universe)
        ("text_inclusive", "text_exclusive"),      # Universe breadth (same method)
        ("text_inclusive", "lw_exclusive"),        # Combined effect
        ("text_inclusive", "sic_substitute"),
        ("text_inclusive", "ew_inclusive"),
        ("lw_exclusive", "ew_established"),
        ("sic_substitute", "lw_exclusive"),
    ]:
        a, b = pair
        if a in aligned and b in aligned:
            test = block_permutation_test(
                aligned[a], aligned[b],
                block_size=63, n_permutations=10000,
            )
            diffs = aligned[a] - aligned[b]
            ci = block_bootstrap_ci(diffs, block_size=63, n_bootstrap=10000)
            stat_tests[f"{a}_vs_{b}"] = {
                "diff_means": float(test.statistic),
                "p_value": float(test.p_value),
                "ci_low": float(ci.ci_low),
                "ci_high": float(ci.ci_high),
            }

    # Cold-start specific analysis
    new_counts = [e["n_new_entrants"] for e in rebalance_log]
    new_weight_pcts = [
        e.get("text_inclusive_new_weight", 0.0)
        for e in rebalance_log
    ]

    cold_start_stats = {
        "avg_new_entrants_per_quarter": float(np.mean(new_counts)),
        "max_new_entrants": int(np.max(new_counts)) if new_counts else 0,
        "min_new_entrants": int(np.min(new_counts)) if new_counts else 0,
        "avg_new_weight_in_text_portfolio": float(np.mean(new_weight_pcts)),
        "max_new_weight": float(np.max(new_weight_pcts)) if new_weight_pcts else 0,
        "n_quarters_with_new_entrants": int(sum(1 for c in new_counts if c > 0)),
        "total_quarters": len(rebalance_log),
    }

    results = {
        "experiment": "cold_start",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "p": p,
            "text_source": text_source,
            "start": start,
            "end": end,
            "lookback_days": lookback_days,
            "new_threshold": new_threshold,
            "established_threshold": established_threshold,
            "max_weight": max_weight,
        },
        "metrics": metrics,
        "statistical_tests": stat_tests,
        "cold_start_stats": cold_start_stats,
        "rebalance_log": rebalance_log,
    }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cold-start simulation experiment")
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--text-source", default="10k",
                        choices=["10k", "transcript", "news", "combined", "all"])
    parser.add_argument("--start", default="2007-04-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--new-threshold", type=int, default=63,
                        help="Days of returns below which a firm is 'new'")
    parser.add_argument("--max-weight", type=float, default=0.05)
    args = parser.parse_args()

    t0 = time.time()
    results = run_cold_start_experiment(
        p=args.p,
        text_source=args.text_source,
        start=args.start,
        end=args.end,
        new_threshold=args.new_threshold,
        max_weight=args.max_weight,
    )

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / "results" / f"cold_start_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info("Done in %.1f seconds. Results saved to %s", elapsed, out_path)

    # Print summary
    print(f"\n{'='*70}")
    print(f"COLD-START EXPERIMENT: p={args.p}, {args.text_source}")
    print(f"{'='*70}")

    cs = results.get("cold_start_stats", {})
    print(f"\nNew Entrants (< {args.new_threshold} days of returns):")
    print(f"  Avg per quarter: {cs.get('avg_new_entrants_per_quarter', 0):.1f}")
    print(f"  Range: {cs.get('min_new_entrants', 0)} - {cs.get('max_new_entrants', 0)}")
    print(f"  Quarters with new entrants: {cs.get('n_quarters_with_new_entrants', 0)}/{cs.get('total_quarters', 0)}")
    print(f"  Avg weight in text portfolio: {cs.get('avg_new_weight_in_text_portfolio', 0):.4f}")

    print(f"\nPerformance Comparison (Sharpe Ratios):")
    for strategy in STRATEGIES:
        m = results.get("metrics", {}).get(strategy, {})
        sharpe = m.get("sharpe_ratio", "N/A")
        ret = m.get("annualized_return", "N/A")
        vol = m.get("annualized_volatility", "N/A")
        if isinstance(sharpe, (int, float)):
            print(f"  {strategy:20s}: Sharpe={sharpe:.3f}, Ret={ret:.1%}, Vol={vol:.1%}")
        else:
            print(f"  {strategy:20s}: N/A")

    print(f"\nStatistical Tests:")
    for pair, test in results.get("statistical_tests", {}).items():
        pval = test.get("p_value", "N/A")
        diff = test.get("diff_means", 0)
        print(f"  {pair}: diff={diff:.6f}, p={pval:.3f}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
