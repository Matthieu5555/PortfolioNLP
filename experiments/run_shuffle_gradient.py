"""
Shuffle test gradient: Does firm-specific text content matter more at higher p/n?

At p=500 (alpha~4.5%), the sample covariance dominates and shuffling embeddings
has no effect. Hypothesis: at p=2000 (alpha~6.5%), the semantic target contributes
more weight, so shuffling should degrade performance.

For each p in the grid, we run one unshuffled semantic backtest and N shuffled
variants (ticker→embedding mapping permuted). We report Sharpe gap, z-score,
and empirical p-value at each universe size.

Usage:
    uv run python experiments/run_shuffle_gradient.py
    uv run python experiments/run_shuffle_gradient.py --p-grid "2000"
    uv run python experiments/run_shuffle_gradient.py --p-grid "500,1000,1500,2000" --n-shuffles 10
    uv run python experiments/run_shuffle_gradient.py --text-source transcript
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
from pnlp.primitives.covariance import SemanticShrinkageCovariance
from pnlp.validation.backtest import BacktestEngine
from pnlp.validation.transaction_costs import TransactionCostModel
from pnlp.data.dates import generate_quarterly_dates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Single-p shuffle test
# ---------------------------------------------------------------------------

def run_shuffle_at_p(
    p: int,
    all_doc_embeddings: dict,
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    rebalance_dates: list[date],
    aggregator: FirmAggregator,
    portfolio_config: PortfolioConfig,
    cov_config: CovarianceConfig,
    backtest_config: BacktestConfig,
    liq_config: LiquidityConfig,
    lookback_days: int,
    n_shuffles: int = 10,
    seed: int = 42,
) -> dict:
    """Run unshuffled + N shuffled backtests at a single universe size p."""
    rng = np.random.RandomState(seed)
    tc_model = TransactionCostModel()

    # Override max_firms for this p value
    liq_p = LiquidityConfig(
        adv_threshold=liq_config.adv_threshold,
        adv_lookback_days=liq_config.adv_lookback_days,
        min_return_completeness=liq_config.min_return_completeness,
        covariance_lookback_days=liq_config.covariance_lookback_days,
        max_firms=p,
    )

    logger.info("=" * 60)
    logger.info("SHUFFLE GRADIENT: p=%d (p/n=%.2f), n_shuffles=%d", p, p / lookback_days, n_shuffles)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Run UNSHUFFLED semantic (baseline)
    # ------------------------------------------------------------------
    logger.info("--- Running unshuffled semantic baseline ---")
    unshuffled_weights: dict[date, PortfolioWeights] = {}
    tc_rates_history: dict[date, pd.Series] = {}
    alphas: list[float] = []

    # Cache per-date universe info for reuse in shuffled runs
    date_cache: dict[date, dict] = {}

    for rebal_date in tqdm(rebalance_dates, desc=f"p={p} unshuffled"):
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_p,
        )
        if len(eligible) < 20:
            continue

        # Enforce identical universe: intersect with return data + completeness
        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-lookback_days:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]
        if len(eligible) < 20:
            continue

        # TC rates
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_p.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates = tc_model.get_tc_rates(median_adv, daily_returns, str(rebal_date))
        tc_rates_history[rebal_date] = tc_rates

        sub_embeddings = {t: firm_embeddings[t] for t in eligible}
        lookback_returns = lookback_rets[eligible]

        # Cache for shuffled runs
        date_cache[rebal_date] = {
            "eligible": eligible,
            "sub_embeddings": sub_embeddings,
            "lookback_returns": lookback_returns,
        }

        try:
            cov_estimator = SemanticShrinkageCovariance(
                config=cov_config, historical_returns=lookback_returns,
            )
            cov = cov_estimator.estimate(sub_embeddings, sigma_estimates=None)
            if cov_estimator.last_alpha_ is not None:
                alphas.append(cov_estimator.last_alpha_)
            optimizer = PortfolioOptimizer(portfolio_config)
            w = optimizer.optimize(cov)
            w.metadata["method"] = "semantic_shrinkage"
            unshuffled_weights[rebal_date] = w
        except Exception as e:
            logger.error("Unshuffled failed at %s p=%d: %s", rebal_date, p, e)

    # Backtest unshuffled
    engine = BacktestEngine(backtest_config)
    unshuffled_sharpe = None
    if unshuffled_weights:
        bt_unshuf = engine.run(
            unshuffled_weights, daily_returns,
            strategy_name="unshuffled_semantic",
            tc_rates_history=tc_rates_history,
        )
        unshuffled_sharpe = bt_unshuf.metrics.get("sharpe_ratio", 0)
        logger.info("p=%d unshuffled Sharpe: %.4f", p, unshuffled_sharpe)
    else:
        logger.warning("No unshuffled weights for p=%d", p)
        return {"p": p, "error": "no_unshuffled_weights"}

    # ------------------------------------------------------------------
    # Step 2: Run SHUFFLED versions
    # ------------------------------------------------------------------
    shuffle_results = []
    for shuffle_i in range(n_shuffles):
        logger.info("--- p=%d Shuffle %d/%d ---", p, shuffle_i + 1, n_shuffles)
        shuffled_weights: dict[date, PortfolioWeights] = {}

        for rebal_date in tqdm(date_cache.keys(), desc=f"p={p} shuffle {shuffle_i+1}"):
            cache = date_cache[rebal_date]
            eligible = cache["eligible"]
            sub_embeddings = cache["sub_embeddings"]
            lookback_returns = cache["lookback_returns"]

            # SHUFFLE: randomly permute the embedding assignment among eligible tickers
            shuffled_tickers = list(eligible)
            rng.shuffle(shuffled_tickers)
            shuffled_embeddings = {
                orig: sub_embeddings[shuf]
                for orig, shuf in zip(eligible, shuffled_tickers)
            }

            try:
                cov_estimator = SemanticShrinkageCovariance(
                    config=cov_config, historical_returns=lookback_returns,
                )
                cov = cov_estimator.estimate(shuffled_embeddings, sigma_estimates=None)
                optimizer = PortfolioOptimizer(portfolio_config)
                w = optimizer.optimize(cov)
                w.metadata["method"] = "shuffled_semantic"
                shuffled_weights[rebal_date] = w
            except Exception as e:
                logger.error("Shuffled semantic failed at %s p=%d: %s", rebal_date, p, e)

        if shuffled_weights:
            bt_shuf = engine.run(
                shuffled_weights, daily_returns,
                strategy_name=f"shuffled_{shuffle_i}",
                tc_rates_history=tc_rates_history,
            )
            shuf_sharpe = bt_shuf.metrics.get("sharpe_ratio", 0)
            shuffle_results.append({
                "shuffle_id": shuffle_i,
                "sharpe": shuf_sharpe,
                "ann_return": bt_shuf.metrics.get("annualized_return", 0),
                "ann_vol": bt_shuf.metrics.get("annualized_volatility", 0),
                "max_drawdown": bt_shuf.metrics.get("max_drawdown", 0),
                "avg_turnover": bt_shuf.metrics.get("avg_turnover", 0),
            })
            logger.info("p=%d shuffle %d Sharpe: %.4f", p, shuffle_i, shuf_sharpe)

    # ------------------------------------------------------------------
    # Step 3: Compute statistics
    # ------------------------------------------------------------------
    shuffled_sharpes = [r["sharpe"] for r in shuffle_results]
    n_better = sum(1 for s in shuffled_sharpes if s >= unshuffled_sharpe)
    mean_shuf = float(np.mean(shuffled_sharpes)) if shuffled_sharpes else None
    std_shuf = float(np.std(shuffled_sharpes)) if shuffled_sharpes else None

    z_score = None
    if std_shuf and std_shuf > 0:
        z_score = (unshuffled_sharpe - mean_shuf) / std_shuf

    avg_alpha = float(np.mean(alphas)) if alphas else None

    return {
        "p": p,
        "p_over_n": round(p / lookback_days, 3),
        "avg_oracle_alpha": round(avg_alpha, 6) if avg_alpha is not None else None,
        "n_shuffles": n_shuffles,
        "unshuffled_sharpe": round(unshuffled_sharpe, 4),
        "shuffled_mean_sharpe": round(mean_shuf, 4) if mean_shuf is not None else None,
        "shuffled_std_sharpe": round(std_shuf, 6) if std_shuf is not None else None,
        "z_score": round(z_score, 4) if z_score is not None else None,
        "empirical_p_value": round(n_better / len(shuffled_sharpes), 4) if shuffled_sharpes else None,
        "n_shuffles_better": n_better,
        "fraction_better": round(n_better / len(shuffled_sharpes), 4) if shuffled_sharpes else None,
        "unshuffled_metrics": {
            "ann_return": bt_unshuf.metrics.get("annualized_return", 0),
            "ann_vol": bt_unshuf.metrics.get("annualized_volatility", 0),
            "max_drawdown": bt_unshuf.metrics.get("max_drawdown", 0),
            "avg_turnover": bt_unshuf.metrics.get("avg_turnover", 0),
        },
        "per_shuffle": shuffle_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shuffle test gradient: does text content matter more at higher p/n?",
    )
    parser.add_argument("--p-grid", type=str, default="500,1000,1500,2000",
                        help="Comma-separated universe sizes (default: 500,1000,1500,2000)")
    parser.add_argument("--n-shuffles", type=int, default=10,
                        help="Number of shuffled runs per p value (default: 10)")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2013, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--objective", type=str, default="min_variance",
                        choices=["min_variance", "max_sharpe", "mean_variance", "risk_parity"])
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--lookback-days", type=int, default=504)
    parser.add_argument("--adv-threshold", type=float, default=1_000_000.0)
    parser.add_argument("--adv-lookback", type=int, default=21)
    parser.add_argument("--text-source", type=str, default="10k",
                        choices=["10k", "transcript", "news", "combined", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    p_grid = [int(x.strip()) for x in args.p_grid.split(",")]

    # Load data once
    all_doc_embeddings = load_doc_embeddings(args.text_source)
    _, daily_returns, dollar_volume = load_price_data(list(all_doc_embeddings.keys()))

    rebalance_dates = generate_quarterly_dates(args.start, args.end)
    logger.info("%d quarterly rebalance dates from %s to %s",
                len(rebalance_dates), rebalance_dates[0], rebalance_dates[-1])

    embedding_config = EmbeddingConfig()
    portfolio_config = PortfolioConfig(
        objective=args.objective, long_only=True, max_weight=args.max_weight,
    )
    cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
    backtest_config = BacktestConfig(start_date=args.start, end_date=args.end)
    liq_config = LiquidityConfig(
        adv_threshold=args.adv_threshold,
        adv_lookback_days=args.adv_lookback,
        covariance_lookback_days=args.lookback_days,
    )
    aggregator = FirmAggregator(embedding_config)

    # Run shuffle test for each p (largest first for headline result)
    p_grid_sorted = sorted(p_grid, reverse=True)
    all_results: list[dict] = []

    for p in p_grid_sorted:
        t0 = time.time()
        result = run_shuffle_at_p(
            p, all_doc_embeddings, daily_returns, dollar_volume,
            rebalance_dates, aggregator, portfolio_config, cov_config,
            backtest_config, liq_config, args.lookback_days,
            n_shuffles=args.n_shuffles, seed=args.seed,
        )
        result["runtime_seconds"] = round(time.time() - t0, 1)
        all_results.append(result)

        # Print interim result
        if "error" not in result:
            print(f"\n  p={p}: unshuffled={result['unshuffled_sharpe']:.4f}, "
                  f"shuffled_mean={result['shuffled_mean_sharpe']:.4f}, "
                  f"z={result['z_score']:.2f}, "
                  f"p_emp={result['empirical_p_value']:.3f}, "
                  f"alpha={result['avg_oracle_alpha']:.4f}, "
                  f"runtime={result['runtime_seconds']:.0f}s")

    # Sort results by p for display
    all_results.sort(key=lambda r: r["p"])

    # Print summary table
    print("\n" + "=" * 100)
    print("SHUFFLE GRADIENT RESULTS")
    print("=" * 100)
    print(f"{'p':>6}  {'p/n':>6}  {'alpha':>8}  {'unshuf':>8}  "
          f"{'shuf_mean':>10}  {'shuf_std':>10}  {'z_score':>8}  "
          f"{'p_emp':>8}  {'n_better':>8}  {'time_s':>8}")
    print("-" * 100)

    for r in all_results:
        if "error" in r:
            print(f"{r['p']:>6}  ERROR: {r['error']}")
            continue
        print(
            f"{r['p']:>6}  "
            f"{r['p_over_n']:>6.3f}  "
            f"{r['avg_oracle_alpha'] or 0:>8.4f}  "
            f"{r['unshuffled_sharpe']:>8.4f}  "
            f"{r['shuffled_mean_sharpe']:>10.4f}  "
            f"{r['shuffled_std_sharpe']:>10.6f}  "
            f"{r['z_score'] or 0:>8.4f}  "
            f"{r['empirical_p_value'] or 0:>8.4f}  "
            f"{r['n_shuffles_better']:>8d}  "
            f"{r.get('runtime_seconds', 0):>8.0f}"
        )
    print("=" * 100)

    # Save results
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"shuffle_gradient_{ts}.json"

    output = {
        "experiment": "shuffle_gradient",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "p_grid": p_grid,
            "n_shuffles": args.n_shuffles,
            "start": str(args.start),
            "end": str(args.end),
            "objective": args.objective,
            "max_weight": args.max_weight,
            "lookback_days": args.lookback_days,
            "text_source": args.text_source,
            "seed": args.seed,
        },
        "shuffle_gradient": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Results saved to %s", out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
