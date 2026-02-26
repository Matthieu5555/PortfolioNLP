"""
Dimensional scaling experiment: How does semantic shrinkage vs Ledoit-Wolf
performance change as the ratio p/n grows?

Hypothesis: Semantic shrinkage outperforms LW increasingly as p/n grows,
because sklearn LW shrinks toward a scaled identity (no correlation structure)
while semantic shrinkage preserves text-derived pairwise correlations.

Usage:
    uv run python scripts/run_scaling_experiment.py
    uv run python scripts/run_scaling_experiment.py --p-grid 100,500,1000
    uv run python scripts/run_scaling_experiment.py --text-source transcript
    uv run python scripts/run_scaling_experiment.py --p-grid 100,200 --start 2020-01-01 --end 2021-06-30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.baselines.equal_weight import equal_weight_portfolio
from pnlp.baselines.shrinkage import ledoit_wolf_portfolio
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
from pnlp.embeddings.firm_aggregator import FirmAggregator, FirmEmbedding
from pnlp.portfolio.optimizer import PortfolioOptimizer, PortfolioWeights
from pnlp.primitives.covariance import SemanticShrinkageCovariance
from pnlp.validation.backtest import BacktestEngine, BacktestResult
from pnlp.validation.statistical_tests import block_bootstrap_ci, block_permutation_test, lo_sharpe_correction
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
# Per-strategy weight computation
# ---------------------------------------------------------------------------

def compute_semantic_weights(
    firm_embeddings: dict[str, FirmEmbedding],
    daily_returns: pd.DataFrame,
    eligible: list[str],
    rebal_date: date,
    portfolio_config: PortfolioConfig,
    cov_config: CovarianceConfig,
    lookback_days: int = 504,
) -> tuple[PortfolioWeights, float | None]:
    """Returns (weights, oracle_alpha)."""
    sub_embeddings = {t: firm_embeddings[t] for t in eligible}

    returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
    lookback_returns = returns_before_T[eligible].iloc[-lookback_days:]

    cov_estimator = SemanticShrinkageCovariance(
        config=cov_config,
        historical_returns=lookback_returns,
    )
    cov = cov_estimator.estimate(sub_embeddings, sigma_estimates=None)

    optimizer = PortfolioOptimizer(portfolio_config)
    weights = optimizer.optimize(cov)
    weights.metadata["method"] = "semantic_shrinkage"

    return weights, cov_estimator.last_alpha_


def compute_lw_weights(
    daily_returns: pd.DataFrame,
    eligible: list[str],
    rebal_date: date,
    portfolio_config: PortfolioConfig,
    lookback_days: int = 504,
) -> tuple[PortfolioWeights, float | None]:
    """Returns (weights, lw_shrinkage)."""
    returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
    lookback_returns = returns_before_T[eligible].iloc[-lookback_days:]

    weights = ledoit_wolf_portfolio(
        lookback_returns,
        objective=portfolio_config.objective,
        long_only=portfolio_config.long_only,
        max_weight=portfolio_config.max_weight,
    )
    lw_shrinkage = weights.metadata.get("shrinkage")
    return weights, lw_shrinkage


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(results: list[BacktestResult]) -> dict:
    from statsmodels.stats.multitest import multipletests

    tests: dict[str, dict] = {}
    all_rets = {r.strategy_name: r.portfolio_returns for r in results}
    common_idx = all_rets[results[0].strategy_name].index
    for r in results[1:]:
        common_idx = common_idx.intersection(all_rets[r.strategy_name].index)
    aligned = {name: rets.loc[common_idx].values for name, rets in all_rets.items()}

    strategy_names = list(aligned.keys())
    pair_keys = []
    raw_pvals = []
    for i, name_a in enumerate(strategy_names):
        for j, name_b in enumerate(strategy_names):
            if i >= j:
                continue
            test = block_permutation_test(
                aligned[name_a], aligned[name_b],
                block_size=63, n_permutations=10000,
            )
            diffs = aligned[name_a] - aligned[name_b]
            ci = block_bootstrap_ci(diffs, block_size=63, n_bootstrap=10000)
            key = f"{name_a}_vs_{name_b}"
            tests[key] = {
                "diff_means": float(test.statistic),
                "p_value": float(test.p_value),
                "ci_low": float(ci.ci_low),
                "ci_high": float(ci.ci_high),
            }
            pair_keys.append(key)
            raw_pvals.append(test.p_value)

    # BH-FDR correction across all within-p pairwise tests
    if len(raw_pvals) > 1:
        _, bh_adjusted, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
        for key, adj_p in zip(pair_keys, bh_adjusted):
            tests[key]["p_value_bh"] = round(float(adj_p), 4)

    for r in results:
        name = r.strategy_name
        rets = aligned[name]
        eta = lo_sharpe_correction(rets, max_lag=63)
        raw_sharpe = r.metrics.get("sharpe_ratio", 0.0)
        tests[f"{name}_lo_correction"] = {
            "eta": float(eta),
            "raw_sharpe": float(raw_sharpe),
            "corrected_sharpe": float(raw_sharpe / eta) if eta > 0 else 0.0,
        }

    return tests


# ---------------------------------------------------------------------------
# Single p-value backtest
# ---------------------------------------------------------------------------

def run_single_p(
    p: int,
    all_doc_embeddings: dict[str, list[DocumentEmbedding]],
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    rebalance_dates: list[date],
    aggregator: FirmAggregator,
    portfolio_config: PortfolioConfig,
    cov_config: CovarianceConfig,
    backtest_config: BacktestConfig,
    liq_config: LiquidityConfig,
    lookback_days: int,
) -> dict:
    """Run full walk-forward backtest for a single universe size p."""
    logger.info("=" * 60)
    logger.info("Starting p=%d (p/n=%.2f)", p, p / lookback_days)
    logger.info("=" * 60)

    # Override max_firms for this p value
    liq_p = LiquidityConfig(
        adv_threshold=liq_config.adv_threshold,
        adv_lookback_days=liq_config.adv_lookback_days,
        min_return_completeness=liq_config.min_return_completeness,
        covariance_lookback_days=liq_config.covariance_lookback_days,
        max_firms=p,
    )

    tc_model = TransactionCostModel()
    semantic_weights: dict[date, PortfolioWeights] = {}
    lw_weights: dict[date, PortfolioWeights] = {}
    ew_weights: dict[date, PortfolioWeights] = {}
    tc_rates_history: dict[date, pd.Series] = {}
    rebalance_log: list[dict] = []

    for rebal_date in tqdm(rebalance_dates, desc=f"p={p}"):
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

        # Per-ticker TC rates
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_p.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates_history[rebal_date] = tc_model.get_tc_rates(
            median_adv, daily_returns, str(rebal_date),
        )

        log_entry: dict = {
            "date": str(rebal_date),
            "p": p,
            "n_eligible": len(eligible),
        }

        # Semantic shrinkage
        sem_ok = False
        sem_alpha = None
        try:
            sem_w, sem_alpha = compute_semantic_weights(
                firm_embeddings, daily_returns, eligible, rebal_date,
                portfolio_config, cov_config, lookback_days,
            )
            semantic_weights[rebal_date] = sem_w
            sem_ok = True
        except Exception as e:
            logger.error("Semantic failed at %s p=%d: %s", rebal_date, p, e)

        # Ledoit-Wolf
        lw_ok = False
        lw_shrinkage = None
        try:
            lw_w, lw_shrinkage = compute_lw_weights(
                daily_returns, eligible, rebal_date,
                portfolio_config, lookback_days,
            )
            lw_weights[rebal_date] = lw_w
            lw_ok = True
        except Exception as e:
            logger.error("LW failed at %s p=%d: %s", rebal_date, p, e)

        # Equal weight
        ew_w = equal_weight_portfolio(eligible)
        ew_weights[rebal_date] = ew_w

        log_entry["semantic_alpha"] = float(sem_alpha) if sem_alpha is not None else None
        log_entry["lw_shrinkage"] = float(lw_shrinkage) if lw_shrinkage is not None else None
        log_entry["semantic_positions"] = int((sem_w.weights.abs() > 1e-6).sum()) if sem_ok else 0
        log_entry["lw_positions"] = int((lw_w.weights.abs() > 1e-6).sum()) if lw_ok else 0
        rebalance_log.append(log_entry)

    # Run backtests with stratified TC
    engine = BacktestEngine(backtest_config)
    results: list[BacktestResult] = []

    for name, weights_hist in [
        ("semantic_shrinkage", semantic_weights),
        ("ledoit_wolf", lw_weights),
        ("equal_weight", ew_weights),
    ]:
        if not weights_hist:
            continue
        bt = engine.run(
            weights_hist, daily_returns, strategy_name=name,
            tc_rates_history=tc_rates_history,
        )
        results.append(bt)

    if len(results) < 2:
        logger.warning("Fewer than 2 strategies completed for p=%d", p)
        return {"p": p, "error": "insufficient_strategies"}

    metrics = {r.strategy_name: r.metrics for r in results}
    stat_tests = run_statistical_tests(results)

    alphas = [e["semantic_alpha"] for e in rebalance_log if e.get("semantic_alpha") is not None]
    lw_shrks = [e["lw_shrinkage"] for e in rebalance_log if e.get("lw_shrinkage") is not None]

    sem_sharpe = metrics.get("semantic_shrinkage", {}).get("sharpe_ratio", None)
    lw_sharpe = metrics.get("ledoit_wolf", {}).get("sharpe_ratio", None)
    ew_sharpe = metrics.get("equal_weight", {}).get("sharpe_ratio", None)

    sharpe_gap = (sem_sharpe - lw_sharpe) if (sem_sharpe is not None and lw_sharpe is not None) else None
    sem_vs_lw = stat_tests.get("semantic_shrinkage_vs_ledoit_wolf", {})

    summary = {
        "p": p,
        "p_over_n": round(p / lookback_days, 3),
        "avg_semantic_alpha": round(sum(alphas) / len(alphas), 6) if alphas else None,
        "avg_lw_shrinkage": round(sum(lw_shrks) / len(lw_shrks), 6) if lw_shrks else None,
        "semantic_sharpe": round(sem_sharpe, 4) if sem_sharpe is not None else None,
        "lw_sharpe": round(lw_sharpe, 4) if lw_sharpe is not None else None,
        "ew_sharpe": round(ew_sharpe, 4) if ew_sharpe is not None else None,
        "sharpe_gap": round(sharpe_gap, 4) if sharpe_gap is not None else None,
        "sem_vs_lw_pvalue": round(sem_vs_lw.get("p_value", 1.0), 4),
        "sem_vs_lw_ci_low": sem_vs_lw.get("ci_low"),
        "sem_vs_lw_ci_high": sem_vs_lw.get("ci_high"),
        "sem_lo_corrected": round(
            stat_tests.get("semantic_shrinkage_lo_correction", {}).get("corrected_sharpe", 0), 4
        ),
        "lw_lo_corrected": round(
            stat_tests.get("ledoit_wolf_lo_correction", {}).get("corrected_sharpe", 0), 4
        ),
        "metrics": metrics,
        "statistical_tests": stat_tests,
        "rebalance_log": rebalance_log,
    }

    logger.info(
        "p=%d: sem_sharpe=%.4f, lw_sharpe=%.4f, gap=%.4f, avg_alpha=%.4f",
        p,
        sem_sharpe or 0,
        lw_sharpe or 0,
        sharpe_gap or 0,
        summary["avg_semantic_alpha"] or 0,
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dimensional scaling experiment: semantic shrinkage vs LW across universe sizes",
    )
    parser.add_argument("--p-grid", type=str, default="100,250,500,1000,1500,2000")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2013, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--objective", type=str, default="min_variance",
                        choices=["min_variance", "max_sharpe", "mean_variance", "risk_parity"])
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--lookback-days", type=int, default=504)
    parser.add_argument("--adv-threshold", type=float, default=1_000_000.0)
    parser.add_argument("--adv-lookback", type=int, default=21)
    parser.add_argument("--text-source", type=str, default="10k",
                        choices=["10k", "transcript", "news", "combined", "all"],
                        help="Text source for embeddings (default: 10k)")
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

    # Run experiment for each p
    all_results: list[dict] = []
    for p in p_grid:
        result = run_single_p(
            p, all_doc_embeddings, daily_returns, dollar_volume,
            rebalance_dates, aggregator, portfolio_config, cov_config,
            backtest_config, liq_config, args.lookback_days,
        )
        all_results.append(result)

    # BH correction on semantic vs LW p-values across p grid
    from statsmodels.stats.multitest import multipletests

    raw_pvalues = [r.get("sem_vs_lw_pvalue", 1.0) for r in all_results if "error" not in r]
    if len(raw_pvalues) > 1:
        _, bh_adjusted, _, _ = multipletests(raw_pvalues, alpha=0.05, method="fdr_bh")
        idx = 0
        for r in all_results:
            if "error" not in r:
                r["sem_vs_lw_bh_adjusted"] = round(float(bh_adjusted[idx]), 4)
                idx += 1

    # Print scaling table
    print("\n" + "=" * 100)
    print("DIMENSIONAL SCALING RESULTS")
    print("=" * 100)
    print(f"{'p':>6}  {'p/n':>6}  {'avg_alpha':>10}  {'avg_lw_shrk':>12}  "
          f"{'sem_sharpe':>11}  {'lw_sharpe':>10}  {'gap':>8}  {'p_value':>8}  "
          f"{'sem_lo':>8}  {'lw_lo':>8}")
    print("-" * 100)

    for r in all_results:
        if "error" in r:
            print(f"{r['p']:>6}  {'ERROR':>6}")
            continue
        print(
            f"{r['p']:>6}  "
            f"{r['p_over_n']:>6.3f}  "
            f"{r['avg_semantic_alpha'] or 0:>10.6f}  "
            f"{r['avg_lw_shrinkage'] or 0:>12.6f}  "
            f"{r['semantic_sharpe'] or 0:>11.4f}  "
            f"{r['lw_sharpe'] or 0:>10.4f}  "
            f"{r['sharpe_gap'] or 0:>8.4f}  "
            f"{r['sem_vs_lw_pvalue']:>8.4f}  "
            f"{r['sem_lo_corrected']:>8.4f}  "
            f"{r['lw_lo_corrected']:>8.4f}"
        )
    print("=" * 100)

    # Save full results
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = args.output or str(output_dir / f"scaling_experiment_{timestamp}.json")

    scaling_table = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ("metrics", "statistical_tests", "rebalance_log")}
        scaling_table.append(row)

    output = {
        "config": {
            "p_grid": p_grid,
            "start": str(args.start),
            "end": str(args.end),
            "objective": args.objective,
            "max_weight": args.max_weight,
            "lookback_days": args.lookback_days,
            "adv_threshold": args.adv_threshold,
            "adv_lookback": args.adv_lookback,
            "text_source": args.text_source,
            "tc_model": "stratified",
        },
        "scaling_table": scaling_table,
        "full_results": all_results,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
