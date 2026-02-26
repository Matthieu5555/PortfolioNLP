"""
Walk-forward backtest: Semantic Shrinkage vs Ledoit-Wolf vs 1/N.

Core research question: Does shrinking sample covariance toward a
text-similarity prior improve portfolio performance over standard
Ledoit-Wolf shrinkage?

Usage:
    uv run python experiments/run_backtest.py
    uv run python experiments/run_backtest.py --max-firms 500
    uv run python experiments/run_backtest.py --text-source transcript
    uv run python experiments/run_backtest.py --text-source combined --max-firms 500
    uv run python experiments/run_backtest.py --start 2015-01-01 --end 2023-12-31 --objective min_variance
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
from pnlp.embeddings.firm_aggregator import FirmAggregator
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

def compute_semantic_shrinkage_weights(
    firm_embeddings: dict,
    daily_returns: pd.DataFrame,
    eligible: list[str],
    rebal_date: date,
    portfolio_config: PortfolioConfig,
    cov_config: CovarianceConfig,
    lookback_days: int = 504,
) -> PortfolioWeights:
    """Compute portfolio weights using semantic shrinkage covariance."""
    sub_embeddings = {t: firm_embeddings[t] for t in eligible}

    # Strict lag: returns before T only
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
    return weights


def compute_lw_weights(
    daily_returns: pd.DataFrame,
    eligible: list[str],
    rebal_date: date,
    portfolio_config: PortfolioConfig,
    lookback_days: int = 504,
) -> PortfolioWeights:
    """Compute portfolio weights using Ledoit-Wolf shrinkage covariance."""
    returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
    lookback_returns = returns_before_T[eligible].iloc[-lookback_days:]

    weights = ledoit_wolf_portfolio(
        lookback_returns,
        objective=portfolio_config.objective,
        long_only=portfolio_config.long_only,
        max_weight=portfolio_config.max_weight,
    )
    return weights


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def run_statistical_tests(results: list[BacktestResult]) -> dict:
    """Run block permutation tests and Lo Sharpe correction across strategies."""
    tests: dict[str, dict] = {}

    all_rets = {r.strategy_name: r.portfolio_returns for r in results}
    common_idx = all_rets[results[0].strategy_name].index
    for r in results[1:]:
        common_idx = common_idx.intersection(all_rets[r.strategy_name].index)

    aligned = {name: rets.loc[common_idx].values for name, rets in all_rets.items()}

    strategy_names = list(aligned.keys())
    for i, name_a in enumerate(strategy_names):
        for j, name_b in enumerate(strategy_names):
            if i >= j:
                continue
            test = block_permutation_test(
                aligned[name_a], aligned[name_b],
                block_size=63, n_permutations=10000,
            )
            # Block bootstrap CI on the return difference
            diffs = aligned[name_a] - aligned[name_b]
            ci = block_bootstrap_ci(diffs, block_size=63, n_bootstrap=10000)
            tests[f"{name_a}_vs_{name_b}"] = {
                "diff_means": float(test.statistic),
                "p_value": float(test.p_value),
                "ci_low": float(ci.ci_low),
                "ci_high": float(ci.ci_high),
                "description": test.description,
            }

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
# Results output
# ---------------------------------------------------------------------------

def save_results(
    results: list[BacktestResult],
    comparison: pd.DataFrame,
    stat_tests: dict,
    rebalance_log: list[dict],
    args: argparse.Namespace,
) -> None:
    """Save all backtest results to JSON and parquet."""
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = args.output or str(output_dir / f"backtest_{timestamp}.json")

    output = {
        "config": {
            "start": str(args.start),
            "end": str(args.end),
            "objective": args.objective,
            "max_weight": args.max_weight,
            "max_firms": args.max_firms,
            "lookback_days": args.lookback_days,
            "adv_threshold": args.adv_threshold,
            "adv_lookback": args.adv_lookback,
            "text_source": args.text_source,
            "tc_model": "stratified",
        },
        "metrics": {r.strategy_name: r.metrics for r in results},
        "statistical_tests": stat_tests,
        "rebalance_log": rebalance_log,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", path)

    # Also save return series as parquet for further analysis
    returns_df = pd.DataFrame({r.strategy_name: r.portfolio_returns for r in results})
    returns_path = output_dir / f"returns_{timestamp}.parquet"
    returns_df.to_parquet(returns_path)
    logger.info("Return series saved to %s", returns_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic shrinkage vs Ledoit-Wolf vs 1/N backtest",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2007, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--objective", type=str, default="min_variance",
                        choices=["min_variance", "max_sharpe", "mean_variance", "risk_parity"])
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--max-firms", type=int, default=None)
    parser.add_argument("--lookback-days", type=int, default=504)
    parser.add_argument("--adv-threshold", type=float, default=1_000_000.0,
                        help="Min trailing median daily dollar volume in USD (default: 1M)")
    parser.add_argument("--adv-lookback", type=int, default=21,
                        help="Trading days for ADV calculation (default: 21)")
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

    # ------------------------------------------------------------------
    # 1. Load all data
    # ------------------------------------------------------------------
    all_doc_embeddings = load_doc_embeddings(args.text_source)
    _, daily_returns, dollar_volume = load_price_data(list(all_doc_embeddings.keys()))

    # ------------------------------------------------------------------
    # 2. Generate rebalance dates
    # ------------------------------------------------------------------
    rebalance_dates = generate_quarterly_dates(args.start, args.end)
    logger.info(
        "Backtest: %d quarterly rebalance dates from %s to %s",
        len(rebalance_dates), rebalance_dates[0], rebalance_dates[-1],
    )

    # ------------------------------------------------------------------
    # 3. Config objects
    # ------------------------------------------------------------------
    embedding_config = EmbeddingConfig()
    portfolio_config = PortfolioConfig(
        objective=args.objective,
        long_only=True,
        max_weight=args.max_weight,
    )
    cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
    backtest_config = BacktestConfig(start_date=args.start, end_date=args.end)
    liq_config = LiquidityConfig(
        adv_threshold=args.adv_threshold,
        adv_lookback_days=args.adv_lookback,
        covariance_lookback_days=args.lookback_days,
        max_firms=args.max_firms,
    )
    aggregator = FirmAggregator(embedding_config)
    tc_model = TransactionCostModel()

    # ------------------------------------------------------------------
    # 4. Walk-forward rebalancing loop
    # ------------------------------------------------------------------
    semantic_weights: dict[date, PortfolioWeights] = {}
    lw_weights: dict[date, PortfolioWeights] = {}
    ew_weights: dict[date, PortfolioWeights] = {}
    tc_rates_history: dict[date, pd.Series] = {}
    rebalance_log: list[dict] = []

    for rebal_date in tqdm(rebalance_dates, desc="Rebalancing"):
        # 4a. Aggregate firm embeddings as of this date
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )

        # 4b. Filter to investable universe (shared across all strategies)
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_config,
        )

        if len(eligible) < 20:
            logger.warning("Only %d eligible firms at %s, skipping", len(eligible), rebal_date)
            continue

        # 4b2. Enforce identical universe: intersect with return data + completeness
        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-args.lookback_days:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]

        if len(eligible) < 20:
            logger.warning("Only %d eligible after completeness filter at %s, skipping", len(eligible), rebal_date)
            continue

        # 4c. Compute per-ticker TC rates for this rebalance
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates = tc_model.get_tc_rates(median_adv, daily_returns, str(rebal_date))
        tc_rates_history[rebal_date] = tc_rates

        # 4d. Semantic shrinkage
        sem_ok = False
        try:
            sem_w = compute_semantic_shrinkage_weights(
                firm_embeddings, daily_returns, eligible, rebal_date,
                portfolio_config, cov_config, args.lookback_days,
            )
            semantic_weights[rebal_date] = sem_w
            sem_ok = True
        except Exception as e:
            logger.error("Semantic shrinkage failed at %s: %s", rebal_date, e)

        # 4e. Ledoit-Wolf
        lw_ok = False
        try:
            lw_w = compute_lw_weights(
                daily_returns, eligible, rebal_date,
                portfolio_config, args.lookback_days,
            )
            lw_weights[rebal_date] = lw_w
            lw_ok = True
        except Exception as e:
            logger.error("Ledoit-Wolf failed at %s: %s", rebal_date, e)

        # 4f. Equal weight (always succeeds)
        ew_w = equal_weight_portfolio(eligible)
        ew_weights[rebal_date] = ew_w

        rebalance_log.append({
            "date": str(rebal_date),
            "n_eligible": len(eligible),
            "semantic_positions": int((sem_w.weights.abs() > 1e-6).sum()) if sem_ok else 0,
            "lw_positions": int((lw_w.weights.abs() > 1e-6).sum()) if lw_ok else 0,
            "ew_positions": len(eligible),
        })

    # ------------------------------------------------------------------
    # 5. Run backtests with stratified TC
    # ------------------------------------------------------------------
    engine = BacktestEngine(backtest_config)
    results: list[BacktestResult] = []

    for name, weights_hist in [
        ("semantic_shrinkage", semantic_weights),
        ("ledoit_wolf", lw_weights),
        ("equal_weight", ew_weights),
    ]:
        if not weights_hist:
            logger.warning("No weights for %s, skipping backtest", name)
            continue
        bt = engine.run(
            weights_hist, daily_returns, strategy_name=name,
            tc_rates_history=tc_rates_history,
        )
        results.append(bt)
        logger.info(
            "%s: Sharpe=%.3f, MaxDD=%.3f, AnnRet=%.3f",
            name,
            bt.metrics.get("sharpe_ratio", 0),
            bt.metrics.get("max_drawdown", 0),
            bt.metrics.get("annualized_return", 0),
        )

    if len(results) < 2:
        logger.error("Fewer than 2 strategies completed, cannot compare")
        return

    # ------------------------------------------------------------------
    # 6. Compare and print results
    # ------------------------------------------------------------------
    comparison = BacktestEngine.compare(results)
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(comparison.to_string(float_format="%.4f"))

    # ------------------------------------------------------------------
    # 7. Statistical tests
    # ------------------------------------------------------------------
    stat_tests = run_statistical_tests(results)
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    for test_name, test_result in stat_tests.items():
        if "p_value" in test_result:
            print(f"  {test_name}: diff={test_result['diff_means']:.6f}, p={test_result['p_value']:.4f}")
        elif "corrected_sharpe" in test_result:
            print(f"  {test_name}: raw={test_result['raw_sharpe']:.4f}, "
                  f"eta={test_result['eta']:.4f}, corrected={test_result['corrected_sharpe']:.4f}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 8. Save everything
    # ------------------------------------------------------------------
    save_results(results, comparison, stat_tests, rebalance_log, args)


if __name__ == "__main__":
    main()
