"""
PCA factor model experiment: How many "text factors" are needed?

Tests whether a rank-k PCA approximation of the cosine similarity matrix
works as well as the full cosine matrix as a shrinkage target. If k=10
matches k=768, the useful structure in the embedding space is ~10-dimensional
(industry clusters), and the remaining dimensions are noise.

Usage:
    uv run python experiments/run_pca_experiment.py
    uv run python experiments/run_pca_experiment.py --k-grid 1,3,5,10,20,50,100
    uv run python experiments/run_pca_experiment.py --k-grid 5,10,20 --max-firms 1000
    uv run python experiments/run_pca_experiment.py --text-source transcript
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
from pnlp.primitives.covariance import PCAFactorCovariance, SemanticShrinkageCovariance
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
# Helpers
# ---------------------------------------------------------------------------


def run_statistical_tests(results: list[BacktestResult]) -> dict:
    tests: dict[str, dict] = {}
    all_rets = {r.strategy_name: r.portfolio_returns for r in results}
    common_idx = all_rets[results[0].strategy_name].index
    for r in results[1:]:
        common_idx = common_idx.intersection(all_rets[r.strategy_name].index)
    aligned = {name: rets.loc[common_idx].values for name, rets in all_rets.items()}

    # Only test key pairs (each PCA-k vs LW, semantic_full vs LW)
    lw_key = "ledoit_wolf"
    if lw_key not in aligned:
        return tests

    from statsmodels.stats.multitest import multipletests

    pair_keys = []
    raw_pvals = []
    for name in aligned:
        if name == lw_key or name == "equal_weight":
            continue
        test = block_permutation_test(
            aligned[name], aligned[lw_key],
            block_size=63, n_permutations=10000,
        )
        diffs = aligned[name] - aligned[lw_key]
        ci = block_bootstrap_ci(diffs, block_size=63, n_bootstrap=10000)
        key = f"{name}_vs_{lw_key}"
        tests[key] = {
            "diff_means": float(test.statistic),
            "p_value": float(test.p_value),
            "ci_low": float(ci.ci_low),
            "ci_high": float(ci.ci_high),
        }
        pair_keys.append(key)
        raw_pvals.append(test.p_value)

    # BH-FDR correction across all PCA-k vs LW tests
    if len(raw_pvals) > 1:
        _, bh_adjusted, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
        for key, adj_p in zip(pair_keys, bh_adjusted):
            tests[key]["p_value_bh"] = round(float(adj_p), 4)

    # Lo correction for all
    for r in results:
        name = r.strategy_name
        rets = aligned.get(name)
        if rets is None:
            continue
        eta = lo_sharpe_correction(rets, max_lag=63)
        raw_sharpe = r.metrics.get("sharpe_ratio", 0.0)
        tests[f"{name}_lo_correction"] = {
            "eta": float(eta),
            "raw_sharpe": float(raw_sharpe),
            "corrected_sharpe": float(raw_sharpe / eta) if eta > 0 else 0.0,
        }

    return tests


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA factor model experiment: k-sweep for embedding PCA factors",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2007, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--k-grid", type=str, default="1,3,5,10,20,50,100",
                        help="Comma-separated PCA factor counts (default: 1,3,5,10,20,50,100)")
    parser.add_argument("--max-firms", type=int, default=500)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--lookback-days", type=int, default=504)
    parser.add_argument("--adv-threshold", type=float, default=1_000_000.0)
    parser.add_argument("--adv-lookback", type=int, default=21)
    parser.add_argument("--text-source", type=str, default="10k",
                        choices=["10k", "transcript", "news", "combined", "all"])
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    k_grid = [int(k) for k in args.k_grid.split(",")]

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    all_doc_embeddings = load_doc_embeddings(args.text_source)
    _, daily_returns, dollar_volume = load_price_data(list(all_doc_embeddings.keys()))

    rebalance_dates = generate_quarterly_dates(args.start, args.end)
    logger.info(
        "PCA experiment: k_grid=%s, max_firms=%d, %d rebalance dates (%s to %s)",
        k_grid, args.max_firms, len(rebalance_dates),
        rebalance_dates[0], rebalance_dates[-1],
    )

    # ------------------------------------------------------------------
    # 2. Config
    # ------------------------------------------------------------------
    embedding_config = EmbeddingConfig()
    portfolio_config = PortfolioConfig(
        objective="min_variance", long_only=True, max_weight=args.max_weight,
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
    optimizer = PortfolioOptimizer(portfolio_config)

    # ------------------------------------------------------------------
    # 3. Walk-forward loop: all PCA-k + semantic + LW + EW per rebalance
    # ------------------------------------------------------------------
    # Strategy names
    pca_names = [f"pca_k{k}" for k in k_grid]
    all_strategy_names = pca_names + ["semantic_full", "ledoit_wolf", "equal_weight"]

    all_weights: dict[str, dict[date, PortfolioWeights]] = {
        name: {} for name in all_strategy_names
    }
    tc_rates_history: dict[date, pd.Series] = {}
    eigenvalue_log: list[dict] = []
    rebalance_log: list[dict] = []

    for rebal_date in tqdm(rebalance_dates, desc="Rebalancing"):
        # 3a. Aggregate embeddings
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )

        # 3b. Filter universe (shared)
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_config,
        )
        if len(eligible) < 20:
            continue

        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-args.lookback_days:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]

        if len(eligible) < 20:
            continue

        sub_embeddings = {t: firm_embeddings[t] for t in eligible}
        lookback_returns = returns_before_T[eligible].iloc[-args.lookback_days:]

        # 3c. TC rates
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates_history[rebal_date] = tc_model.get_tc_rates(
            median_adv, daily_returns, str(rebal_date),
        )

        log_entry: dict = {"date": str(rebal_date), "n_eligible": len(eligible)}

        # 3d. Full SVD (once) for eigenvalue analysis
        E = np.array([sub_embeddings[t].embedding for t in sorted(sub_embeddings.keys())])
        _, s, _ = np.linalg.svd(E, full_matrices=False)
        s_sq = s ** 2
        total_var = s_sq.sum()
        cumulative = np.cumsum(s_sq) / total_var

        ev_entry = {
            "date": str(rebal_date),
            "n_firms": len(eligible),
            "n_components": len(s),
        }
        for k in k_grid:
            k_eff = min(k, len(s))
            ev_entry[f"top_{k}_explained"] = float(cumulative[k_eff - 1])
        ev_entry["top_10_explained"] = float(cumulative[min(9, len(s) - 1)])
        ev_entry["top_20_explained"] = float(cumulative[min(19, len(s) - 1)])
        eigenvalue_log.append(ev_entry)

        # 3e. PCA-k strategies (all k values, single SVD)
        for k in k_grid:
            try:
                pca_est = PCAFactorCovariance(
                    config=cov_config, n_factors=k, historical_returns=lookback_returns,
                )
                cov = pca_est.estimate(sub_embeddings)
                w = optimizer.optimize(cov)
                w.metadata["method"] = f"pca_k{k}"
                all_weights[f"pca_k{k}"][rebal_date] = w
                log_entry[f"pca_k{k}_alpha"] = pca_est.last_alpha_
                log_entry[f"pca_k{k}_explained"] = float(pca_est.explained_variance_ratio_.sum())
            except Exception as e:
                logger.error("PCA k=%d failed at %s: %s", k, rebal_date, e)

        # 3f. Full semantic (same as SemanticShrinkageCovariance)
        try:
            sem_est = SemanticShrinkageCovariance(
                config=cov_config, historical_returns=lookback_returns,
            )
            cov = sem_est.estimate(sub_embeddings)
            w = optimizer.optimize(cov)
            w.metadata["method"] = "semantic_full"
            all_weights["semantic_full"][rebal_date] = w
            log_entry["semantic_alpha"] = sem_est.last_alpha_
        except Exception as e:
            logger.error("Semantic failed at %s: %s", rebal_date, e)

        # 3g. Ledoit-Wolf
        try:
            lw_w = ledoit_wolf_portfolio(
                lookback_returns,
                objective=portfolio_config.objective,
                long_only=portfolio_config.long_only,
                max_weight=portfolio_config.max_weight,
            )
            all_weights["ledoit_wolf"][rebal_date] = lw_w
        except Exception as e:
            logger.error("LW failed at %s: %s", rebal_date, e)

        # 3h. Equal weight
        ew_w = equal_weight_portfolio(eligible)
        all_weights["equal_weight"][rebal_date] = ew_w

        rebalance_log.append(log_entry)

    # ------------------------------------------------------------------
    # 4. Backtest all strategies
    # ------------------------------------------------------------------
    engine = BacktestEngine(backtest_config)
    results: list[BacktestResult] = []

    for name in all_strategy_names:
        weights_hist = all_weights[name]
        if not weights_hist:
            logger.warning("No weights for %s, skipping", name)
            continue
        bt = engine.run(
            weights_hist, daily_returns, strategy_name=name,
            tc_rates_history=tc_rates_history,
        )
        results.append(bt)
        logger.info(
            "%s: Sharpe=%.3f, AnnRet=%.3f, Vol=%.3f",
            name,
            bt.metrics.get("sharpe_ratio", 0),
            bt.metrics.get("annualized_return", 0),
            bt.metrics.get("annualized_volatility", 0),
        )

    if len(results) < 2:
        logger.error("Fewer than 2 strategies completed")
        return

    # ------------------------------------------------------------------
    # 5. Statistical tests (each PCA-k vs LW)
    # ------------------------------------------------------------------
    stat_tests = run_statistical_tests(results)

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PCA FACTOR MODEL EXPERIMENT")
    print("=" * 80)

    print(f"\nConfig: max_firms={args.max_firms}, text_source={args.text_source}")
    print(f"k-grid: {k_grid}")

    # Eigenvalue analysis summary
    if eigenvalue_log:
        avg_ev = {}
        for k in k_grid:
            key = f"top_{k}_explained"
            vals = [e[key] for e in eigenvalue_log if key in e]
            avg_ev[k] = np.mean(vals) if vals else 0
        print("\nAverage explained variance by k:")
        for k, ev in avg_ev.items():
            print(f"  k={k:4d}: {ev:.3f}")

    print("\nSharpe ratios:")
    for r in results:
        sharpe = r.metrics.get("sharpe_ratio", 0)
        ann_ret = r.metrics.get("annualized_return", 0)
        ann_vol = r.metrics.get("annualized_volatility", 0)
        turnover = r.metrics.get("avg_turnover", 0)
        print(f"  {r.strategy_name:20s}: Sharpe={sharpe:.3f}  AnnRet={ann_ret:.3f}  Vol={ann_vol:.3f}  Turn={turnover:.3f}")

    print("\nStatistical tests (vs LW):")
    for test_name, test_result in stat_tests.items():
        if "p_value" in test_result:
            print(f"  {test_name}: diff={test_result['diff_means']:.6f}, p={test_result['p_value']:.4f}")

    print("=" * 80)

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = args.output or str(output_dir / f"pca_experiment_{timestamp}.json")

    # Compute average eigenvalue summary
    avg_explained = {}
    if eigenvalue_log:
        for k in k_grid:
            key = f"top_{k}_explained"
            vals = [e[key] for e in eigenvalue_log if key in e]
            avg_explained[str(k)] = float(np.mean(vals)) if vals else 0.0

    output = {
        "config": {
            "start": str(args.start),
            "end": str(args.end),
            "max_firms": args.max_firms,
            "max_weight": args.max_weight,
            "lookback_days": args.lookback_days,
            "text_source": args.text_source,
            "k_grid": k_grid,
            "tc_model": "stratified",
        },
        "eigenvalue_analysis": {
            "average_explained_variance_by_k": avg_explained,
            "per_quarter": eigenvalue_log,
        },
        "k_sweep": {},
        "baselines": {},
        "statistical_tests": stat_tests,
        "rebalance_log": rebalance_log,
    }

    for r in results:
        name = r.strategy_name
        entry = {
            "sharpe": r.metrics.get("sharpe_ratio", 0),
            "annualized_return": r.metrics.get("annualized_return", 0),
            "annualized_volatility": r.metrics.get("annualized_volatility", 0),
            "max_drawdown": r.metrics.get("max_drawdown", 0),
            "turnover": r.metrics.get("avg_turnover", 0),
            "sortino": r.metrics.get("sortino_ratio", 0),
        }
        if name.startswith("pca_k"):
            k = int(name.split("k")[1])
            entry["gap_vs_lw"] = entry["sharpe"] - output.get("baselines", {}).get("ledoit_wolf", {}).get("sharpe", 0)
            output["k_sweep"][str(k)] = entry
        else:
            output["baselines"][name] = entry

    # Compute gaps after baselines are filled
    lw_sharpe = output["baselines"].get("ledoit_wolf", {}).get("sharpe", 0)
    for k_str, entry in output["k_sweep"].items():
        entry["gap_vs_lw"] = entry["sharpe"] - lw_sharpe

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", path)

    # Save return series
    returns_df = pd.DataFrame({r.strategy_name: r.portfolio_returns for r in results})
    returns_path = output_dir / f"returns_pca_{timestamp}.parquet"
    returns_df.to_parquet(returns_path)
    logger.info("Return series saved to %s", returns_path)


if __name__ == "__main__":
    main()
