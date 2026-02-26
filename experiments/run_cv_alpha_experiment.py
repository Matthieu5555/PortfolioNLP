"""
Cross-validated shrinkage intensity: portfolio-loss calibration.

The LW oracle calibrates alpha to minimize Frobenius loss ‖Σ̂ − Σ‖_F.
But we care about PORTFOLIO loss (realized variance). This mismatch
explains why the oracle recommends ~4.5% when portfolio-optimal is ~75%.

This experiment:
1. Walk-forward backtest with CV-calibrated alpha (nested CV at each rebalance)
2. Tracks the selected alpha over time
3. Compares CV-alpha vs fixed alphas vs LW oracle

Usage:
    uv run python experiments/run_cv_alpha_experiment.py
    uv run python experiments/run_cv_alpha_experiment.py --p 500
    uv run python experiments/run_cv_alpha_experiment.py --p-grid "200,500"
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
from pnlp.primitives.covariance import SemanticShrinkageCovariance
from pnlp.validation.backtest import BacktestEngine, BacktestResult
from pnlp.validation.statistical_tests import (
    block_bootstrap_ci,
    block_permutation_test,
    lo_sharpe_correction,
)
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


# ---------------------------------------------------------------------------
# Single p-value experiment
# ---------------------------------------------------------------------------

def run_single_p(
    p: int,
    all_doc_embeddings: dict,
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    rebalance_dates: list[date],
    aggregator: FirmAggregator,
    portfolio_config: PortfolioConfig,
    backtest_config: BacktestConfig,
    liq_config: LiquidityConfig,
    lookback_days: int,
) -> dict:
    """Run CV-alpha experiment at a single universe size p."""
    logger.info("=" * 60)
    logger.info("CV-ALPHA EXPERIMENT: p=%d (p/n=%.2f)", p, p / lookback_days)
    logger.info("=" * 60)

    tc_model = TransactionCostModel()

    # Weight histories per strategy
    all_weights: dict[str, dict[date, PortfolioWeights]] = {
        s: {} for s in list(STRATEGIES.keys()) + ["lw", "ew"]
    }
    tc_rates_history: dict[date, pd.Series] = {}
    alpha_log: dict[str, list[dict]] = {s: [] for s in STRATEGIES}

    for rebal_date in tqdm(rebalance_dates, desc=f"p={p}"):
        # Aggregate embeddings
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )

        # Filter universe
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_config,
        )
        if len(eligible) < 20:
            continue

        # Enforce identical universe
        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-lookback_days:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]
        if len(eligible) < 20:
            continue

        # TC rates
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates = tc_model.get_tc_rates(median_adv, daily_returns, str(rebal_date))
        tc_rates_history[rebal_date] = tc_rates

        sub_emb = {t: firm_embeddings[t] for t in eligible}

        # --- Text-based strategies ---
        for strat_name, strat_cfg in STRATEGIES.items():
            try:
                cov_config = CovarianceConfig(
                    method="shrinkage",
                    shrinkage_intensity=strat_cfg["intensity"],
                )
                cov_est = SemanticShrinkageCovariance(
                    config=cov_config,
                    historical_returns=lookback_rets,
                )
                cov = cov_est.estimate(sub_emb, sigma_estimates=None)

                optimizer = PortfolioOptimizer(portfolio_config)
                w = optimizer.optimize(cov)
                w.metadata["method"] = strat_name
                w.metadata["alpha"] = cov_est.last_alpha_
                all_weights[strat_name][rebal_date] = w

                alpha_log[strat_name].append({
                    "date": str(rebal_date),
                    "alpha": float(cov_est.last_alpha_) if cov_est.last_alpha_ is not None else None,
                    "n_firms": len(eligible),
                })
            except Exception as e:
                logger.error("%s failed at %s: %s", strat_name, rebal_date, e)

        # --- LW baseline ---
        try:
            lw_w = ledoit_wolf_portfolio(
                lookback_rets,
                objective=portfolio_config.objective,
                long_only=portfolio_config.long_only,
                max_weight=portfolio_config.max_weight,
            )
            all_weights["lw"][rebal_date] = lw_w
        except Exception as e:
            logger.error("LW failed at %s: %s", rebal_date, e)

        # --- EW ---
        ew_w = equal_weight_portfolio(eligible)
        all_weights["ew"][rebal_date] = ew_w

    # --- Run backtests ---
    engine = BacktestEngine(backtest_config)
    results: dict[str, BacktestResult] = {}

    for strat in all_weights:
        if not all_weights[strat]:
            continue
        bt = engine.run(
            all_weights[strat], daily_returns,
            strategy_name=strat,
            tc_rates_history=tc_rates_history,
        )
        results[strat] = bt
        logger.info(
            "p=%d %s: Sharpe=%.3f, AnnRet=%.3f, Vol=%.3f",
            p, strat,
            bt.metrics.get("sharpe_ratio", 0),
            bt.metrics.get("annualized_return", 0),
            bt.metrics.get("annualized_volatility", 0),
        )

    # --- Statistical tests: all strategies vs LW ---
    stat_tests = {}
    if "lw" in results:
        result_list = list(results.values())
        all_rets = {r.strategy_name: r.portfolio_returns for r in result_list}
        common_idx = result_list[0].portfolio_returns.index
        for r in result_list[1:]:
            common_idx = common_idx.intersection(r.portfolio_returns.index)
        aligned = {name: rets.loc[common_idx].values for name, rets in all_rets.items()}

        for strat in list(STRATEGIES.keys()) + ["ew"]:
            if strat in aligned and "lw" in aligned:
                test = block_permutation_test(
                    aligned[strat], aligned["lw"],
                    block_size=63, n_permutations=10000,
                )
                diffs = aligned[strat] - aligned["lw"]
                ci = block_bootstrap_ci(diffs, block_size=63, n_bootstrap=10000)
                stat_tests[f"{strat}_vs_lw"] = {
                    "diff_means": float(test.statistic),
                    "p_value": float(test.p_value),
                    "ci_low": float(ci.ci_low),
                    "ci_high": float(ci.ci_high),
                }

        # CV-alpha vs best fixed alpha
        if "cv_alpha" in aligned:
            for fixed in ["alpha_0.25", "alpha_0.50", "alpha_0.75"]:
                if fixed in aligned:
                    test = block_permutation_test(
                        aligned["cv_alpha"], aligned[fixed],
                        block_size=63, n_permutations=10000,
                    )
                    stat_tests[f"cv_alpha_vs_{fixed}"] = {
                        "diff_means": float(test.statistic),
                        "p_value": float(test.p_value),
                    }

        # Lo correction
        for strat, bt in results.items():
            if strat in aligned:
                eta = lo_sharpe_correction(aligned[strat], max_lag=63)
                raw = bt.metrics.get("sharpe_ratio", 0.0)
                stat_tests[f"{strat}_lo"] = {
                    "eta": float(eta),
                    "raw_sharpe": float(raw),
                    "corrected_sharpe": float(raw / eta) if eta > 0 else 0.0,
                }

    # Alpha time series for CV strategy
    cv_alphas = alpha_log.get("cv_alpha", [])
    alpha_summary = {}
    if cv_alphas:
        alpha_vals = [a["alpha"] for a in cv_alphas if a["alpha"] is not None]
        if alpha_vals:
            alpha_summary = {
                "mean": float(np.mean(alpha_vals)),
                "median": float(np.median(alpha_vals)),
                "std": float(np.std(alpha_vals)),
                "min": float(np.min(alpha_vals)),
                "max": float(np.max(alpha_vals)),
                "q25": float(np.percentile(alpha_vals, 25)),
                "q75": float(np.percentile(alpha_vals, 75)),
            }

    return {
        "p": p,
        "p_over_n": round(p / lookback_days, 2),
        "n_rebalances": len(tc_rates_history),
        "metrics": {strat: bt.metrics for strat, bt in results.items()},
        "statistical_tests": stat_tests,
        "cv_alpha_summary": alpha_summary,
        "alpha_time_series": alpha_log,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validated alpha calibration experiment",
    )
    parser.add_argument("--p-grid", type=str, default="500",
                        help="Comma-separated universe sizes (default: 500)")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2007, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--lookback-days", type=int, default=504)
    parser.add_argument("--text-source", type=str, default="10k",
                        choices=["10k", "transcript", "news", "combined"])
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_grid = [int(x.strip()) for x in args.p_grid.split(",")]

    # Load data
    all_doc_embeddings = load_doc_embeddings(args.text_source)
    _, daily_returns, dollar_volume = load_price_data(list(all_doc_embeddings.keys()))

    rebalance_dates = generate_quarterly_dates(args.start, args.end)
    logger.info(
        "CV-alpha experiment: p_grid=%s, %d quarters, %s to %s",
        p_grid, len(rebalance_dates), rebalance_dates[0], rebalance_dates[-1],
    )

    embedding_config = EmbeddingConfig()
    portfolio_config = PortfolioConfig(
        objective="min_variance",
        long_only=True,
        max_weight=0.05,
    )
    backtest_config = BacktestConfig(start_date=args.start, end_date=args.end)
    aggregator = FirmAggregator(embedding_config)

    all_results = []
    for p in p_grid:
        liq_config = LiquidityConfig(
            adv_threshold=1_000_000.0,
            adv_lookback_days=21,
            covariance_lookback_days=args.lookback_days,
            max_firms=p,
        )
        result = run_single_p(
            p=p,
            all_doc_embeddings=all_doc_embeddings,
            daily_returns=daily_returns,
            dollar_volume=dollar_volume,
            rebalance_dates=rebalance_dates,
            aggregator=aggregator,
            portfolio_config=portfolio_config,
            backtest_config=backtest_config,
            liq_config=liq_config,
            lookback_days=args.lookback_days,
        )
        all_results.append(result)

    # --- Summary ---
    print("\n" + "=" * 90)
    print("CV-ALPHA EXPERIMENT RESULTS")
    print("=" * 90)

    for res in all_results:
        p = res["p"]
        print(f"\n--- p={p} (p/n={res['p_over_n']}) ---")

        header = f"{'Strategy':<18} {'Sharpe':>8} {'AnnRet':>8} {'Vol':>8} {'vs LW p':>10}"
        print(header)
        print("-" * 60)

        for strat in list(STRATEGIES.keys()) + ["lw", "ew"]:
            m = res["metrics"].get(strat, {})
            if not m:
                continue
            p_val_str = ""
            test = res.get("statistical_tests", {}).get(f"{strat}_vs_lw")
            if test:
                p_val_str = f"{test['p_value']:10.4f}"
            elif strat == "lw":
                p_val_str = "    —     "
            print(
                f"{strat:<18} "
                f"{m.get('sharpe_ratio', 0):8.3f} "
                f"{m.get('annualized_return', 0):8.3f} "
                f"{m.get('annualized_volatility', 0):8.3f} "
                f"{p_val_str}"
            )

        # CV-alpha summary
        cv_sum = res.get("cv_alpha_summary", {})
        if cv_sum:
            print(f"\n  CV-alpha distribution: mean={cv_sum['mean']:.3f}, "
                  f"median={cv_sum['median']:.3f}, std={cv_sum['std']:.3f}, "
                  f"range=[{cv_sum['min']:.3f}, {cv_sum['max']:.3f}]")

    # --- Save ---
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = args.output or str(output_dir / f"cv_alpha_experiment_{timestamp}.json")

    output = {
        "config": {
            "p_grid": p_grid,
            "start": str(args.start),
            "end": str(args.end),
            "lookback_days": args.lookback_days,
            "text_source": args.text_source,
            "strategies": STRATEGIES,
        },
        "results": all_results,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
