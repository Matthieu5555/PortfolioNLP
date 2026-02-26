"""
Alpha sweep + cold-start: Can text embeddings substitute for return history?

Tests the full spectrum from pure sample covariance (alpha=0) to pure text
covariance (alpha=1.0) across universe sizes. Also tests true cold-start:
text-only covariance with NO return history at all.

Hypothesis: At high p/n, optimal alpha is much higher than the LW oracle
prescribes. Pure text (alpha=1.0) may be competitive with LW at p=2000
because the sample covariance is mostly noise.

Usage:
    uv run python experiments/run_alpha_sweep.py
    uv run python experiments/run_alpha_sweep.py --p-grid "200,500"
    uv run python experiments/run_alpha_sweep.py --p-grid "200,500,1000,2000" --text-source 10k
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
from pnlp.embeddings.firm_aggregator import FirmAggregator, FirmEmbedding
from pnlp.portfolio.optimizer import PortfolioOptimizer, PortfolioWeights
from pnlp.primitives.covariance import CosineSimilarityCovariance, SemanticShrinkageCovariance
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

def _bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_min = 1.0
    for rank in range(n, 0, -1):
        idx, p = indexed[rank - 1]
        adj = min(p * n / rank, running_min, 1.0)
        adjusted[idx] = adj
        running_min = adj
    return adjusted



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
    """Run alpha sweep + cold-start at a single universe size p."""
    logger.info("=" * 60)
    logger.info("ALPHA SWEEP: p=%d (p/n=%.2f)", p, p / lookback_days)
    logger.info("=" * 60)

    tc_model = TransactionCostModel()

    # Override max_firms for this p value
    liq_p = LiquidityConfig(
        adv_threshold=liq_config.adv_threshold,
        adv_lookback_days=liq_config.adv_lookback_days,
        min_return_completeness=liq_config.min_return_completeness,
        covariance_lookback_days=liq_config.covariance_lookback_days,
        max_firms=p,
    )

    # Weight histories per strategy
    strategy_weights: dict[str, dict[date, PortfolioWeights]] = {}
    for alpha in ALPHA_GRID:
        label = f"alpha_{alpha}" if isinstance(alpha, float) else "alpha_auto"
        strategy_weights[label] = {}
    strategy_weights["cold_start_text_vol"] = {}   # cosine corr + sample stds
    strategy_weights["cold_start_pure_text"] = {}   # cosine corr only (no returns at all)
    strategy_weights["ledoit_wolf"] = {}
    strategy_weights["equal_weight"] = {}

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

        # Enforce identical universe
        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-lookback_days:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]
        if len(eligible) < 20:
            continue

        # Shared computations
        sub_embeddings = {t: firm_embeddings[t] for t in eligible}
        lookback_returns = lookback_rets[eligible]

        # TC rates (shared across all strategies)
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_p.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates = tc_model.get_tc_rates(median_adv, daily_returns, str(rebal_date))
        tc_rates_history[rebal_date] = tc_rates

        # Sample stds for cold-start-with-vol variant
        sample_stds = lookback_returns.fillna(0.0).std()

        log_entry: dict = {"date": str(rebal_date), "p": p, "n_eligible": len(eligible)}

        # --- Alpha sweep ---
        for alpha in ALPHA_GRID:
            label = f"alpha_{alpha}" if isinstance(alpha, float) else "alpha_auto"
            try:
                cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity=alpha)
                cov_estimator = SemanticShrinkageCovariance(
                    config=cov_config, historical_returns=lookback_returns,
                )
                cov = cov_estimator.estimate(sub_embeddings, sigma_estimates=None)
                if label == "alpha_auto":
                    log_entry["auto_alpha"] = cov_estimator.last_alpha_
                optimizer = PortfolioOptimizer(portfolio_config)
                w = optimizer.optimize(cov)
                w.metadata["method"] = label
                strategy_weights[label][rebal_date] = w
            except Exception as e:
                logger.error("%s failed at %s p=%d: %s", label, rebal_date, p, e)

        # --- Cold-start: text correlation + sample volatilities ---
        try:
            cov_config_cs = CovarianceConfig()
            cs_estimator = CosineSimilarityCovariance(config=cov_config_cs)
            cov_text_vol = cs_estimator.estimate(sub_embeddings, sigma_estimates=sample_stds)
            optimizer = PortfolioOptimizer(portfolio_config)
            w = optimizer.optimize(cov_text_vol)
            w.metadata["method"] = "cold_start_text_vol"
            strategy_weights["cold_start_text_vol"][rebal_date] = w
        except Exception as e:
            logger.error("cold_start_text_vol failed at %s p=%d: %s", rebal_date, p, e)

        # --- Cold-start: pure text (no return data at all) ---
        try:
            cs_pure = CosineSimilarityCovariance(config=CovarianceConfig())
            cov_pure = cs_pure.estimate(sub_embeddings, sigma_estimates=None)
            optimizer = PortfolioOptimizer(portfolio_config)
            w = optimizer.optimize(cov_pure)
            w.metadata["method"] = "cold_start_pure_text"
            strategy_weights["cold_start_pure_text"][rebal_date] = w
        except Exception as e:
            logger.error("cold_start_pure_text failed at %s p=%d: %s", rebal_date, p, e)

        # --- Ledoit-Wolf baseline ---
        try:
            lw_w = ledoit_wolf_portfolio(
                lookback_returns,
                objective=portfolio_config.objective,
                long_only=portfolio_config.long_only,
                max_weight=portfolio_config.max_weight,
            )
            strategy_weights["ledoit_wolf"][rebal_date] = lw_w
            log_entry["lw_shrinkage"] = lw_w.metadata.get("shrinkage")
        except Exception as e:
            logger.error("LW failed at %s p=%d: %s", rebal_date, p, e)

        # --- Equal weight ---
        ew_w = equal_weight_portfolio(eligible)
        strategy_weights["equal_weight"][rebal_date] = ew_w

        rebalance_log.append(log_entry)

    # ------------------------------------------------------------------
    # Backtest all strategies
    # ------------------------------------------------------------------
    engine = BacktestEngine(backtest_config)
    results: dict[str, BacktestResult] = {}

    for name, weights_hist in strategy_weights.items():
        if not weights_hist:
            logger.warning("No weights for %s at p=%d", name, p)
            continue
        bt = engine.run(
            weights_hist, daily_returns, strategy_name=name,
            tc_rates_history=tc_rates_history,
        )
        results[name] = bt

    # ------------------------------------------------------------------
    # Statistical tests (each alpha vs LW, cold-start vs LW)
    # ------------------------------------------------------------------
    stat_tests = {}
    lw_result = results.get("ledoit_wolf")
    if lw_result is not None:
        lw_rets = lw_result.portfolio_returns

        for name, bt in results.items():
            if name == "ledoit_wolf" or name == "equal_weight":
                continue
            other_rets = bt.portfolio_returns
            common = lw_rets.index.intersection(other_rets.index)
            if len(common) < 100:
                continue

            lw_vals = lw_rets.loc[common].values
            other_vals = other_rets.loc[common].values

            test = block_permutation_test(other_vals, lw_vals, block_size=63, n_permutations=10000)
            diffs = other_vals - lw_vals
            ci = block_bootstrap_ci(diffs, block_size=63, n_bootstrap=10000)

            stat_tests[f"{name}_vs_lw"] = {
                "diff_means": float(test.statistic),
                "p_value": float(test.p_value),
                "ci_low": float(ci.ci_low),
                "ci_high": float(ci.ci_high),
            }

    # Lo corrections
    for name, bt in results.items():
        rets = bt.portfolio_returns.values
        eta = lo_sharpe_correction(rets, max_lag=63)
        raw_sharpe = bt.metrics.get("sharpe_ratio", 0.0)
        stat_tests[f"{name}_lo_correction"] = {
            "eta": float(eta),
            "raw_sharpe": float(raw_sharpe),
            "corrected_sharpe": float(raw_sharpe / eta) if eta > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # BH-FDR correction across alpha comparisons vs LW
    # ------------------------------------------------------------------
    vs_lw_keys = [k for k in stat_tests if k.endswith("_vs_lw")]
    if len(vs_lw_keys) >= 2:
        raw_pvals = [(k, stat_tests[k]["p_value"]) for k in vs_lw_keys]
        keys, pvals = zip(*raw_pvals)
        adjusted = _bh_fdr(list(pvals))
        for k, adj_p in zip(keys, adjusted):
            stat_tests[k]["p_value_bh"] = adj_p

    # ------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------
    metrics_summary = {}
    for name, bt in results.items():
        metrics_summary[name] = {
            "sharpe": bt.metrics.get("sharpe_ratio"),
            "ann_return": bt.metrics.get("annualized_return"),
            "ann_vol": bt.metrics.get("annualized_volatility"),
            "max_drawdown": bt.metrics.get("max_drawdown"),
            "avg_turnover": bt.metrics.get("avg_turnover"),
        }

    auto_alphas = [e.get("auto_alpha") for e in rebalance_log if e.get("auto_alpha") is not None]

    summary = {
        "p": p,
        "p_over_n": round(p / lookback_days, 3),
        "avg_auto_alpha": round(float(np.mean(auto_alphas)), 6) if auto_alphas else None,
        "n_rebalances": len(rebalance_log),
        "metrics": metrics_summary,
        "statistical_tests": stat_tests,
        "rebalance_log": rebalance_log,
    }

    # Print interim results
    print(f"\n{'─' * 70}")
    print(f"  p={p} (p/n={p / lookback_days:.2f})")
    print(f"{'─' * 70}")
    print(f"  {'Strategy':<25} {'Sharpe':>8}  {'AnnRet':>8}  {'AnnVol':>8}  {'vs LW p':>8}")
    print(f"  {'─' * 55}")
    for name in ["alpha_0.0", "alpha_0.25", "alpha_0.5", "alpha_0.75", "alpha_1.0",
                  "alpha_auto", "cold_start_text_vol", "cold_start_pure_text",
                  "ledoit_wolf", "equal_weight"]:
        if name not in metrics_summary:
            continue
        m = metrics_summary[name]
        p_val_str = ""
        test_key = f"{name}_vs_lw"
        if test_key in stat_tests:
            p_val_str = f"{stat_tests[test_key]['p_value']:>8.4f}"
        print(f"  {name:<25} {m['sharpe'] or 0:>8.4f}  {m['ann_return'] or 0:>8.4f}  "
              f"{m['ann_vol'] or 0:>8.4f}  {p_val_str}")
    print(f"{'─' * 70}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha sweep + cold-start: can text substitute for return history?",
    )
    parser.add_argument("--p-grid", type=str, default="200,500,1000,2000",
                        help="Comma-separated universe sizes (default: 200,500,1000,2000)")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2007, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--objective", type=str, default="min_variance",
                        choices=["min_variance", "max_sharpe", "mean_variance", "risk_parity"])
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
    backtest_config = BacktestConfig(start_date=args.start, end_date=args.end)
    liq_config = LiquidityConfig(
        adv_threshold=args.adv_threshold,
        adv_lookback_days=args.adv_lookback,
        covariance_lookback_days=args.lookback_days,
    )
    aggregator = FirmAggregator(embedding_config)

    # Run experiment for each p (smallest first for quick results)
    all_results: list[dict] = []

    for p in sorted(p_grid):
        t0 = time.time()
        result = run_single_p(
            p, all_doc_embeddings, daily_returns, dollar_volume,
            rebalance_dates, aggregator, portfolio_config,
            backtest_config, liq_config, args.lookback_days,
        )
        result["runtime_seconds"] = round(time.time() - t0, 1)
        all_results.append(result)

    # Print final summary table
    print("\n" + "=" * 120)
    print("ALPHA SWEEP SUMMARY")
    print("=" * 120)
    print(f"{'p':>6}  {'alpha':>8}  {'sharpe':>8}  {'ann_ret':>8}  {'ann_vol':>8}  {'vs_lw_p':>8}  {'strategy':<25}")
    print("-" * 120)

    for r in all_results:
        p = r["p"]
        for name in ["alpha_0.0", "alpha_0.25", "alpha_0.5", "alpha_0.75", "alpha_1.0",
                      "alpha_auto", "cold_start_text_vol", "cold_start_pure_text",
                      "ledoit_wolf", "equal_weight"]:
            m = r["metrics"].get(name)
            if m is None:
                continue
            p_val = r["statistical_tests"].get(f"{name}_vs_lw", {}).get("p_value", "")
            p_str = f"{p_val:>8.4f}" if isinstance(p_val, float) else f"{'':>8}"
            alpha_str = name.replace("alpha_", "").replace("cold_start_", "cs_")
            print(
                f"{p:>6}  {alpha_str:>8}  "
                f"{m['sharpe'] or 0:>8.4f}  {m['ann_return'] or 0:>8.4f}  "
                f"{m['ann_vol'] or 0:>8.4f}  {p_str}  {name:<25}"
            )
        print("-" * 120)
    print("=" * 120)

    # Save results
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"alpha_sweep_{ts}.json"

    output = {
        "experiment": "alpha_sweep",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "p_grid": p_grid,
            "alpha_grid": [str(a) for a in ALPHA_GRID],
            "cold_start_variants": ["text_vol", "pure_text"],
            "start": str(args.start),
            "end": str(args.end),
            "objective": args.objective,
            "max_weight": args.max_weight,
            "lookback_days": args.lookback_days,
            "text_source": args.text_source,
        },
        "results": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Results saved to %s", out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
