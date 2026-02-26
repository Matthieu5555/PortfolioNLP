"""
Return History Ablation: The Information Content of Text in Return-Equivalent Units.

THE headline experiment. Varies the lookback window for covariance estimation
from 0 to 504 days and compares text-based, sample-based, and hybrid methods.
Answers: "How many days of return data does a single 10-K filing replace?"

At L=0, return-based methods (LW) are undefined — text still works.
At L=5 with p=500, LW shrinks almost entirely to identity (rank-5 sample).
The crossover point (where text equals LW with L days of returns) quantifies
the information content of text in return-equivalent units.

Key design: universe is filtered using FULL 504-day lookback for ALL strategies.
Same firms, different covariance input. The only variable is the lookback.

Strategies:
  1. text_only:       CosineSimilarityCovariance (no returns, no vol scaling)
  2. text_vols:       CosineSimilarityCovariance + sample vol from L-day returns
  3. semantic:        SemanticShrinkageCovariance with L-day sample cov
  4. lw:              Ledoit-Wolf with L-day returns (undefined at L=0)
  5. sic_sector:      SIC block-diagonal + sample vol from L-day returns
  6. equal_weight:    1/N (unconditional baseline)

Usage:
    uv run python experiments/run_lookback_ablation.py
    uv run python experiments/run_lookback_ablation.py --p 500 --lookback-grid "0,5,10,21,42,63,126,252,504"
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
# Per-lookback weight computation
# ---------------------------------------------------------------------------

def compute_weights_for_lookback(
    L: int,
    eligible: list[str],
    firm_embeddings: dict,
    daily_returns: pd.DataFrame,
    rebal_date: date,
    sic_codes: dict[str, str],
    portfolio_config: PortfolioConfig,
    cov_config: CovarianceConfig,
    full_lookback: int = 504,
) -> dict[str, tuple[PortfolioWeights | None, dict]]:
    """Compute portfolio weights for all strategies at a given lookback.

    Returns dict mapping strategy_name -> (weights, metadata).
    """
    results: dict[str, tuple[PortfolioWeights | None, dict]] = {}
    rebal_str = str(rebal_date)
    optimizer = PortfolioOptimizer(portfolio_config)

    sub_embeddings = {t: firm_embeddings[t] for t in eligible}

    # Get returns for this lookback
    returns_before_T = daily_returns.loc[daily_returns.index < rebal_str]

    if L > 0:
        lookback_returns = returns_before_T[eligible].iloc[-L:].fillna(0.0)
        sample_stds = lookback_returns.std()
        n_obs = len(lookback_returns)
    else:
        lookback_returns = None
        sample_stds = None
        n_obs = 0

    # --- Strategy 1: Pure text (no returns, no vol scaling) ---
    try:
        text_cov = CosineSimilarityCovariance(CovarianceConfig())
        cov = text_cov.estimate(sub_embeddings, sigma_estimates=None)
        w = optimizer.optimize(cov)
        w.metadata["method"] = "text_only"
        results["text_only"] = (w, {"alpha": None})
    except Exception as e:
        logger.warning("text_only failed at L=%d: %s", L, e)
        results["text_only"] = (None, {"error": str(e)})

    # --- Strategy 2: Text + sample vols ---
    try:
        text_cov = CosineSimilarityCovariance(CovarianceConfig())
        cov = text_cov.estimate(sub_embeddings, sigma_estimates=sample_stds)
        w = optimizer.optimize(cov)
        w.metadata["method"] = "text_vols"
        results["text_vols"] = (w, {"alpha": None})
    except Exception as e:
        logger.warning("text_vols failed at L=%d: %s", L, e)
        results["text_vols"] = (None, {"error": str(e)})

    # --- Strategy 3: Semantic shrinkage ---
    try:
        sem_est = SemanticShrinkageCovariance(cov_config, lookback_returns)
        cov = sem_est.estimate(sub_embeddings, sigma_estimates=None)
        w = optimizer.optimize(cov)
        w.metadata["method"] = "semantic"
        results["semantic"] = (w, {"alpha": sem_est.last_alpha_})
    except Exception as e:
        logger.warning("semantic failed at L=%d: %s", L, e)
        results["semantic"] = (None, {"error": str(e)})

    # --- Strategy 4: Ledoit-Wolf ---
    if L >= 10 and lookback_returns is not None:
        try:
            w = ledoit_wolf_portfolio(
                lookback_returns,
                objective=portfolio_config.objective,
                long_only=portfolio_config.long_only,
                max_weight=portfolio_config.max_weight,
            )
            lw_shrinkage = w.metadata.get("shrinkage")
            results["lw"] = (w, {"shrinkage": lw_shrinkage})
        except Exception as e:
            logger.warning("lw failed at L=%d: %s", L, e)
            results["lw"] = (None, {"error": str(e)})
    else:
        # LW undefined at L=0 or very small L
        results["lw"] = (None, {"error": f"undefined_at_L={L}"})

    # --- Strategy 5: SIC sector ---
    try:
        sic_cov = sic_sector_covariance(
            eligible, sic_codes,
            sigma_estimates=sample_stds,
        )
        w = optimizer.optimize(sic_cov)
        w.metadata["method"] = "sic_sector"
        results["sic_sector"] = (w, {})
    except Exception as e:
        logger.warning("sic_sector failed at L=%d: %s", L, e)
        results["sic_sector"] = (None, {"error": str(e)})

    # --- Strategy 6: Equal weight ---
    w = equal_weight_portfolio(eligible)
    results["equal_weight"] = (w, {})

    return results


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(results: list[BacktestResult]) -> dict:
    """Run pairwise block permutation tests and Lo corrections."""
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
            diffs = aligned[name_a] - aligned[name_b]
            ci = block_bootstrap_ci(diffs, block_size=63, n_bootstrap=10000)
            tests[f"{name_a}_vs_{name_b}"] = {
                "diff_means": float(test.statistic),
                "p_value": float(test.p_value),
                "ci_low": float(ci.ci_low),
                "ci_high": float(ci.ci_high),
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
# Main experiment
# ---------------------------------------------------------------------------

def run_lookback_ablation(
    p: int,
    lookback_grid: list[int],
    text_source: str = "10k",
    start: str = "2013-04-01",
    end: str = "2024-04-01",
    full_lookback: int = 504,
    max_weight: float = 0.05,
) -> dict:
    """Run the lookback ablation experiment."""

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    # Load data once
    doc_embeddings = load_doc_embeddings(text_source)
    _, daily_returns, dollar_volume = load_price_data(list(doc_embeddings.keys()))

    rebal_dates = generate_quarterly_dates(start_date, end_date)
    logger.info("%d quarterly rebalance dates from %s to %s",
                len(rebal_dates), rebal_dates[0], rebal_dates[-1])

    aggregator = FirmAggregator(EmbeddingConfig())
    portfolio_config = PortfolioConfig(
        objective="min_variance", long_only=True, max_weight=max_weight,
    )
    cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
    backtest_config = BacktestConfig(start_date=start_date, end_date=end_date)

    # Universe filter uses FULL lookback (same firms for all L)
    liq_config = LiquidityConfig(
        max_firms=p,
        covariance_lookback_days=full_lookback,
    )

    sic_codes = load_sic_codes()
    tc_model = TransactionCostModel()

    # Storage: weights_history[L][strategy] = {rebal_date: PortfolioWeights}
    all_weights: dict[int, dict[str, dict[date, PortfolioWeights]]] = {
        L: {s: {} for s in STRATEGIES}
        for L in lookback_grid
    }
    tc_rates_history: dict[date, pd.Series] = {}
    rebalance_log: list[dict] = []

    for rebal_date in tqdm(rebal_dates, desc=f"p={p} ablation"):
        # Aggregate embeddings (once per rebal_date)
        firm_embeddings = aggregator.aggregate_universe(
            doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )

        # Filter universe using FULL lookback (same firms for all L)
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_config,
        )
        if len(eligible) < 20:
            continue

        # Enforce identical universe: intersect with return data + completeness
        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-full_lookback:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]

        if len(eligible) < 20:
            continue

        # Per-ticker TC rates (once per rebal_date)
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates_history[rebal_date] = tc_model.get_tc_rates(
            median_adv, daily_returns, str(rebal_date),
        )

        log_entry = {
            "date": str(rebal_date),
            "n_eligible": len(eligible),
        }

        # Compute weights for EACH lookback
        for L in lookback_grid:
            try:
                strategy_weights = compute_weights_for_lookback(
                    L, eligible, firm_embeddings, daily_returns, rebal_date,
                    sic_codes, portfolio_config, cov_config, full_lookback,
                )

                for strategy_name, (w, meta) in strategy_weights.items():
                    if w is not None and len(w.weights) > 0:
                        all_weights[L][strategy_name][rebal_date] = w

                # Log oracle alpha for semantic
                sem_meta = strategy_weights.get("semantic", (None, {}))[1]
                log_entry[f"L={L}_semantic_alpha"] = sem_meta.get("alpha")
                log_entry[f"L={L}_lw_shrinkage"] = strategy_weights.get(
                    "lw", (None, {}))[1].get("shrinkage")

            except Exception as e:
                logger.error("Failed at %s L=%d: %s", rebal_date, L, e)

        rebalance_log.append(log_entry)

    # --- Run backtests for each (L, strategy) ---
    engine = BacktestEngine(backtest_config)
    all_results: dict[int, dict] = {}

    for L in lookback_grid:
        logger.info("Running backtests for L=%d", L)
        bt_results: list[BacktestResult] = []

        for strategy_name in STRATEGIES:
            weights_hist = all_weights[L][strategy_name]
            if not weights_hist:
                logger.warning("No weights for L=%d %s, skipping", L, strategy_name)
                continue
            try:
                bt = engine.run(
                    weights_hist, daily_returns,
                    strategy_name=strategy_name,
                    tc_rates_history=tc_rates_history,
                )
                bt_results.append(bt)
            except Exception as e:
                logger.error("Backtest failed for L=%d %s: %s", L, strategy_name, e)

        if len(bt_results) < 2:
            all_results[L] = {"L": L, "error": "insufficient_strategies"}
            continue

        metrics = {r.strategy_name: r.metrics for r in bt_results}
        stat_tests = run_statistical_tests(bt_results)

        all_results[L] = {
            "L": L,
            "p_over_n_effective": p / max(L, 1),
            "metrics": metrics,
            "statistical_tests": stat_tests,
        }

    # --- Build summary table ---
    summary_table = []
    for L in lookback_grid:
        r = all_results.get(L, {})
        if "error" in r:
            summary_table.append({"L": L, "error": r["error"]})
            continue

        metrics = r.get("metrics", {})
        row = {
            "L": L,
            "p_over_n": p / max(L, 1),
        }
        for s in STRATEGIES:
            m = metrics.get(s, {})
            row[f"{s}_sharpe"] = m.get("sharpe_ratio")
            row[f"{s}_annret"] = m.get("annualized_return")
            row[f"{s}_vol"] = m.get("annualized_volatility")
            row[f"{s}_maxdd"] = m.get("max_drawdown")
            row[f"{s}_turnover"] = m.get("turnover")

        # Key comparisons
        tests = r.get("statistical_tests", {})
        for pair in ["text_only_vs_lw", "semantic_vs_lw", "text_vols_vs_lw",
                      "sic_sector_vs_lw", "text_only_vs_sic_sector"]:
            t = tests.get(pair, {})
            row[f"{pair}_pvalue"] = t.get("p_value")

        summary_table.append(row)

    # --- BH-FDR correction across lookback windows ---
    comparison_keys = ["text_vols_vs_lw_pvalue", "semantic_vs_lw_pvalue",
                       "text_only_vs_lw_pvalue", "sic_sector_vs_lw_pvalue",
                       "text_only_vs_sic_sector_pvalue"]
    for key in comparison_keys:
        raw_pvals = [(i, row[key]) for i, row in enumerate(summary_table) if row.get(key) is not None]
        if len(raw_pvals) < 2:
            continue
        indices, pvals = zip(*raw_pvals)
        adjusted = _bh_fdr(list(pvals))
        bh_key = key.replace("_pvalue", "_pvalue_bh")
        for idx, adj_p in zip(indices, adjusted):
            summary_table[idx][bh_key] = adj_p

    results = {
        "experiment": "lookback_ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "p": p,
            "lookback_grid": lookback_grid,
            "text_source": text_source,
            "start": start,
            "end": end,
            "full_lookback": full_lookback,
            "max_weight": max_weight,
            "strategies": STRATEGIES,
        },
        "summary_table": summary_table,
        "full_results": {str(L): r for L, r in all_results.items()},
        "rebalance_log": rebalance_log,
    }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Return history lookback ablation")
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--lookback-grid", type=str, default="0,5,10,21,42,63,126,252,504")
    parser.add_argument("--text-source", default="10k",
                        choices=["10k", "transcript", "news", "combined", "all"])
    parser.add_argument("--start", default="2007-04-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--max-weight", type=float, default=0.05)
    args = parser.parse_args()

    lookback_grid = [int(x) for x in args.lookback_grid.split(",")]

    t0 = time.time()
    results = run_lookback_ablation(
        p=args.p,
        lookback_grid=lookback_grid,
        text_source=args.text_source,
        start=args.start,
        end=args.end,
        max_weight=args.max_weight,
    )

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / "results" / f"lookback_ablation_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info("Done in %.1f seconds. Results saved to %s", elapsed, out_path)

    # Print summary table
    print(f"\n{'='*120}")
    print(f"LOOKBACK ABLATION: p={args.p}, {args.text_source}")
    print(f"{'='*120}")

    header = (
        f"{'L':>5}  {'p/n':>6}  "
        f"{'text_only':>10}  {'text_vols':>10}  {'semantic':>10}  "
        f"{'lw':>10}  {'sic_sector':>10}  {'EW':>10}"
    )
    print(f"\nSharpe Ratios:")
    print(header)
    print("-" * 90)

    for row in results["summary_table"]:
        if "error" in row:
            print(f"{row['L']:>5}  ERROR")
            continue
        line = (
            f"{row['L']:>5}  "
            f"{row.get('p_over_n', 0):>6.1f}  "
        )
        for s in STRATEGIES:
            sharpe = row.get(f"{s}_sharpe")
            if sharpe is not None:
                line += f"{sharpe:>10.3f}  "
            else:
                line += f"{'N/A':>10}  "
        print(line)

    # Highlight crossover
    print(f"\n{'='*120}")
    print("CROSSOVER ANALYSIS:")
    text_only_sharpes = {
        row["L"]: row.get("text_only_sharpe")
        for row in results["summary_table"]
        if "error" not in row
    }
    text_only_sharpe = text_only_sharpes.get(0)  # L=0 text performance
    if text_only_sharpe is not None:
        print(f"  Text-only (L=0) Sharpe: {text_only_sharpe:.3f}")
        for row in results["summary_table"]:
            if "error" in row:
                continue
            lw_sharpe = row.get("lw_sharpe")
            if lw_sharpe is not None and lw_sharpe >= text_only_sharpe:
                print(f"  LW matches text at L={row['L']} days "
                      f"(LW Sharpe={lw_sharpe:.3f})")
                break
        else:
            print("  LW never matches text-only (text dominates at all lookbacks tested)")

    print(f"{'='*120}")


if __name__ == "__main__":
    main()
