"""
Multi-target shrinkage: combining text, SIC, and identity targets.

Instead of single-target shrinkage Σ = (1-α)S + αT, uses:
    Σ = c₀·S + c₁·T_text + c₂·T_SIC + c₃·I

Based on Lancewicki & Aladjem (IEEE 2014) and Oriol (2024).

The weight decomposition reveals WHEN text vs SIC information dominates:
- At high p/n, c₁ (text) should dominate
- At low p/n, c₂ (SIC) may dominate because coarser structure is more stable

Usage:
    uv run python scripts/run_multitarget_experiment.py
    uv run python scripts/run_multitarget_experiment.py --p-grid "200,500"
    uv run python scripts/run_multitarget_experiment.py --p-grid "50,100,200,500,1000" --text-source 10k
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
from pnlp.baselines.sic_sector import load_sic_codes, sic_sector_portfolio
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
from pnlp.primitives.covariance import (
    MultiTargetShrinkageCovariance,
    SemanticShrinkageCovariance,
)
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
# Single p experiment
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
    sic_codes: dict[str, str],
    lookback_days: int,
) -> dict:
    """Run multi-target experiment at a single universe size."""
    logger.info("=" * 60)
    logger.info("MULTI-TARGET EXPERIMENT: p=%d (p/n=%.2f)", p, p / lookback_days)
    logger.info("=" * 60)

    cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
    tc_model = TransactionCostModel()

    strategies = ["multitarget_cv", "multitarget_frob", "semantic_auto", "sic_sector", "lw", "ew"]
    all_weights: dict[str, dict[date, PortfolioWeights]] = {s: {} for s in strategies}
    tc_rates_history: dict[date, pd.Series] = {}
    weight_log: list[dict] = []

    for rebal_date in tqdm(rebalance_dates, desc=f"p={p}"):
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )

        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_config,
        )
        if len(eligible) < 20:
            continue

        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-lookback_days:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]
        if len(eligible) < 20:
            continue

        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates = tc_model.get_tc_rates(median_adv, daily_returns, str(rebal_date))
        tc_rates_history[rebal_date] = tc_rates

        sub_emb = {t: firm_embeddings[t] for t in eligible}
        sample_stds = lookback_rets.std()
        entry = {"date": str(rebal_date), "n_firms": len(eligible)}

        # --- Multi-target (CV calibration) ---
        try:
            mt_cv_est = MultiTargetShrinkageCovariance(
                config=cov_config,
                historical_returns=lookback_rets,
                sic_codes=sic_codes,
                calibration="cv",
            )
            mt_cv_cov = mt_cv_est.estimate(sub_emb)
            optimizer = PortfolioOptimizer(portfolio_config)
            mt_cv_w = optimizer.optimize(mt_cv_cov)
            mt_cv_w.metadata["method"] = "multitarget_cv"
            mt_cv_w.metadata["weights"] = mt_cv_est.last_weights_
            all_weights["multitarget_cv"][rebal_date] = mt_cv_w
            if mt_cv_est.last_weights_:
                entry["mt_cv_weights"] = mt_cv_est.last_weights_
        except Exception as e:
            logger.error("MultiTarget-CV failed at %s: %s", rebal_date, e)

        # --- Multi-target (Frobenius calibration) ---
        try:
            mt_frob_est = MultiTargetShrinkageCovariance(
                config=cov_config,
                historical_returns=lookback_rets,
                sic_codes=sic_codes,
                calibration="frobenius",
            )
            mt_frob_cov = mt_frob_est.estimate(sub_emb)
            optimizer = PortfolioOptimizer(portfolio_config)
            mt_frob_w = optimizer.optimize(mt_frob_cov)
            mt_frob_w.metadata["method"] = "multitarget_frob"
            mt_frob_w.metadata["weights"] = mt_frob_est.last_weights_
            all_weights["multitarget_frob"][rebal_date] = mt_frob_w
            if mt_frob_est.last_weights_:
                entry["mt_frob_weights"] = mt_frob_est.last_weights_
        except Exception as e:
            logger.error("MultiTarget-Frob failed at %s: %s", rebal_date, e)

        # --- Semantic (single-target, auto) ---
        try:
            sem_est = SemanticShrinkageCovariance(
                config=cov_config,
                historical_returns=lookback_rets,
            )
            sem_cov = sem_est.estimate(sub_emb)
            optimizer = PortfolioOptimizer(portfolio_config)
            sem_w = optimizer.optimize(sem_cov)
            sem_w.metadata["method"] = "semantic_auto"
            all_weights["semantic_auto"][rebal_date] = sem_w
        except Exception as e:
            logger.error("Semantic failed at %s: %s", rebal_date, e)

        # --- SIC sector ---
        try:
            sic_w = sic_sector_portfolio(
                eligible, sic_codes, sigma_estimates=sample_stds,
                objective=portfolio_config.objective,
                long_only=portfolio_config.long_only,
                max_weight=portfolio_config.max_weight,
            )
            all_weights["sic_sector"][rebal_date] = sic_w
        except Exception as e:
            logger.error("SIC sector failed at %s: %s", rebal_date, e)

        # --- LW ---
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
        all_weights["ew"][rebal_date] = equal_weight_portfolio(eligible)

        weight_log.append(entry)

    # --- Run backtests ---
    engine = BacktestEngine(backtest_config)
    results: dict[str, BacktestResult] = {}

    for strat in strategies:
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

    # --- Statistical tests ---
    stat_tests = {}
    if len(results) >= 2:
        result_list = list(results.values())
        all_rets = {r.strategy_name: r.portfolio_returns for r in result_list}
        common_idx = result_list[0].portfolio_returns.index
        for r in result_list[1:]:
            common_idx = common_idx.intersection(r.portfolio_returns.index)
        aligned = {name: rets.loc[common_idx].values for name, rets in all_rets.items()}

        pairs = [
            ("multitarget_cv", "lw"),
            ("multitarget_cv", "semantic_auto"),
            ("multitarget_cv", "sic_sector"),
            ("multitarget_cv", "ew"),
            ("multitarget_cv", "multitarget_frob"),
            ("multitarget_frob", "lw"),
            ("semantic_auto", "lw"),
            ("sic_sector", "lw"),
        ]
        for a, b in pairs:
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

        for strat, bt in results.items():
            if strat in aligned:
                eta = lo_sharpe_correction(aligned[strat], max_lag=63)
                raw = bt.metrics.get("sharpe_ratio", 0.0)
                stat_tests[f"{strat}_lo"] = {
                    "eta": float(eta),
                    "raw_sharpe": float(raw),
                    "corrected_sharpe": float(raw / eta) if eta > 0 else 0.0,
                }

    # Weight decomposition summary for both calibration modes
    weight_summaries = {}
    for mode, key in [("cv", "mt_cv_weights"), ("frobenius", "mt_frob_weights")]:
        entries = [e for e in weight_log if key in e]
        if entries:
            summary = {}
            for w_name in ["sample", "text", "sic", "identity"]:
                vals = [e[key][w_name] for e in entries]
                summary[w_name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            weight_summaries[mode] = summary

    return {
        "p": p,
        "p_over_n": round(p / lookback_days, 2),
        "n_rebalances": len(tc_rates_history),
        "metrics": {strat: bt.metrics for strat, bt in results.items()},
        "statistical_tests": stat_tests,
        "weight_decomposition": weight_summaries,
        "weight_time_series": {
            "cv": [{"date": e["date"], **e["mt_cv_weights"]}
                   for e in weight_log if "mt_cv_weights" in e],
            "frobenius": [{"date": e["date"], **e["mt_frob_weights"]}
                          for e in weight_log if "mt_frob_weights" in e],
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-target shrinkage covariance experiment",
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
    sic_codes = load_sic_codes()

    rebalance_dates = generate_quarterly_dates(args.start, args.end)
    logger.info(
        "Multi-target experiment: p_grid=%s, %d quarters, %s to %s",
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
            sic_codes=sic_codes,
            lookback_days=args.lookback_days,
        )
        all_results.append(result)

    # --- Summary ---
    print("\n" + "=" * 100)
    print("MULTI-TARGET SHRINKAGE RESULTS")
    print("=" * 100)

    for res in all_results:
        p = res["p"]
        print(f"\n--- p={p} (p/n={res['p_over_n']}) ---")

        header = f"{'Strategy':<18} {'Sharpe':>8} {'AnnRet':>8} {'Vol':>8} {'MaxDD':>8} {'vs LW p':>10}"
        print(header)
        print("-" * 70)

        for strat in ["multitarget_cv", "multitarget_frob", "semantic_auto", "sic_sector", "lw", "ew"]:
            m = res["metrics"].get(strat, {})
            if not m:
                continue
            p_val_str = ""
            test = res.get("statistical_tests", {}).get(f"{strat}_vs_lw")
            if test:
                sig = " *" if test["p_value"] < 0.05 else ""
                p_val_str = f"{test['p_value']:10.4f}{sig}"
            elif strat == "lw":
                p_val_str = "    —     "
            print(
                f"{strat:<18} "
                f"{m.get('sharpe_ratio', 0):8.3f} "
                f"{m.get('annualized_return', 0):8.3f} "
                f"{m.get('annualized_volatility', 0):8.3f} "
                f"{m.get('max_drawdown', 0):8.3f} "
                f"{p_val_str}"
            )

        # Weight decomposition for both modes
        wd = res.get("weight_decomposition", {})
        for mode in ["cv", "frobenius"]:
            if mode in wd:
                print(f"\n  Multi-target [{mode}] weight decomposition (mean ± std):")
                for key in ["sample", "text", "sic", "identity"]:
                    if key in wd[mode]:
                        print(f"    {key:>10}: {wd[mode][key]['mean']:.3f} ± {wd[mode][key]['std']:.3f} "
                              f"  [{wd[mode][key]['min']:.3f}, {wd[mode][key]['max']:.3f}]")

    # --- Save ---
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = args.output or str(output_dir / f"multitarget_experiment_{timestamp}.json")

    output = {
        "config": {
            "p_grid": p_grid,
            "start": str(args.start),
            "end": str(args.end),
            "lookback_days": args.lookback_days,
            "text_source": args.text_source,
        },
        "results": all_results,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
