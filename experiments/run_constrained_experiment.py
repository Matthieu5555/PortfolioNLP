"""
Constrained portfolio optimization: Where 1/N fails.

Tests semantic shrinkage, Ledoit-Wolf, SIC sector, and 1/N under realistic
institutional constraints where 1/N is infeasible or suboptimal:

1. Sector limits (max 25% or 15% per 2-digit SIC sector)
2. Volatility targeting (10% annualized)

The key argument: institutional investors CAN'T use unconstrained 1/N. They
have mandates. Under constraints, optimized methods (especially with better
covariance estimates) should dominate.

Usage:
    uv run python experiments/run_constrained_experiment.py
    uv run python experiments/run_constrained_experiment.py --p 200
    uv run python experiments/run_constrained_experiment.py --p 500 --text-source 10k
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

from pnlp.baselines.equal_weight import (
    constrained_equal_weight_portfolio,
    equal_weight_portfolio,
)
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


def _build_sector_limits(
    tickers: list[str],
    sic_codes: dict[str, str],
    max_sector_weight: float,
) -> tuple[dict[str, str], dict[str, float]]:
    """Build ticker_sectors and sector_limits dicts for optimization.

    Returns:
        ticker_sectors: ticker -> 2-digit SIC code
        sector_limits: 2-digit SIC code -> max_sector_weight
    """
    ticker_sectors = {}
    sectors_seen = set()
    for t in tickers:
        sic = sic_codes.get(t)
        if sic:
            ticker_sectors[t] = sic
            sectors_seen.add(sic)
        else:
            # Unknown sector — give unique sector (unconstrained)
            unique = f"_unk_{t}"
            ticker_sectors[t] = unique
            sectors_seen.add(unique)

    sector_limits = {s: max_sector_weight for s in sectors_seen}
    return ticker_sectors, sector_limits


def _sector_weight_summary(
    weights: pd.Series,
    ticker_sectors: dict[str, str],
) -> dict[str, float]:
    """Summarize portfolio weights by sector."""
    sector_wts: dict[str, float] = {}
    for t, w in weights.items():
        sec = ticker_sectors.get(t, "_unknown")
        sector_wts[sec] = sector_wts.get(sec, 0.0) + w
    return sector_wts


# ---------------------------------------------------------------------------
# Constraint regimes
# ---------------------------------------------------------------------------

CONSTRAINT_REGIMES = {
    "unconstrained": {"sector_limit": None, "vol_target": None},
    "sector_25pct": {"sector_limit": 0.25, "vol_target": None},
    "sector_15pct": {"sector_limit": 0.15, "vol_target": None},
    "vol_target_10pct": {"sector_limit": None, "vol_target": 0.10},
    "sector_25pct_vol_10pct": {"sector_limit": 0.25, "vol_target": 0.10},
}


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def run_constrained_backtest(
    regime_name: str,
    sector_limit: float | None,
    vol_target: float | None,
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
    """Run walk-forward backtest under a specific constraint regime."""
    logger.info("=" * 60)
    logger.info("CONSTRAINT REGIME: %s (p=%d)", regime_name, p)
    logger.info("=" * 60)

    cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
    tc_model = TransactionCostModel()

    # Weight histories for each strategy
    strategies = ["semantic", "lw", "ew", "sic_sector"]
    all_weights: dict[str, dict[date, PortfolioWeights]] = {s: {} for s in strategies}
    tc_rates_history: dict[date, pd.Series] = {}
    rebalance_log: list[dict] = []

    for rebal_date in tqdm(rebalance_dates, desc=f"{regime_name}"):
        # Aggregate embeddings
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )

        # Filter universe (shared)
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

        # Build constraint structures
        ticker_sectors = None
        sector_limits = None
        if sector_limit is not None:
            ticker_sectors, sector_limits = _build_sector_limits(
                eligible, sic_codes, sector_limit,
            )

        # TC rates
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates = tc_model.get_tc_rates(median_adv, daily_returns, str(rebal_date))
        tc_rates_history[rebal_date] = tc_rates

        # Sample volatilities for SIC sector
        sample_stds = lookback_rets.std()

        log_entry = {"date": str(rebal_date), "n_eligible": len(eligible)}

        # --- Semantic shrinkage ---
        try:
            sub_emb = {t: firm_embeddings[t] for t in eligible}
            sem_cov_est = SemanticShrinkageCovariance(
                config=cov_config,
                historical_returns=lookback_rets,
            )
            sem_cov = sem_cov_est.estimate(sub_emb, sigma_estimates=None)

            optimizer = PortfolioOptimizer(portfolio_config)
            sem_w = optimizer.optimize(
                sem_cov,
                ticker_sectors=ticker_sectors,
                sector_limits=sector_limits,
                vol_target=vol_target,
            )
            sem_w.metadata["method"] = "semantic_shrinkage"
            all_weights["semantic"][rebal_date] = sem_w
            log_entry["semantic_ok"] = True
        except Exception as e:
            logger.error("Semantic failed at %s: %s", rebal_date, e)
            log_entry["semantic_ok"] = False

        # --- Ledoit-Wolf ---
        try:
            lw_w = ledoit_wolf_portfolio(
                lookback_rets,
                objective=portfolio_config.objective,
                long_only=portfolio_config.long_only,
                max_weight=portfolio_config.max_weight,
                ticker_sectors=ticker_sectors,
                sector_limits=sector_limits,
                vol_target=vol_target,
            )
            all_weights["lw"][rebal_date] = lw_w
            log_entry["lw_ok"] = True
        except Exception as e:
            logger.error("LW failed at %s: %s", rebal_date, e)
            log_entry["lw_ok"] = False

        # --- Equal weight ---
        if sector_limit is not None and ticker_sectors is not None:
            ew_w = constrained_equal_weight_portfolio(
                eligible, ticker_sectors, sector_limits,
            )
        else:
            ew_w = equal_weight_portfolio(eligible)

        # Vol targeting for EW: scale using sample covariance estimate
        if vol_target is not None:
            ew_vals = ew_w.weights.values
            ew_cov = lookback_rets[eligible].fillna(0.0).cov().values
            pred_var = ew_vals @ ew_cov @ ew_vals
            pred_vol = np.sqrt(pred_var * 252)
            if pred_vol > 1e-10:
                scale = vol_target / pred_vol
                ew_w = PortfolioWeights(
                    weights=ew_w.weights * scale,
                    objective_value=ew_w.objective_value,
                    metadata={**ew_w.metadata, "vol_scale": float(scale)},
                )
        all_weights["ew"][rebal_date] = ew_w

        # --- SIC sector ---
        try:
            sic_cov = sic_sector_covariance(
                eligible, sic_codes, sigma_estimates=sample_stds,
            )
            sic_optimizer = PortfolioOptimizer(portfolio_config)
            sic_w = sic_optimizer.optimize(
                sic_cov,
                ticker_sectors=ticker_sectors,
                sector_limits=sector_limits,
                vol_target=vol_target,
            )
            sic_w.metadata["method"] = "sic_sector"
            all_weights["sic_sector"][rebal_date] = sic_w
            log_entry["sic_ok"] = True
        except Exception as e:
            logger.error("SIC sector failed at %s: %s", rebal_date, e)
            log_entry["sic_ok"] = False

        # Log sector concentration
        if ticker_sectors:
            for strat in strategies:
                if rebal_date in all_weights[strat]:
                    sw = _sector_weight_summary(
                        all_weights[strat][rebal_date].weights, ticker_sectors,
                    )
                    log_entry[f"{strat}_max_sector_wt"] = max(sw.values()) if sw else 0.0

        rebalance_log.append(log_entry)

    # --- Run backtests ---
    engine = BacktestEngine(backtest_config)
    results: dict[str, BacktestResult] = {}

    for strat in strategies:
        if not all_weights[strat]:
            continue
        bt = engine.run(
            all_weights[strat], daily_returns,
            strategy_name=f"{strat}_{regime_name}",
            tc_rates_history=tc_rates_history,
        )
        results[strat] = bt
        logger.info(
            "%s/%s: Sharpe=%.3f, MaxDD=%.3f, AnnRet=%.3f, Vol=%.3f",
            regime_name, strat,
            bt.metrics.get("sharpe_ratio", 0),
            bt.metrics.get("max_drawdown", 0),
            bt.metrics.get("annualized_return", 0),
            bt.metrics.get("annualized_volatility", 0),
        )

    # --- Statistical tests ---
    stat_tests = {}
    result_list = list(results.values())
    if len(result_list) >= 2:
        all_rets = {r.strategy_name: r.portfolio_returns for r in result_list}
        common_idx = result_list[0].portfolio_returns.index
        for r in result_list[1:]:
            common_idx = common_idx.intersection(r.portfolio_returns.index)
        aligned = {name: rets.loc[common_idx].values for name, rets in all_rets.items()}

        pairs = [
            ("semantic", "lw"),
            ("semantic", "ew"),
            ("lw", "ew"),
            ("sic_sector", "ew"),
            ("semantic", "sic_sector"),
        ]
        for a, b in pairs:
            key_a = f"{a}_{regime_name}"
            key_b = f"{b}_{regime_name}"
            if key_a in aligned and key_b in aligned:
                test = block_permutation_test(
                    aligned[key_a], aligned[key_b],
                    block_size=63, n_permutations=10000,
                )
                diffs = aligned[key_a] - aligned[key_b]
                ci = block_bootstrap_ci(diffs, block_size=63, n_bootstrap=10000)
                stat_tests[f"{a}_vs_{b}"] = {
                    "diff_means": float(test.statistic),
                    "p_value": float(test.p_value),
                    "ci_low": float(ci.ci_low),
                    "ci_high": float(ci.ci_high),
                }

        # Lo correction
        for strat, bt in results.items():
            name = f"{strat}_{regime_name}"
            if name in aligned:
                eta = lo_sharpe_correction(aligned[name], max_lag=63)
                raw = bt.metrics.get("sharpe_ratio", 0.0)
                stat_tests[f"{strat}_lo"] = {
                    "eta": float(eta),
                    "raw_sharpe": float(raw),
                    "corrected_sharpe": float(raw / eta) if eta > 0 else 0.0,
                }

    return {
        "regime": regime_name,
        "sector_limit": sector_limit,
        "vol_target": vol_target,
        "p": p,
        "n_rebalances": len(rebalance_log),
        "metrics": {strat: bt.metrics for strat, bt in results.items()},
        "statistical_tests": stat_tests,
        "rebalance_log": rebalance_log,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constrained portfolio optimization experiment",
    )
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2007, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--lookback-days", type=int, default=504)
    parser.add_argument("--text-source", type=str, default="10k",
                        choices=["10k", "transcript", "news", "combined"])
    parser.add_argument("--regimes", type=str, default=None,
                        help="Comma-separated regime names (default: all)")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    all_doc_embeddings = load_doc_embeddings(args.text_source)
    _, daily_returns, dollar_volume = load_price_data(list(all_doc_embeddings.keys()))
    sic_codes = load_sic_codes()

    rebalance_dates = generate_quarterly_dates(args.start, args.end)
    logger.info(
        "Constrained experiment: p=%d, %d quarters, %s to %s",
        args.p, len(rebalance_dates), rebalance_dates[0], rebalance_dates[-1],
    )

    embedding_config = EmbeddingConfig()
    portfolio_config = PortfolioConfig(
        objective="min_variance",
        long_only=True,
        max_weight=0.05,
    )
    backtest_config = BacktestConfig(start_date=args.start, end_date=args.end)
    liq_config = LiquidityConfig(
        adv_threshold=1_000_000.0,
        adv_lookback_days=21,
        covariance_lookback_days=args.lookback_days,
        max_firms=args.p,
    )
    aggregator = FirmAggregator(embedding_config)

    # Select regimes
    if args.regimes:
        regime_names = [r.strip() for r in args.regimes.split(",")]
    else:
        regime_names = list(CONSTRAINT_REGIMES.keys())

    # Run each constraint regime
    all_results = []
    for regime_name in regime_names:
        regime = CONSTRAINT_REGIMES[regime_name]
        result = run_constrained_backtest(
            regime_name=regime_name,
            sector_limit=regime["sector_limit"],
            vol_target=regime["vol_target"],
            p=args.p,
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

    # --- Summary table ---
    print("\n" + "=" * 100)
    print("CONSTRAINED OPTIMIZATION RESULTS")
    print("=" * 100)

    header = f"{'Regime':<28} {'Strategy':<14} {'Sharpe':>8} {'AnnRet':>8} {'Vol':>8} {'MaxDD':>8} {'Turn':>8}"
    print(header)
    print("-" * 100)

    for res in all_results:
        for strat in ["semantic", "lw", "ew", "sic_sector"]:
            m = res["metrics"].get(strat, {})
            if not m:
                continue
            print(
                f"{res['regime']:<28} {strat:<14} "
                f"{m.get('sharpe_ratio', 0):8.3f} "
                f"{m.get('annualized_return', 0):8.3f} "
                f"{m.get('annualized_volatility', 0):8.3f} "
                f"{m.get('max_drawdown', 0):8.3f} "
                f"{m.get('avg_turnover', 0):8.3f}"
            )
        print()

    # Stat tests summary
    print("\nSTATISTICAL TESTS (p-values)")
    print("-" * 80)
    for res in all_results:
        for test_name, test_val in res.get("statistical_tests", {}).items():
            if "p_value" in test_val:
                sig = " *" if test_val["p_value"] < 0.05 else ""
                print(f"  {res['regime']:>28} {test_name:<24} p={test_val['p_value']:.4f}{sig}")

    # --- Save ---
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = args.output or str(output_dir / f"constrained_experiment_{timestamp}.json")

    # Strip rebalance_log for compactness (keep only first/last and summary)
    for res in all_results:
        log = res.pop("rebalance_log", [])
        res["n_rebalances_with_data"] = len(log)
        if log:
            res["first_rebalance"] = log[0]
            res["last_rebalance"] = log[-1]

    output = {
        "config": {
            "p": args.p,
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
