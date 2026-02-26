"""
Robustness checks for the semantic shrinkage portfolio.

Phase 1 (fast, ~1 min): Uses saved return series.
  - Fama-French 5-factor + Momentum regression
  - SPY benchmark comparison
  - Skewness / kurtosis

Phase 2 (slow, ~30 min total): Requires re-running the walk-forward loop.
  - Shuffle test: randomize ticker-embedding mapping, confirm signal disappears.
  - TC sensitivity: re-run backtest engine with different TC multipliers.

Usage:
    uv run python experiments/run_robustness.py                          # Phase 1 only (fast)
    uv run python experiments/run_robustness.py --text-source transcript  # use transcripts
    uv run python experiments/run_robustness.py --run-shuffle             # + shuffle test
    uv run python experiments/run_robustness.py --run-tc-sensitivity      # + TC sweep
    uv run python experiments/run_robustness.py --all                     # everything
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
from pnlp.config import (
    DATA_DIR,
    BacktestConfig,
    CovarianceConfig,
    EmbeddingConfig,
    PortfolioConfig,
)
from pnlp.data.embeddings_loader import load_doc_embeddings
from pnlp.data.prices import PriceFetcher, load_price_data
from pnlp.data.universe_filter import LiquidityConfig, filter_investable_universe
from pnlp.embeddings.firm_aggregator import FirmAggregator, FirmEmbedding
from pnlp.portfolio.optimizer import PortfolioOptimizer, PortfolioWeights
from pnlp.primitives.covariance import SemanticShrinkageCovariance
from pnlp.validation.backtest import BacktestEngine, BacktestResult
from pnlp.validation.metrics import compute_portfolio_metrics, fama_french_regression
from pnlp.validation.transaction_costs import TransactionCostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def find_latest_returns_parquet() -> Path | None:
    results_dir = DATA_DIR / "results"
    parquets = sorted(results_dir.glob("returns_*.parquet"))
    return parquets[-1] if parquets else None


def load_spy_returns() -> pd.Series:
    """Load SPY daily returns from Tiingo cache."""
    fetcher = PriceFetcher()
    prices = fetcher.fetch(["SPY"], start=date(2000, 1, 1), end=date(2025, 12, 31))
    if "SPY" not in prices:
        raise RuntimeError("SPY prices not found in Tiingo cache. Run: uv run python -c "
                           "\"from pnlp.data.prices import PriceFetcher; "
                           "PriceFetcher().fetch(['SPY'], start=date(2010,1,1), end=date(2025,12,31))\"")
    returns = PriceFetcher.compute_returns(prices, frequency="daily")
    return returns["SPY"]


# ============================================================================
# PHASE 1: Fast analyses on saved return series
# ============================================================================

def run_factor_regression(returns_df: pd.DataFrame, ff_factors: pd.DataFrame | None) -> dict:
    """Run FF5+Mom regression on each strategy's return series."""
    results = {}
    for col in returns_df.columns:
        logger.info("Running FF5+Mom regression for %s...", col)
        try:
            reg = fama_french_regression(
                returns_df[col],
                factors=ff_factors,
                model="ff5_mom",
                max_lag=10,
            )
            results[col] = reg
        except Exception as e:
            logger.error("Factor regression failed for %s: %s", col, e)
            results[col] = {"error": str(e)}
    return results


def run_benchmark_comparison(returns_df: pd.DataFrame) -> dict:
    """Compare all strategies to SPY benchmark."""
    try:
        spy_returns = load_spy_returns()
    except Exception as e:
        logger.error("Could not load SPY: %s", e)
        return {"error": str(e)}

    spy_metrics = compute_portfolio_metrics(spy_returns)
    results = {"spy": spy_metrics}
    for col in returns_df.columns:
        metrics = compute_portfolio_metrics(returns_df[col], benchmark_returns=spy_returns)
        results[col] = metrics
    return results


def run_higher_moments(returns_df: pd.DataFrame) -> dict:
    """Compute skewness, kurtosis, VaR, CVaR for each strategy."""
    from scipy import stats as sp_stats

    results = {}
    for col in returns_df.columns:
        rets = returns_df[col].dropna()
        results[col] = {
            "skewness": float(sp_stats.skew(rets)),
            "kurtosis": float(sp_stats.kurtosis(rets)),  # excess kurtosis
            "jarque_bera_stat": float(sp_stats.jarque_bera(rets).statistic),
            "jarque_bera_pvalue": float(sp_stats.jarque_bera(rets).pvalue),
            "var_5pct": float(np.percentile(rets, 5)),
            "cvar_5pct": float(rets[rets <= np.percentile(rets, 5)].mean()),
            "var_1pct": float(np.percentile(rets, 1)),
            "cvar_1pct": float(rets[rets <= np.percentile(rets, 1)].mean()),
        }
    return results


# ============================================================================
# PHASE 2: Shuffle test (requires re-running walk-forward)
# ============================================================================

def run_shuffle_test(
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
    n_shuffles: int = 5,
    seed: int = 42,
) -> dict:
    """Shuffle ticker-embedding mapping and compare to UNSHUFFLED semantic.

    If the semantic signal is real, shuffled portfolios should perform WORSE
    than the unshuffled semantic portfolio (not just compared to LW).
    """
    from pnlp.validation.statistical_tests import block_permutation_test

    rng = np.random.RandomState(seed)
    shuffle_results = []
    tc_model = TransactionCostModel()

    # Step 1: Run UNSHUFFLED semantic (baseline for comparison)
    logger.info("=== Running unshuffled semantic baseline ===")
    unshuffled_weights: dict[date, PortfolioWeights] = {}
    tc_rates_history: dict[date, pd.Series] = {}

    for rebal_date in tqdm(rebalance_dates, desc="Unshuffled baseline"):
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )
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

        sub_embeddings = {t: firm_embeddings[t] for t in eligible}
        lookback_returns = lookback_rets[eligible]

        try:
            cov_estimator = SemanticShrinkageCovariance(
                config=cov_config, historical_returns=lookback_returns,
            )
            cov = cov_estimator.estimate(sub_embeddings, sigma_estimates=None)
            optimizer = PortfolioOptimizer(portfolio_config)
            w = optimizer.optimize(cov)
            w.metadata["method"] = "semantic_shrinkage"
            unshuffled_weights[rebal_date] = w
        except Exception as e:
            logger.error("Unshuffled semantic failed at %s: %s", rebal_date, e)

    # Backtest unshuffled
    engine = BacktestEngine(backtest_config)
    unshuffled_sharpe = None
    unshuffled_returns = None
    if unshuffled_weights:
        bt_unshuf = engine.run(
            unshuffled_weights, daily_returns,
            strategy_name="unshuffled_semantic",
            tc_rates_history=tc_rates_history,
        )
        unshuffled_sharpe = bt_unshuf.metrics.get("sharpe_ratio", 0)
        unshuffled_returns = bt_unshuf.portfolio_returns
        logger.info("Unshuffled semantic Sharpe: %.4f", unshuffled_sharpe)

    # Step 2: Run SHUFFLED versions
    for shuffle_i in range(n_shuffles):
        logger.info("=== Shuffle %d/%d ===", shuffle_i + 1, n_shuffles)

        shuffled_weights: dict[date, PortfolioWeights] = {}

        for rebal_date in tqdm(rebalance_dates, desc=f"Shuffle {shuffle_i+1}"):
            firm_embeddings = aggregator.aggregate_universe(
                all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
            )
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
            comp = lookback_rets.notna().sum() / len(lookback_rets)
            eligible = [t for t in eligible if comp.get(t, 0) >= 0.8]
            if len(eligible) < 20:
                continue

            sub_embeddings = {t: firm_embeddings[t] for t in eligible}

            # SHUFFLE: randomly permute the embedding assignment among eligible tickers
            shuffled_tickers = list(eligible)
            rng.shuffle(shuffled_tickers)
            shuffled_embeddings = {
                orig: sub_embeddings[shuf]
                for orig, shuf in zip(eligible, shuffled_tickers)
            }

            lookback_returns = lookback_rets[eligible]

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
                logger.error("Shuffled semantic failed at %s: %s", rebal_date, e)

        if shuffled_weights:
            bt_shuf = engine.run(
                shuffled_weights, daily_returns,
                strategy_name=f"shuffled_{shuffle_i}",
                tc_rates_history=tc_rates_history,
            )
            shuffle_results.append({
                "shuffle_id": shuffle_i,
                "sharpe": bt_shuf.metrics.get("sharpe_ratio", 0),
                "ann_return": bt_shuf.metrics.get("annualized_return", 0),
                "ann_vol": bt_shuf.metrics.get("annualized_volatility", 0),
                "max_drawdown": bt_shuf.metrics.get("max_drawdown", 0),
            })

    shuffled_sharpes = [r["sharpe"] for r in shuffle_results]

    # Statistical test: unshuffled vs mean of shuffled
    placebo_test = None
    if unshuffled_returns is not None and shuffle_results:
        # Compare unshuffled Sharpe to distribution of shuffled Sharpes
        n_better = sum(1 for s in shuffled_sharpes if s >= unshuffled_sharpe)
        placebo_test = {
            "unshuffled_sharpe": unshuffled_sharpe,
            "n_shuffles_better": n_better,
            "fraction_better": n_better / len(shuffled_sharpes),
        }

    return {
        "n_shuffles": n_shuffles,
        "unshuffled_sharpe": unshuffled_sharpe,
        "shuffled_sharpes": shuffled_sharpes,
        "shuffled_mean_sharpe": float(np.mean(shuffled_sharpes)) if shuffled_sharpes else None,
        "shuffled_std_sharpe": float(np.std(shuffled_sharpes)) if shuffled_sharpes else None,
        "sharpe_gap_unshuf_minus_shuf": float(unshuffled_sharpe - np.mean(shuffled_sharpes)) if (unshuffled_sharpe is not None and shuffled_sharpes) else None,
        "placebo_test": placebo_test,
        "per_shuffle": shuffle_results,
    }


# ============================================================================
# PHASE 2: TC sensitivity (reuse weights, only re-run backtest engine)
# ============================================================================

def run_tc_sensitivity(
    all_doc_embeddings: dict[str, list[DocumentEmbedding]],
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    rebalance_dates: list[date],
    aggregator: FirmAggregator,
    portfolio_config: PortfolioConfig,
    cov_config: CovarianceConfig,
    liq_config: LiquidityConfig,
    lookback_days: int,
    tc_multipliers: list[float] = [0.5, 1.0, 1.5, 2.0],
    start: date = date(2013, 4, 1),
    end: date = date(2024, 6, 30),
) -> dict:
    """Re-run backtest with different TC multiplier levels.

    Computes weights once and TC rates once, then scales TC rates by
    the multiplier for each sensitivity run. This tests how robust the
    semantic shrinkage advantage is to TC assumptions.
    """
    tc_model = TransactionCostModel()

    # Step 1: Compute weights and TC rates once
    logger.info("Computing weights for TC sensitivity (this is the slow part)...")
    semantic_weights: dict[date, PortfolioWeights] = {}
    lw_weights: dict[date, PortfolioWeights] = {}
    ew_weights: dict[date, PortfolioWeights] = {}
    base_tc_rates: dict[date, pd.Series] = {}

    for rebal_date in tqdm(rebalance_dates, desc="TC sensitivity weights"):
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )
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

        # Base TC rates
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates = tc_model.get_tc_rates(median_adv, daily_returns, str(rebal_date))
        base_tc_rates[rebal_date] = tc_rates

        sub_embeddings = {t: firm_embeddings[t] for t in eligible}

        # Strict lag
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_returns = returns_before_T[eligible].iloc[-lookback_days:]

        try:
            cov_estimator = SemanticShrinkageCovariance(
                config=cov_config, historical_returns=lookback_returns,
            )
            cov = cov_estimator.estimate(sub_embeddings, sigma_estimates=None)
            optimizer = PortfolioOptimizer(portfolio_config)
            w = optimizer.optimize(cov)
            w.metadata["method"] = "semantic_shrinkage"
            semantic_weights[rebal_date] = w
        except Exception as e:
            logger.error("Semantic failed at %s: %s", rebal_date, e)

        try:
            lw_w = ledoit_wolf_portfolio(
                lookback_returns,
                objective=portfolio_config.objective,
                long_only=portfolio_config.long_only,
                max_weight=portfolio_config.max_weight,
            )
            lw_weights[rebal_date] = lw_w
        except Exception as e:
            logger.error("LW failed at %s: %s", rebal_date, e)

        ew_weights[rebal_date] = equal_weight_portfolio(eligible)

    # Step 2: Run backtest at each TC multiplier level (fast — just simulation)
    results = {}
    for mult in tc_multipliers:
        logger.info("Running backtest with TC multiplier = %.1fx", mult)
        scaled_tc_rates = {
            d: rates * mult for d, rates in base_tc_rates.items()
        }
        config = BacktestConfig(start_date=start, end_date=end)
        engine = BacktestEngine(config)

        tc_result = {}
        for name, weights_hist in [
            ("semantic_shrinkage", semantic_weights),
            ("ledoit_wolf", lw_weights),
            ("equal_weight", ew_weights),
        ]:
            if not weights_hist:
                continue
            bt = engine.run(
                weights_hist, daily_returns, strategy_name=name,
                tc_rates_history=scaled_tc_rates,
            )
            tc_result[name] = {
                "sharpe": bt.metrics.get("sharpe_ratio", 0),
                "ann_return": bt.metrics.get("annualized_return", 0),
                "ann_vol": bt.metrics.get("annualized_volatility", 0),
                "max_drawdown": bt.metrics.get("max_drawdown", 0),
                "avg_turnover": bt.metrics.get("avg_turnover", 0),
            }

        results[f"{mult:.1f}x"] = tc_result

    return results


# ============================================================================
# CLI and main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness checks for semantic shrinkage portfolio")
    parser.add_argument("--returns-file", type=str, default=None,
                        help="Path to saved returns parquet. If None, uses most recent.")
    parser.add_argument("--run-shuffle", action="store_true",
                        help="Run shuffle test (Phase 2, ~15 min)")
    parser.add_argument("--run-tc-sensitivity", action="store_true",
                        help="Run TC sensitivity (Phase 2, ~15 min)")
    parser.add_argument("--all", action="store_true",
                        help="Run all checks including Phase 2")
    parser.add_argument("--objective", type=str, default="min_variance",
                        choices=["min_variance", "max_sharpe", "mean_variance", "risk_parity"])
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--max-firms", type=int, default=500,
                        help="Universe size for Phase 2 tests (default: 500)")
    parser.add_argument("--n-shuffles", type=int, default=5,
                        help="Number of shuffle iterations (default: 5)")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2007, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
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


def main() -> None:
    args = parse_args()
    if args.all:
        args.run_shuffle = True
        args.run_tc_sensitivity = True

    all_results: dict = {}

    # -----------------------------------------------------------------------
    # Load saved returns for Phase 1
    # -----------------------------------------------------------------------
    returns_path = args.returns_file
    if returns_path is None:
        found = find_latest_returns_parquet()
        if found:
            returns_path = str(found)
        else:
            logger.error("No saved returns parquet found. Run run_backtest.py first.")
            sys.exit(1)

    logger.info("Loading saved returns from %s", returns_path)
    returns_df = pd.read_parquet(returns_path)
    logger.info("Returns: %d days, strategies: %s", len(returns_df), list(returns_df.columns))

    # -----------------------------------------------------------------------
    # Phase 1a: Fama-French factor regression
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FAMA-FRENCH 5-FACTOR + MOMENTUM REGRESSION")
    print("=" * 80)

    ff_factors = None
    try:
        from pnlp.data.fama_french import load_ff_factors
        start_str = str(returns_df.index.min().date())
        end_str = str(returns_df.index.max().date())
        ff_factors = load_ff_factors(start=start_str, end=end_str)
    except Exception as e:
        logger.warning("Could not load FF factors: %s", e)
        logger.warning("Skipping factor regression. Download FF data manually or ensure internet access.")

    if ff_factors is not None:
        factor_results = run_factor_regression(returns_df, ff_factors)
        all_results["factor_regression"] = factor_results

        for strategy, reg in factor_results.items():
            if "error" in reg:
                print(f"\n  {strategy}: ERROR — {reg['error']}")
                continue
            print(f"\n  {strategy}:")
            print(f"    Alpha (annualized): {reg['alpha_annual']*100:+.2f}% (t={reg['alpha_tstat']:.2f}, p={reg['alpha_pvalue']:.4f})")
            print(f"    R²: {reg['r_squared']:.3f}  (adj: {reg['r_squared_adj']:.3f})")
            print(f"    Factor loadings:")
            for factor, beta in reg["betas"].items():
                t = reg["tstats"][factor]
                sig = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
                print(f"      {factor:>8s}: {beta:+.4f} (t={t:+.2f}) {sig}")
    else:
        print("  SKIPPED — FF factors not available (no internet?)")
        print("  To fix: ensure internet access and re-run, or place FF CSV files in data/ff_factors/")

    # -----------------------------------------------------------------------
    # Phase 1b: SPY benchmark
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SPY BENCHMARK COMPARISON")
    print("=" * 80)

    try:
        benchmark_results = run_benchmark_comparison(returns_df)
        all_results["benchmark"] = benchmark_results

        spy_m = benchmark_results.get("spy", {})
        print(f"\n  SPY:  Sharpe={spy_m.get('sharpe_ratio', 0):.4f}  "
              f"AnnRet={spy_m.get('annualized_return', 0):.4f}  "
              f"AnnVol={spy_m.get('annualized_volatility', 0):.4f}  "
              f"MaxDD={spy_m.get('max_drawdown', 0):.4f}")

        for strategy in returns_df.columns:
            m = benchmark_results.get(strategy, {})
            print(f"  {strategy}:  Sharpe={m.get('sharpe_ratio', 0):.4f}  "
                  f"AnnRet={m.get('annualized_return', 0):.4f}  "
                  f"InfoRatio={m.get('information_ratio', 'N/A')}  "
                  f"CAPM_alpha={m.get('capm_alpha', 'N/A')}  "
                  f"Beta={m.get('capm_beta', 'N/A')}")
    except Exception as e:
        logger.warning("SPY benchmark failed: %s", e)
        print(f"  SKIPPED — {e}")

    # -----------------------------------------------------------------------
    # Phase 1c: Higher moments
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HIGHER MOMENTS & TAIL RISK")
    print("=" * 80)

    moments_results = run_higher_moments(returns_df)
    all_results["higher_moments"] = moments_results

    print(f"\n  {'Strategy':<22s}  {'Skew':>8s}  {'Kurt':>8s}  {'VaR 5%':>10s}  {'CVaR 5%':>10s}  {'VaR 1%':>10s}  {'CVaR 1%':>10s}")
    print("  " + "-" * 82)
    for strategy, m in moments_results.items():
        print(f"  {strategy:<22s}  {m['skewness']:>8.3f}  {m['kurtosis']:>8.3f}  "
              f"{m['var_5pct']*100:>9.3f}%  {m['cvar_5pct']*100:>9.3f}%  "
              f"{m['var_1pct']*100:>9.3f}%  {m['cvar_1pct']*100:>9.3f}%")

    # -----------------------------------------------------------------------
    # Phase 2: Load full data if needed
    # -----------------------------------------------------------------------
    if args.run_shuffle or args.run_tc_sensitivity:
        logger.info("Loading full dataset for Phase 2...")
        all_doc_embeddings = load_doc_embeddings(args.text_source)
        _, daily_returns, dollar_volume = load_price_data(list(all_doc_embeddings.keys()))
        rebalance_dates = generate_quarterly_dates(args.start, args.end)
        embedding_config = EmbeddingConfig()
        aggregator = FirmAggregator(embedding_config)
        portfolio_config = PortfolioConfig(
            objective=getattr(args, "objective", "min_variance"),
            long_only=True,
            max_weight=getattr(args, "max_weight", 0.05),
        )
        cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
        liq_config = LiquidityConfig(
            adv_threshold=args.adv_threshold,
            adv_lookback_days=args.adv_lookback,
            covariance_lookback_days=args.lookback_days,
            max_firms=args.max_firms,
        )

    # -----------------------------------------------------------------------
    # Phase 2: Shuffle test
    # -----------------------------------------------------------------------
    if args.run_shuffle:
        print("\n" + "=" * 80)
        print("SHUFFLE TEST (Placebo)")
        print("=" * 80)

        backtest_config = BacktestConfig(start_date=args.start, end_date=args.end)
        shuffle_results = run_shuffle_test(
            all_doc_embeddings, daily_returns, dollar_volume,
            rebalance_dates,
            aggregator, portfolio_config, cov_config, backtest_config,
            liq_config,
            lookback_days=args.lookback_days,
            n_shuffles=args.n_shuffles,
        )
        all_results["shuffle_test"] = shuffle_results

        print(f"\n  Unshuffled semantic Sharpe: {shuffle_results['unshuffled_sharpe']:.4f}")
        print(f"  Shuffled Sharpes: {[f'{s:.4f}' for s in shuffle_results['shuffled_sharpes']]}")
        print(f"  Shuffled mean:  {shuffle_results['shuffled_mean_sharpe']:.4f}")
        print(f"  Shuffled std:   {shuffle_results['shuffled_std_sharpe']:.4f}")
        gap = shuffle_results.get("sharpe_gap_unshuf_minus_shuf")
        if gap is not None:
            print(f"  Gap (unshuf - shuf): {gap:+.4f}")
        pt = shuffle_results.get("placebo_test")
        if pt:
            print(f"  Shuffles >= unshuffled: {pt['n_shuffles_better']}/{shuffle_results['n_shuffles']}")
        print(f"\n  If semantic signal is real: shuffled Sharpe should be WORSE than unshuffled.")

    # -----------------------------------------------------------------------
    # Phase 2: TC sensitivity
    # -----------------------------------------------------------------------
    if args.run_tc_sensitivity:
        print("\n" + "=" * 80)
        print("TRANSACTION COST SENSITIVITY")
        print("=" * 80)

        tc_results = run_tc_sensitivity(
            all_doc_embeddings, daily_returns, dollar_volume,
            rebalance_dates,
            aggregator, portfolio_config, cov_config,
            liq_config,
            lookback_days=args.lookback_days,
            tc_multipliers=[0.5, 1.0, 1.5, 2.0],
            start=args.start, end=args.end,
        )
        all_results["tc_sensitivity"] = tc_results

        print(f"\n  {'TC mult':<10s}  {'Sem Sharpe':>11s}  {'LW Sharpe':>10s}  {'Gap':>8s}  {'EW Sharpe':>10s}")
        print("  " + "-" * 55)
        for mult_label, tc_data in sorted(tc_results.items()):
            sem = tc_data.get("semantic_shrinkage", {})
            lw = tc_data.get("ledoit_wolf", {})
            ew = tc_data.get("equal_weight", {})
            sem_s = sem.get("sharpe", 0)
            lw_s = lw.get("sharpe", 0)
            gap = sem_s - lw_s
            print(f"  {mult_label:>10s}  {sem_s:>11.4f}  {lw_s:>10.4f}  {gap:>+8.4f}  {ew.get('sharpe', 0):>10.4f}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = args.output or str(output_dir / f"robustness_{timestamp}.json")

    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Robustness results saved to %s", path)

    print("\n" + "=" * 80)
    print(f"Results saved to {path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
