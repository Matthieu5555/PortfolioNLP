"""
Temporal sensitivity test: Does the embedding model's training window matter?

Tests whether the semantic shrinkage advantage varies across three sub-periods:
1. Pre-training (2007-2016): Model trained on ~2023 data, backtest is "before" training.
   If look-ahead bias mattered, performance should be artificially inflated here.
2. Training-contemporaneous (2017-2022): Model and data are roughly aligned.
3. Post-training (2023-2025): Model has no future knowledge at all.

If the semantic-vs-LW Sharpe gap is stable across all three periods, the model's
training date is irrelevant — the signal comes from textual similarity structure,
not from the model's "future knowledge".

Usage:
    uv run python scripts/run_temporal_sensitivity.py
    uv run python scripts/run_temporal_sensitivity.py --max-firms 200
    uv run python scripts/run_temporal_sensitivity.py --text-source transcript
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
from pnlp.portfolio.optimizer import PortfolioOptimizer
from pnlp.primitives.covariance import SemanticShrinkageCovariance
from pnlp.validation.backtest import BacktestEngine
from pnlp.validation.statistical_tests import block_permutation_test, lo_sharpe_correction
from pnlp.validation.transaction_costs import TransactionCostModel
from pnlp.data.dates import generate_quarterly_dates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# The three sub-periods
PERIODS = [
    ("pre_training", date(2007, 4, 1), date(2016, 12, 31)),
    ("training_contemporaneous", date(2017, 1, 1), date(2022, 12, 31)),
    ("post_training", date(2023, 1, 1), date(2025, 12, 31)),
    ("full_window", date(2007, 4, 1), date(2025, 12, 31)),
]



def run_one_period(
    period_name: str,
    start: date,
    end: date,
    all_doc_embeddings: dict,
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    aggregator: FirmAggregator,
    portfolio_config: PortfolioConfig,
    cov_config: CovarianceConfig,
    liq_config: LiquidityConfig,
    tc_model: TransactionCostModel,
    lookback_days: int,
) -> dict:
    """Run a single sub-period backtest and return summary metrics."""
    rebalance_dates = generate_quarterly_dates(start, end)
    logger.info(
        "\n%s: %d quarterly dates from %s to %s",
        period_name, len(rebalance_dates), start, end,
    )

    semantic_weights = {}
    lw_weights = {}
    ew_weights = {}
    tc_rates_history = {}

    for rebal_date in tqdm(rebalance_dates, desc=period_name):
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_config,
        )
        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-lookback_days:]
        completeness = lookback_rets.notna().sum() / max(len(lookback_rets), 1)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]

        if len(eligible) < 20:
            continue

        # TC rates
        dv_before = dollar_volume.loc[dollar_volume.index < str(rebal_date)]
        adv_window = dv_before.iloc[-liq_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates = tc_model.get_tc_rates(median_adv, daily_returns, str(rebal_date))
        tc_rates_history[rebal_date] = tc_rates

        # Semantic shrinkage
        try:
            sub_embs = {t: firm_embeddings[t] for t in eligible}
            lookback_returns = returns_before_T[eligible].iloc[-lookback_days:]
            cov_est = SemanticShrinkageCovariance(
                config=cov_config, historical_returns=lookback_returns,
            )
            cov = cov_est.estimate(sub_embs, sigma_estimates=None)
            optimizer = PortfolioOptimizer(portfolio_config)
            sem_w = optimizer.optimize(cov)
            semantic_weights[rebal_date] = sem_w
        except Exception as e:
            logger.error("Semantic failed at %s: %s", rebal_date, e)

        # Ledoit-Wolf
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

        # Equal weight
        ew_w = equal_weight_portfolio(eligible)
        ew_weights[rebal_date] = ew_w

    # Run backtests
    backtest_config = BacktestConfig(start_date=start, end_date=end)
    engine = BacktestEngine(backtest_config)
    results = {}

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
        results[name] = bt

    if "semantic_shrinkage" not in results or "ledoit_wolf" not in results:
        logger.warning("Missing strategies for %s, skipping stats", period_name)
        return {"period": period_name, "start": str(start), "end": str(end), "error": "insufficient_data"}

    # Compute metrics and statistical test
    sem_bt = results["semantic_shrinkage"]
    lw_bt = results["ledoit_wolf"]

    common_idx = sem_bt.portfolio_returns.index.intersection(lw_bt.portfolio_returns.index)
    sem_rets = sem_bt.portfolio_returns.loc[common_idx].values
    lw_rets = lw_bt.portfolio_returns.loc[common_idx].values

    perm_test = block_permutation_test(sem_rets, lw_rets, block_size=63, n_permutations=10000)

    # Lo correction
    sem_eta = lo_sharpe_correction(sem_rets, max_lag=63)
    lw_eta = lo_sharpe_correction(lw_rets, max_lag=63)

    period_result = {
        "period": period_name,
        "start": str(start),
        "end": str(end),
        "n_rebalances": len(rebalance_dates),
        "n_trading_days": len(common_idx),
    }

    for name, bt in results.items():
        period_result[f"{name}_sharpe"] = round(bt.metrics.get("sharpe_ratio", 0), 4)
        period_result[f"{name}_ann_return"] = round(bt.metrics.get("annualized_return", 0), 4)
        period_result[f"{name}_ann_vol"] = round(bt.metrics.get("annualized_volatility", 0), 4)
        period_result[f"{name}_max_dd"] = round(bt.metrics.get("max_drawdown", 0), 4)

    period_result["sem_vs_lw_sharpe_gap"] = round(
        period_result["semantic_shrinkage_sharpe"] - period_result["ledoit_wolf_sharpe"], 4
    )
    period_result["sem_vs_lw_p_value"] = round(float(perm_test.p_value), 4)
    period_result["sem_lo_corrected"] = round(
        period_result["semantic_shrinkage_sharpe"] / sem_eta if sem_eta > 0 else 0, 4
    )
    period_result["lw_lo_corrected"] = round(
        period_result["ledoit_wolf_sharpe"] / lw_eta if lw_eta > 0 else 0, 4
    )

    return period_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal sensitivity test for embedding look-ahead bias")
    parser.add_argument("--max-firms", type=int, default=500)
    parser.add_argument("--lookback-days", type=int, default=504)
    parser.add_argument("--text-source", type=str, default="10k", choices=["10k", "transcript", "news"])
    parser.add_argument("--max-weight", type=float, default=0.05)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("TEMPORAL SENSITIVITY TEST: Embedding Look-Ahead Bias")
    logger.info("=" * 70)
    logger.info("Hypothesis: If the model's training window creates look-ahead bias,")
    logger.info("the pre-training period (2007-2016) should show an inflated advantage.")
    logger.info("")

    # Load data once
    all_doc_embeddings = load_doc_embeddings(args.text_source)
    _, daily_returns, dollar_volume = load_price_data(list(all_doc_embeddings.keys()))

    # Config
    aggregator = FirmAggregator(EmbeddingConfig())
    portfolio_config = PortfolioConfig(objective="min_variance", long_only=True, max_weight=args.max_weight)
    cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
    liq_config = LiquidityConfig(
        adv_threshold=1_000_000.0, adv_lookback_days=21,
        covariance_lookback_days=args.lookback_days, max_firms=args.max_firms,
    )
    tc_model = TransactionCostModel()

    # Run each sub-period
    all_results = []
    for period_name, start, end in PERIODS:
        result = run_one_period(
            period_name, start, end,
            all_doc_embeddings, daily_returns, dollar_volume,
            aggregator, portfolio_config, cov_config, liq_config, tc_model,
            args.lookback_days,
        )
        all_results.append(result)
        logger.info(
            "%s: Sem=%.3f, LW=%.3f, Gap=%+.4f, p=%.3f",
            result["period"],
            result.get("semantic_shrinkage_sharpe", 0),
            result.get("ledoit_wolf_sharpe", 0),
            result.get("sem_vs_lw_sharpe_gap", 0),
            result.get("sem_vs_lw_p_value", 1),
        )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEMPORAL SENSITIVITY SUMMARY")
    logger.info("=" * 70)
    logger.info("%-25s %8s %8s %8s %8s", "Period", "Sem", "LW", "Gap", "p-val")
    logger.info("-" * 70)
    for r in all_results:
        if "error" in r:
            logger.info("%-25s %s", r["period"], r["error"])
        else:
            logger.info(
                "%-25s %8.3f %8.3f %+8.4f %8.3f",
                r["period"],
                r.get("semantic_shrinkage_sharpe", 0),
                r.get("ledoit_wolf_sharpe", 0),
                r.get("sem_vs_lw_sharpe_gap", 0),
                r.get("sem_vs_lw_p_value", 1),
            )

    # Check for look-ahead pattern
    pre = next((r for r in all_results if r["period"] == "pre_training"), None)
    post = next((r for r in all_results if r["period"] == "post_training"), None)
    if pre and post and "error" not in pre and "error" not in post:
        pre_gap = pre.get("sem_vs_lw_sharpe_gap", 0)
        post_gap = post.get("sem_vs_lw_sharpe_gap", 0)
        if pre_gap > post_gap + 0.05:
            logger.warning("⚠ Pre-training gap (%.4f) > post-training gap (%.4f) by >0.05 — potential look-ahead", pre_gap, post_gap)
        else:
            logger.info("\nNo look-ahead pattern: pre-training gap (%.4f) not inflated vs post-training (%.4f)", pre_gap, post_gap)

    # Save
    results_dir = DATA_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output = {
        "experiment": "temporal_sensitivity",
        "text_source": args.text_source,
        "max_firms": args.max_firms,
        "lookback_days": args.lookback_days,
        "model": "BAAI/bge-base-en-v1.5",
        "model_training_cutoff": "~2023",
        "hypothesis": (
            "If the embedding model's training on ~2023 data creates look-ahead bias, "
            "the pre-training period (2007-2016) should show an inflated semantic advantage "
            "because the model 'knows' how businesses evolved. If the gap is stable across "
            "all periods, the model's training window is irrelevant."
        ),
        "periods": all_results,
    }

    path = results_dir / f"temporal_sensitivity_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("\nResults saved to %s", path)


if __name__ == "__main__":
    main()
