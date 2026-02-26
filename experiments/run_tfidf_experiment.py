"""
TF-IDF baseline experiment: zero look-ahead bias comparison.

Compares three shrinkage targets:
1. Neural embeddings (BAAI/bge-base-en-v1.5) — the main approach
2. TF-IDF embeddings — zero look-ahead bias by construction
3. Ledoit-Wolf (identity target) — standard baseline

If TF-IDF produces similar results to neural embeddings, it proves the
portfolio signal comes from textual similarity structure, not from the
embedding model's training data / future knowledge.

Usage:
    uv run python experiments/run_tfidf_experiment.py
    uv run python experiments/run_tfidf_experiment.py --max-firms 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
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
from pnlp.embeddings.tfidf_encoder import TfidfFirmEncoder
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



def load_filing_texts(filings_dir: Path) -> dict[str, list[tuple[date, str]]]:
    """Load raw filing texts from cached text files.

    Returns:
        dict mapping ticker -> list of (doc_date, text) tuples.
    """
    filing_texts: dict[str, list[tuple[date, str]]] = {}
    ticker_dirs = sorted(filings_dir.iterdir())

    for ticker_dir in ticker_dirs:
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        docs = []
        for txt_file in sorted(ticker_dir.glob("*.txt")):
            year_str = txt_file.stem
            if not year_str.isdigit():
                continue
            doc_date = date(int(year_str), 1, 1)
            text = txt_file.read_text(errors="replace")
            if len(text) >= 100:
                docs.append((doc_date, text))
        if docs:
            filing_texts[ticker] = docs

    total_docs = sum(len(v) for v in filing_texts.values())
    logger.info("Loaded raw texts: %d tickers, %d documents", len(filing_texts), total_docs)
    return filing_texts


def compute_tfidf_weights(
    tfidf_embeddings: dict[str, np.ndarray],
    daily_returns: pd.DataFrame,
    eligible: list[str],
    rebal_date: date,
    portfolio_config: PortfolioConfig,
    cov_config: CovarianceConfig,
    lookback_days: int,
) -> pd.Series | None:
    """Compute portfolio weights using TF-IDF cosine as shrinkage target."""
    # Build FirmEmbedding objects from TF-IDF vectors
    tfidf_firm_embeddings = {}
    for t in eligible:
        if t in tfidf_embeddings:
            tfidf_firm_embeddings[t] = FirmEmbedding(
                ticker=t,
                as_of_date=rebal_date,
                embedding=tfidf_embeddings[t],
                n_documents=1,
            )

    if len(tfidf_firm_embeddings) < 20:
        return None

    # Use only tickers that have TF-IDF embeddings
    eligible_tfidf = [t for t in eligible if t in tfidf_firm_embeddings]

    returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
    lookback_returns = returns_before_T[eligible_tfidf].iloc[-lookback_days:]

    cov_estimator = SemanticShrinkageCovariance(
        config=cov_config,
        historical_returns=lookback_returns,
    )
    cov = cov_estimator.estimate(tfidf_firm_embeddings, sigma_estimates=None)

    optimizer = PortfolioOptimizer(portfolio_config)
    weights = optimizer.optimize(cov)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF baseline experiment")
    parser.add_argument("--max-firms", type=int, default=500)
    parser.add_argument("--lookback-days", type=int, default=504)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2007, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--tfidf-features", type=int, default=768)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("TF-IDF BASELINE EXPERIMENT: Zero Look-Ahead Bias Comparison")
    logger.info("=" * 70)

    # Load neural embeddings (for reference)
    all_doc_embeddings = load_doc_embeddings("10k")
    _, daily_returns, dollar_volume = load_price_data(list(all_doc_embeddings.keys()))

    # Load raw filing texts for TF-IDF
    filings_dir = DATA_DIR / "filings"
    filing_texts = load_filing_texts(filings_dir)

    # Setup
    aggregator = FirmAggregator(EmbeddingConfig())
    portfolio_config = PortfolioConfig(objective="min_variance", long_only=True, max_weight=args.max_weight)
    cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
    liq_config = LiquidityConfig(
        adv_threshold=1_000_000.0, adv_lookback_days=21,
        covariance_lookback_days=args.lookback_days, max_firms=args.max_firms,
    )
    tc_model = TransactionCostModel()
    tfidf_encoder = TfidfFirmEncoder(max_features=args.tfidf_features)

    rebalance_dates = generate_quarterly_dates(args.start, args.end)
    logger.info("Running %d quarterly rebalances from %s to %s", len(rebalance_dates), args.start, args.end)

    # Storage
    neural_weights = {}
    tfidf_weights = {}
    lw_weights = {}
    ew_weights = {}
    tc_rates_history = {}
    rebalance_log = []

    for rebal_date in tqdm(rebalance_dates, desc="Rebalancing"):
        # Neural firm embeddings (standard)
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )

        # Universe filter (shared across all strategies)
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_config,
        )
        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-args.lookback_days:]
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

        # ---- Neural semantic shrinkage ----
        try:
            sub_embs = {t: firm_embeddings[t] for t in eligible}
            lookback_returns = returns_before_T[eligible].iloc[-args.lookback_days:]
            cov_est = SemanticShrinkageCovariance(config=cov_config, historical_returns=lookback_returns)
            cov = cov_est.estimate(sub_embs, sigma_estimates=None)
            optimizer = PortfolioOptimizer(portfolio_config)
            neural_weights[rebal_date] = optimizer.optimize(cov)
        except Exception as e:
            logger.error("Neural failed at %s: %s", rebal_date, e)

        # ---- TF-IDF shrinkage ----
        try:
            tfidf_embs = tfidf_encoder.encode_universe(
                filing_texts, as_of_date=rebal_date, eligible_tickers=eligible,
            )
            tfidf_w = compute_tfidf_weights(
                tfidf_embs, daily_returns, eligible, rebal_date,
                portfolio_config, cov_config, args.lookback_days,
            )
            if tfidf_w is not None:
                tfidf_weights[rebal_date] = tfidf_w
        except Exception as e:
            logger.error("TF-IDF failed at %s: %s", rebal_date, e)

        # ---- Ledoit-Wolf ----
        try:
            lw_w = ledoit_wolf_portfolio(
                lookback_returns, objective=portfolio_config.objective,
                long_only=portfolio_config.long_only, max_weight=portfolio_config.max_weight,
            )
            lw_weights[rebal_date] = lw_w
        except Exception as e:
            logger.error("LW failed at %s: %s", rebal_date, e)

        # ---- Equal weight ----
        ew_weights[rebal_date] = equal_weight_portfolio(eligible)

        rebalance_log.append({
            "date": str(rebal_date),
            "n_eligible": len(eligible),
            "n_tfidf": len(tfidf_embs) if 'tfidf_embs' in dir() else 0,
        })

    # Run backtests
    backtest_config = BacktestConfig(start_date=args.start, end_date=args.end)
    engine = BacktestEngine(backtest_config)
    results = {}

    for name, weights_hist in [
        ("neural_semantic", neural_weights),
        ("tfidf_semantic", tfidf_weights),
        ("ledoit_wolf", lw_weights),
        ("equal_weight", ew_weights),
    ]:
        if not weights_hist:
            logger.warning("No weights for %s", name)
            continue
        bt = engine.run(
            weights_hist, daily_returns, strategy_name=name,
            tc_rates_history=tc_rates_history,
        )
        results[name] = bt
        logger.info(
            "%s: Sharpe=%.3f, AnnRet=%.3f, MaxDD=%.3f",
            name, bt.metrics.get("sharpe_ratio", 0),
            bt.metrics.get("annualized_return", 0),
            bt.metrics.get("max_drawdown", 0),
        )

    # Statistical tests
    stat_tests = {}
    all_strategies = list(results.keys())

    # Align return series
    if len(results) >= 2:
        common_idx = None
        for name, bt in results.items():
            if common_idx is None:
                common_idx = bt.portfolio_returns.index
            else:
                common_idx = common_idx.intersection(bt.portfolio_returns.index)

        aligned = {name: bt.portfolio_returns.loc[common_idx].values for name, bt in results.items()}

        for i, a in enumerate(all_strategies):
            for j, b in enumerate(all_strategies):
                if i >= j:
                    continue
                test = block_permutation_test(aligned[a], aligned[b], block_size=63, n_permutations=10000)
                stat_tests[f"{a}_vs_{b}"] = {
                    "p_value": round(float(test.p_value), 4),
                    "diff_means": round(float(test.statistic), 6),
                }

        for name in all_strategies:
            eta = lo_sharpe_correction(aligned[name], max_lag=63)
            raw_sharpe = results[name].metrics.get("sharpe_ratio", 0)
            stat_tests[f"{name}_lo"] = {
                "raw": round(raw_sharpe, 4),
                "eta": round(float(eta), 4),
                "corrected": round(raw_sharpe / eta if eta > 0 else 0, 4),
            }

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TF-IDF EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info("%-20s %8s %8s %8s %8s", "Strategy", "Sharpe", "AnnRet", "Vol", "MaxDD")
    logger.info("-" * 70)
    for name, bt in results.items():
        m = bt.metrics
        logger.info(
            "%-20s %8.3f %8.3f %8.3f %8.3f",
            name, m.get("sharpe_ratio", 0), m.get("annualized_return", 0),
            m.get("annualized_volatility", 0), m.get("max_drawdown", 0),
        )

    if "neural_semantic" in results and "tfidf_semantic" in results:
        neural_sharpe = results["neural_semantic"].metrics.get("sharpe_ratio", 0)
        tfidf_sharpe = results["tfidf_semantic"].metrics.get("sharpe_ratio", 0)
        logger.info("\nNeural vs TF-IDF Sharpe gap: %+.4f", neural_sharpe - tfidf_sharpe)

        key = "neural_semantic_vs_tfidf_semantic"
        if key in stat_tests:
            logger.info("Neural vs TF-IDF p-value: %.4f", stat_tests[key]["p_value"])

        if abs(neural_sharpe - tfidf_sharpe) < 0.05:
            logger.info("VERDICT: Neural and TF-IDF are similar — signal is from text structure, not model training data")
        elif tfidf_sharpe > neural_sharpe:
            logger.info("VERDICT: TF-IDF outperforms neural — no look-ahead bias advantage from pretrained model")
        else:
            logger.info("VERDICT: Neural outperforms TF-IDF — pretrained model adds value, but TF-IDF still shows text signal")

    # Save
    results_dir = DATA_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output = {
        "experiment": "tfidf_baseline",
        "config": {
            "max_firms": args.max_firms,
            "lookback_days": args.lookback_days,
            "tfidf_features": args.tfidf_features,
            "start": str(args.start),
            "end": str(args.end),
        },
        "hypothesis": (
            "If TF-IDF (zero look-ahead bias by construction) produces similar "
            "backtest results to BAAI/bge-base-en-v1.5, it proves the portfolio signal "
            "comes from textual similarity structure, not from the model's future knowledge."
        ),
        "metrics": {name: bt.metrics for name, bt in results.items()},
        "statistical_tests": stat_tests,
        "rebalance_log": rebalance_log,
    }

    path = results_dir / f"tfidf_experiment_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nResults saved to %s", path)


if __name__ == "__main__":
    main()
