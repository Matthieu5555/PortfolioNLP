"""
Text vs SIC Codes: Does Text Capture What Sector Codes Miss?

Tests whether embedding-based similarity predicts realized return
correlations better than 2-digit SIC sector membership.

Three phases:
  A. Cross-sectional fit:  regression of realized_corr ~ cosine_sim + SIC_match
  B. Backtest comparison:  text-cov vs SIC-block vs LW vs EW
  C. Case studies:         top disagreement pairs (same SIC/low cosine, different SIC/high cosine)

Usage:
    uv run python experiments/run_sector_analysis.py
    uv run python experiments/run_sector_analysis.py --p 500
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
from pnlp.validation.statistical_tests import block_bootstrap_ci, block_permutation_test
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


def _nw_tstat(vals: np.ndarray, max_lag: int = 4) -> float:
    """Fama-MacBeth t-stat with Newey-West (Bartlett kernel) standard errors."""
    n = len(vals)
    if n < 2 or np.std(vals) == 0:
        return 0.0
    mean_val = np.mean(vals)
    demeaned = vals - mean_val
    gamma0 = np.dot(demeaned, demeaned) / n
    nw_var = gamma0
    for lag in range(1, min(max_lag + 1, n)):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_lag = np.dot(demeaned[lag:], demeaned[:-lag]) / n
        nw_var += 2 * weight * gamma_lag
    se = np.sqrt(max(nw_var, 0) / n)
    return float(mean_val / se) if se > 1e-15 else 0.0


# ---------------------------------------------------------------------------
# Phase A: Cross-sectional fit
# ---------------------------------------------------------------------------

def run_cross_sectional_analysis(
    doc_embeddings: dict,
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    sic_codes: dict[str, str],
    p: int = 200,
    n_dates: int = 10,
    forward_days: int = 63,
    lookback_days: int = 504,
    start: str = "2013-04-01",
    end: str = "2024-04-01",
) -> dict:
    """Phase A: Does cosine_sim predict realized correlation better than SIC?"""

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    all_dates = generate_quarterly_dates(start_date, end_date)

    # Subsample dates for speed (pairwise analysis is O(p^2))
    step = max(1, len(all_dates) // n_dates)
    sample_dates = all_dates[::step][:n_dates]
    logger.info("Phase A: %d sample dates from %d total", len(sample_dates), len(all_dates))

    aggregator = FirmAggregator(EmbeddingConfig())
    liq_config = LiquidityConfig(max_firms=p)

    all_regressions = []

    for rebal_date in tqdm(sample_dates, desc="Phase A: cross-sectional"):
        rebal_str = str(rebal_date)

        firm_embeddings = aggregator.aggregate_universe(
            doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_config,
        )
        eligible = [t for t in eligible if t in daily_returns.columns]

        returns_before = daily_returns.loc[daily_returns.index < rebal_str]
        lookback_rets = returns_before[eligible].iloc[-lookback_days:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]

        if len(eligible) < 50:
            continue

        # Forward realized correlation
        simple_returns = np.exp(daily_returns) - 1.0
        fwd = simple_returns.loc[simple_returns.index >= rebal_str]
        fwd = fwd[eligible].iloc[:forward_days].dropna(axis=1, how="all")
        eligible = [t for t in eligible if t in fwd.columns]
        if len(eligible) < 50:
            continue

        realized_corr = fwd[eligible].corr()

        # Cosine similarity
        E = np.array([firm_embeddings[t].embedding for t in eligible])
        cosine_sim = E @ E.T
        np.clip(cosine_sim, -1.0, 1.0, out=cosine_sim)

        # SIC match
        n = len(eligible)
        sic_match = np.zeros((n, n))
        for i in range(n):
            si = sic_codes.get(eligible[i], f"_unk_{eligible[i]}")
            for j in range(i + 1, n):
                sj = sic_codes.get(eligible[j], f"_unk_{eligible[j]}")
                if si == sj and not si.startswith("_unk"):
                    sic_match[i, j] = 1.0
                    sic_match[j, i] = 1.0

        # Extract upper triangle (excluding diagonal)
        idx_upper = np.triu_indices(n, k=1)
        y = realized_corr.values[idx_upper]
        x_cos = cosine_sim[idx_upper]
        x_sic = sic_match[idx_upper]

        # Remove NaN pairs
        valid = ~np.isnan(y)
        y = y[valid]
        x_cos = x_cos[valid]
        x_sic = x_sic[valid]

        if len(y) < 100:
            continue

        # OLS: y ~ const + cosine + sic
        from scipy import stats as sp_stats

        X_both = np.column_stack([np.ones(len(y)), x_cos, x_sic])
        c_both, _, _, _ = np.linalg.lstsq(X_both, y, rcond=None)
        ss_res_both = np.sum((y - X_both @ c_both) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_both = 1.0 - ss_res_both / ss_tot if ss_tot > 0 else 0.0

        # Text only
        X_text = np.column_stack([np.ones(len(y)), x_cos])
        c_text, _, _, _ = np.linalg.lstsq(X_text, y, rcond=None)
        r2_text = 1.0 - np.sum((y - X_text @ c_text) ** 2) / ss_tot if ss_tot > 0 else 0.0

        # SIC only
        X_sic = np.column_stack([np.ones(len(y)), x_sic])
        c_sic, _, _, _ = np.linalg.lstsq(X_sic, y, rcond=None)
        r2_sic = 1.0 - np.sum((y - X_sic @ c_sic) ** 2) / ss_tot if ss_tot > 0 else 0.0

        # Rank correlation
        rho_text, p_text = sp_stats.spearmanr(x_cos, y)
        rho_sic, p_sic = sp_stats.spearmanr(x_sic, y)

        all_regressions.append({
            "date": rebal_str,
            "n_firms": n,
            "n_pairs": len(y),
            "r2_both": float(r2_both),
            "r2_text_only": float(r2_text),
            "r2_sic_only": float(r2_sic),
            "incremental_r2_text": float(r2_both - r2_sic),
            "incremental_r2_sic": float(r2_both - r2_text),
            "b_cosine": float(c_both[1]),
            "b_sic": float(c_both[2]),
            "spearman_text": float(rho_text),
            "spearman_sic": float(rho_sic),
            "p_text": float(p_text),
            "p_sic": float(p_sic),
            "pct_same_sic": float(x_sic.mean()),
        })

    # Average across dates
    if not all_regressions:
        return {"error": "no_valid_dates"}

    df = pd.DataFrame(all_regressions)
    summary = {}
    for col in ["r2_both", "r2_text_only", "r2_sic_only",
                 "incremental_r2_text", "incremental_r2_sic",
                 "b_cosine", "b_sic", "spearman_text", "spearman_sic"]:
        vals = df[col].values
        summary[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "t_stat": _nw_tstat(vals),
        }

    return {
        "summary": summary,
        "per_date": all_regressions,
    }


# ---------------------------------------------------------------------------
# Phase B: Backtest comparison
# ---------------------------------------------------------------------------

def run_backtest_comparison(
    doc_embeddings: dict,
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    sic_codes: dict[str, str],
    p: int = 200,
    start: str = "2013-04-01",
    end: str = "2024-04-01",
    lookback_days: int = 504,
    max_weight: float = 0.05,
) -> dict:
    """Phase B: Walk-forward backtest of text vs SIC vs LW vs EW."""

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    rebal_dates = generate_quarterly_dates(start_date, end_date)

    aggregator = FirmAggregator(EmbeddingConfig())
    portfolio_config = PortfolioConfig(
        objective="min_variance", long_only=True, max_weight=max_weight,
    )
    cov_config = CovarianceConfig(method="shrinkage", shrinkage_intensity="auto")
    backtest_config = BacktestConfig(start_date=start_date, end_date=end_date)
    liq_config = LiquidityConfig(max_firms=p, covariance_lookback_days=lookback_days)
    optimizer = PortfolioOptimizer(portfolio_config)
    tc_model = TransactionCostModel()

    strategies = ["semantic", "sic_sector", "lw", "equal_weight"]
    weights_history: dict[str, dict[date, PortfolioWeights]] = {
        s: {} for s in strategies
    }
    tc_rates_history: dict[date, pd.Series] = {}

    for rebal_date in tqdm(rebal_dates, desc=f"Phase B p={p}"):
        rebal_str = str(rebal_date)

        firm_embeddings = aggregator.aggregate_universe(
            doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_config,
        )
        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before = daily_returns.loc[daily_returns.index < rebal_str]
        lookback_rets = returns_before[eligible].iloc[-lookback_days:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]

        if len(eligible) < 20:
            continue

        # TC rates
        dv_before = dollar_volume.loc[dollar_volume.index < rebal_str]
        adv_window = dv_before.iloc[-liq_config.adv_lookback_days:]
        median_adv = adv_window.median()
        tc_rates_history[rebal_date] = tc_model.get_tc_rates(
            median_adv, daily_returns, rebal_str,
        )

        sub_emb = {t: firm_embeddings[t] for t in eligible}
        hist_rets = lookback_rets[eligible].fillna(0.0)
        sample_stds = hist_rets.std()

        # Semantic shrinkage
        try:
            sem_est = SemanticShrinkageCovariance(cov_config, hist_rets)
            cov = sem_est.estimate(sub_emb)
            w = optimizer.optimize(cov)
            w.metadata["method"] = "semantic"
            weights_history["semantic"][rebal_date] = w
        except Exception as e:
            logger.error("Semantic failed: %s", e)

        # SIC sector
        try:
            sic_cov = sic_sector_covariance(
                eligible, sic_codes, sigma_estimates=sample_stds,
            )
            w = optimizer.optimize(sic_cov)
            w.metadata["method"] = "sic_sector"
            weights_history["sic_sector"][rebal_date] = w
        except Exception as e:
            logger.error("SIC sector failed: %s", e)

        # LW
        try:
            w = ledoit_wolf_portfolio(
                hist_rets,
                objective=portfolio_config.objective,
                long_only=portfolio_config.long_only,
                max_weight=portfolio_config.max_weight,
            )
            weights_history["lw"][rebal_date] = w
        except Exception as e:
            logger.error("LW failed: %s", e)

        # EW
        w = equal_weight_portfolio(eligible)
        weights_history["equal_weight"][rebal_date] = w

    # Run backtests
    engine = BacktestEngine(backtest_config)
    bt_results: list[BacktestResult] = []

    for name in strategies:
        wh = weights_history[name]
        if not wh:
            continue
        try:
            bt = engine.run(
                wh, daily_returns,
                strategy_name=name,
                tc_rates_history=tc_rates_history,
            )
            bt_results.append(bt)
        except Exception as e:
            logger.error("Backtest failed for %s: %s", name, e)

    metrics = {r.strategy_name: r.metrics for r in bt_results}

    # Statistical tests
    stat_tests: dict = {}
    if len(bt_results) >= 2:
        all_rets = {r.strategy_name: r.portfolio_returns for r in bt_results}
        common_idx = bt_results[0].portfolio_returns.index
        for r in bt_results[1:]:
            common_idx = common_idx.intersection(r.portfolio_returns.index)
        aligned = {name: rets.loc[common_idx].values for name, rets in all_rets.items()}

        for a, b in [("semantic", "sic_sector"), ("semantic", "lw"),
                      ("sic_sector", "lw"), ("sic_sector", "equal_weight")]:
            if a in aligned and b in aligned:
                test = block_permutation_test(
                    aligned[a], aligned[b], block_size=63, n_permutations=10000,
                )
                stat_tests[f"{a}_vs_{b}"] = {
                    "diff_means": float(test.statistic),
                    "p_value": float(test.p_value),
                }

    return {
        "metrics": metrics,
        "statistical_tests": stat_tests,
    }


# ---------------------------------------------------------------------------
# Phase C: Case studies (disagreement pairs)
# ---------------------------------------------------------------------------

def find_disagreement_pairs(
    doc_embeddings: dict,
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    sic_codes: dict[str, str],
    p: int = 200,
    rebal_date_str: str = "2023-01-01",
    lookback_days: int = 504,
    forward_days: int = 63,
    n_pairs: int = 20,
) -> dict:
    """Phase C: Find pairs where text and SIC most disagree."""

    rebal_date = date.fromisoformat(rebal_date_str)
    aggregator = FirmAggregator(EmbeddingConfig())
    liq_config = LiquidityConfig(max_firms=p)

    firm_embeddings = aggregator.aggregate_universe(
        doc_embeddings, as_of_date=rebal_date, min_documents=1,
    )
    eligible = filter_investable_universe(
        firm_embeddings, daily_returns, dollar_volume,
        rebal_date, config=liq_config,
    )
    eligible = [t for t in eligible if t in daily_returns.columns]
    returns_before = daily_returns.loc[daily_returns.index < rebal_date_str]
    lookback_rets = returns_before[eligible].iloc[-lookback_days:]
    completeness = lookback_rets.notna().sum() / len(lookback_rets)
    eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]

    n = len(eligible)
    if n < 50:
        return {"error": "insufficient_firms"}

    # Cosine similarity
    E = np.array([firm_embeddings[t].embedding for t in eligible])
    cosine_sim = E @ E.T
    np.clip(cosine_sim, -1.0, 1.0, out=cosine_sim)

    # Forward realized correlation
    simple_returns = np.exp(daily_returns) - 1.0
    fwd = simple_returns.loc[simple_returns.index >= rebal_date_str]
    fwd = fwd[eligible].iloc[:forward_days]
    realized_corr = fwd.corr().values

    # SIC match
    sic_match = np.zeros((n, n), dtype=bool)
    for i in range(n):
        si = sic_codes.get(eligible[i])
        if si is None:
            continue
        for j in range(i + 1, n):
            sj = sic_codes.get(eligible[j])
            if sj is not None and si == sj:
                sic_match[i, j] = True
                sic_match[j, i] = True

    # Type 1: Same SIC, low cosine similarity (SIC says related, text says not)
    same_sic_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sic_match[i, j]:
                same_sic_pairs.append((i, j, cosine_sim[i, j], realized_corr[i, j]))

    same_sic_pairs.sort(key=lambda x: x[2])  # Sort by cosine (ascending)
    type1_pairs = []
    for i, j, cos, real_corr in same_sic_pairs[:n_pairs]:
        type1_pairs.append({
            "ticker_a": eligible[i],
            "ticker_b": eligible[j],
            "sic_a": sic_codes.get(eligible[i], "unknown"),
            "sic_b": sic_codes.get(eligible[j], "unknown"),
            "cosine_sim": float(cos),
            "realized_corr": float(real_corr) if not np.isnan(real_corr) else None,
            "type": "same_sic_low_cosine",
        })

    # Type 2: Different SIC, high cosine (text says related, SIC says not)
    diff_sic_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not sic_match[i, j] and sic_codes.get(eligible[i]) and sic_codes.get(eligible[j]):
                diff_sic_pairs.append((i, j, cosine_sim[i, j], realized_corr[i, j]))

    diff_sic_pairs.sort(key=lambda x: -x[2])  # Sort by cosine (descending)
    type2_pairs = []
    for i, j, cos, real_corr in diff_sic_pairs[:n_pairs]:
        type2_pairs.append({
            "ticker_a": eligible[i],
            "ticker_b": eligible[j],
            "sic_a": sic_codes.get(eligible[i], "unknown"),
            "sic_b": sic_codes.get(eligible[j], "unknown"),
            "cosine_sim": float(cos),
            "realized_corr": float(real_corr) if not np.isnan(real_corr) else None,
            "type": "diff_sic_high_cosine",
        })

    # Compute actual mean realized correlations for same-SIC and diff-SIC pairs
    same_sic_realized = [rc for _, _, _, rc in same_sic_pairs if np.isfinite(rc)]
    diff_sic_realized = [rc for _, _, _, rc in diff_sic_pairs if np.isfinite(rc)]
    sic_pred_same = float(np.mean(same_sic_realized)) if same_sic_realized else 0.5
    sic_pred_diff = float(np.mean(diff_sic_realized)) if diff_sic_realized else 0.2

    # Compute which predictor was closer to realized
    text_wins_t1 = sum(
        1 for p in type1_pairs
        if p["realized_corr"] is not None
        and abs(p["cosine_sim"] - p["realized_corr"]) < abs(sic_pred_same - p["realized_corr"])
    )
    text_wins_t2 = sum(
        1 for p in type2_pairs
        if p["realized_corr"] is not None
        and abs(p["cosine_sim"] - p["realized_corr"]) < abs(sic_pred_diff - p["realized_corr"])
    )

    return {
        "rebal_date": rebal_date_str,
        "n_firms": n,
        "n_same_sic_pairs": len(same_sic_pairs),
        "n_diff_sic_pairs": len(diff_sic_pairs),
        "sic_pred_same": sic_pred_same,
        "sic_pred_diff": sic_pred_diff,
        "same_sic_low_cosine": type1_pairs,
        "diff_sic_high_cosine": type2_pairs,
        "text_wins_type1": text_wins_t1,
        "text_wins_type2": text_wins_t2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Text vs SIC sector analysis")
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--text-source", default="10k",
                        choices=["10k", "transcript", "news", "combined", "all"])
    parser.add_argument("--start", default="2007-04-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip Phase B (backtest) for faster iteration")
    args = parser.parse_args()

    doc_embeddings = load_doc_embeddings(args.text_source)
    _, daily_returns, dollar_volume = load_price_data(list(doc_embeddings.keys()))
    sic_codes = load_sic_codes()
    logger.info("Loaded %d SIC codes", len(sic_codes))

    t0 = time.time()

    # Phase A: Cross-sectional analysis
    logger.info("=" * 60)
    logger.info("PHASE A: Cross-sectional fit")
    logger.info("=" * 60)
    phase_a = run_cross_sectional_analysis(
        doc_embeddings, daily_returns, dollar_volume,
        sic_codes, p=args.p,
        start=args.start, end=args.end,
    )

    # Phase B: Backtest comparison
    phase_b = {}
    if not args.skip_backtest:
        logger.info("=" * 60)
        logger.info("PHASE B: Backtest comparison")
        logger.info("=" * 60)
        phase_b = run_backtest_comparison(
            doc_embeddings, daily_returns, dollar_volume,
            sic_codes, p=args.p,
            start=args.start, end=args.end,
        )

    # Phase C: Case studies
    logger.info("=" * 60)
    logger.info("PHASE C: Case studies")
    logger.info("=" * 60)
    phase_c = find_disagreement_pairs(
        doc_embeddings, daily_returns, dollar_volume,
        sic_codes, p=args.p,
        rebal_date_str="2023-01-01",
    )

    elapsed = time.time() - t0

    results = {
        "experiment": "sector_analysis",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "p": args.p,
            "text_source": args.text_source,
            "start": args.start,
            "end": args.end,
        },
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_c": phase_c,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / "results" / f"sector_analysis_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Done in %.1f seconds. Results saved to %s", elapsed, out_path)

    # Print summary
    print(f"\n{'='*70}")
    print(f"TEXT vs SIC SECTOR ANALYSIS (p={args.p}, {args.text_source})")
    print(f"{'='*70}")

    if "summary" in phase_a:
        s = phase_a["summary"]
        print(f"\nPhase A: Cross-Sectional Fit (realized_corr ~ cosine + SIC)")
        print(f"  R² (cosine only):      {s['r2_text_only']['mean']:.4f} (t={s['r2_text_only']['t_stat']:.2f})")
        print(f"  R² (SIC only):         {s['r2_sic_only']['mean']:.4f} (t={s['r2_sic_only']['t_stat']:.2f})")
        print(f"  R² (both):             {s['r2_both']['mean']:.4f}")
        print(f"  Incremental R² (text): {s['incremental_r2_text']['mean']:.4f} (t={s['incremental_r2_text']['t_stat']:.2f})")
        print(f"  Incremental R² (SIC):  {s['incremental_r2_sic']['mean']:.4f} (t={s['incremental_r2_sic']['t_stat']:.2f})")
        print(f"  Spearman (text):       {s['spearman_text']['mean']:.4f}")
        print(f"  Spearman (SIC):        {s['spearman_sic']['mean']:.4f}")

    if "metrics" in phase_b:
        print(f"\nPhase B: Backtest (Sharpe Ratios):")
        for name, m in phase_b["metrics"].items():
            sharpe = m.get("sharpe_ratio", "N/A")
            if isinstance(sharpe, (int, float)):
                print(f"  {name:20s}: {sharpe:.3f}")

        if "statistical_tests" in phase_b:
            print(f"\n  Statistical tests:")
            for pair, t in phase_b["statistical_tests"].items():
                print(f"    {pair}: p={t['p_value']:.3f}")

    if "same_sic_low_cosine" in phase_c:
        print(f"\nPhase C: Disagreement Pairs (at 2023-01-01)")
        print(f"  Same SIC, low cosine ({len(phase_c['same_sic_low_cosine'])} pairs):")
        for p_info in phase_c["same_sic_low_cosine"][:5]:
            print(f"    {p_info['ticker_a']}-{p_info['ticker_b']}: "
                  f"SIC={p_info['sic_a']}, cos={p_info['cosine_sim']:.3f}, "
                  f"realized={p_info['realized_corr']:.3f}" if p_info['realized_corr'] else "")
        print(f"  Different SIC, high cosine ({len(phase_c['diff_sic_high_cosine'])} pairs):")
        for p_info in phase_c["diff_sic_high_cosine"][:5]:
            print(f"    {p_info['ticker_a']}-{p_info['ticker_b']}: "
                  f"SIC={p_info['sic_a']}/{p_info['sic_b']}, "
                  f"cos={p_info['cosine_sim']:.3f}, "
                  f"realized={p_info['realized_corr']:.3f}" if p_info['realized_corr'] else "")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
