"""
Text-Based Beta Estimation Experiment.

Tests whether projection onto the first principal component of the embedding
space ("text-beta") predicts future realized CAPM beta better than trailing
return-based beta.

Key questions:
- Does text-beta have incremental predictive power beyond trailing beta?
- Is text-beta more useful for firms with short return histories?
- Can a single 10-K filing estimate a firm's market sensitivity?

Usage:
    uv run python experiments/run_beta_experiment.py
    uv run python experiments/run_beta_experiment.py --p 500 --text-source 10k
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
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR, EmbeddingConfig
from pnlp.data.embeddings_loader import load_doc_embeddings
from pnlp.data.fama_french import load_ff_factors
from pnlp.data.prices import load_price_data
from pnlp.data.universe_filter import LiquidityConfig, filter_investable_universe
from pnlp.embeddings.firm_aggregator import FirmAggregator
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


def compute_capm_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
) -> float | None:
    """Compute CAPM beta from return series."""
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 20:
        return None
    r_stock = aligned.iloc[:, 0].values
    r_mkt = aligned.iloc[:, 1].values
    cov = np.cov(r_stock, r_mkt)
    var_mkt = cov[1, 1]
    if var_mkt < 1e-12:
        return None
    return float(cov[0, 1] / var_mkt)


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
# Main experiment
# ---------------------------------------------------------------------------

def run_beta_experiment(
    p: int = 500,
    text_source: str = "10k",
    start: str = "2013-04-01",
    end: str = "2024-04-01",
    trailing_lookback: int = 252,
    forward_horizon: int = 63,
) -> dict:
    """Run the text-based beta prediction experiment."""

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    # Load data
    doc_embeddings = load_doc_embeddings(text_source)
    _, daily_returns, dollar_volume = load_price_data(list(doc_embeddings.keys()))

    # Convert to simple returns for beta computation
    simple_returns = np.exp(daily_returns) - 1.0

    # Load FF factors
    ff_factors = load_ff_factors(start="2011-01-01", end="2025-01-01")
    mkt_rf = ff_factors["Mkt-RF"]

    # Generate rebalance dates
    rebal_dates = generate_quarterly_dates(start_date, end_date)
    logger.info("%d quarterly rebalance dates from %s to %s",
                len(rebal_dates), rebal_dates[0], rebal_dates[-1])

    aggregator = FirmAggregator(EmbeddingConfig())
    liq_config = LiquidityConfig(max_firms=p)

    # Storage for cross-sectional results
    all_fm_results = []  # Fama-MacBeth regressions
    all_rank_corrs = []  # Spearman rank correlations
    all_subgroup = defaultdict(list)  # By return history length

    prev_scores = None
    prev_tickers = None

    for rebal_date in tqdm(rebal_dates[:-1], desc="Beta experiment"):
        # Need at least forward_horizon days after rebal_date
        rebal_str = str(rebal_date)
        returns_after = simple_returns.loc[simple_returns.index >= rebal_str]
        if len(returns_after) < forward_horizon:
            continue

        # Aggregate embeddings
        firm_embeddings = aggregator.aggregate_universe(
            doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )

        # Filter universe
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, liq_config,
        )
        if len(eligible) < 50:
            continue

        # --- Step 1: SVD for text-beta ---
        E = np.array([firm_embeddings[t].embedding for t in eligible])
        U, S, Vt = np.linalg.svd(E, full_matrices=False)
        pc1_scores = U[:, 0]  # (n_firms,) — projection onto PC1

        # Sign-align with previous quarter
        if prev_scores is not None and prev_tickers is not None:
            common = [t for t in eligible if t in prev_tickers]
            if len(common) >= 20:
                curr_idx = [eligible.index(t) for t in common]
                prev_idx = [prev_tickers.index(t) for t in common]
                corr = np.corrcoef(
                    pc1_scores[curr_idx],
                    prev_scores[prev_idx],
                )[0, 1]
                if corr < 0:
                    pc1_scores = -pc1_scores

        prev_scores = pc1_scores.copy()
        prev_tickers = list(eligible)

        text_beta = pd.Series(pc1_scores, index=eligible, name="text_beta")

        # --- Step 2: Trailing realized beta ---
        returns_before = simple_returns.loc[simple_returns.index < rebal_str]
        trailing_returns = returns_before.iloc[-trailing_lookback:]
        mkt_trailing = mkt_rf.loc[mkt_rf.index.isin(trailing_returns.index)]

        trailing_betas = {}
        return_history_days = {}
        for t in eligible:
            if t not in trailing_returns.columns:
                continue
            r = trailing_returns[t].dropna()
            return_history_days[t] = len(r)
            beta = compute_capm_beta(r, mkt_trailing)
            if beta is not None:
                trailing_betas[t] = beta

        trailing_beta = pd.Series(trailing_betas, name="trailing_beta")

        # --- Step 3: Forward realized beta ---
        forward_returns = returns_after.iloc[:forward_horizon]
        mkt_forward = mkt_rf.loc[mkt_rf.index.isin(forward_returns.index)]

        forward_betas = {}
        for t in eligible:
            if t not in forward_returns.columns:
                continue
            r = forward_returns[t].dropna()
            beta = compute_capm_beta(r, mkt_forward)
            if beta is not None:
                forward_betas[t] = beta

        forward_beta = pd.Series(forward_betas, name="forward_beta")

        # --- Step 4: Cross-sectional analysis ---
        # Align all series
        common_tickers = sorted(
            set(text_beta.index)
            & set(trailing_beta.index)
            & set(forward_beta.index)
        )
        if len(common_tickers) < 30:
            continue

        tb = text_beta.loc[common_tickers].values
        tr = trailing_beta.loc[common_tickers].values
        fb = forward_beta.loc[common_tickers].values

        # Rank correlations
        rho_text, p_text = stats.spearmanr(tb, fb)
        rho_trail, p_trail = stats.spearmanr(tr, fb)

        all_rank_corrs.append({
            "date": str(rebal_date),
            "n_firms": len(common_tickers),
            "spearman_text_forward": float(rho_text),
            "p_text": float(p_text),
            "spearman_trailing_forward": float(rho_trail),
            "p_trailing": float(p_trail),
        })

        # Fama-MacBeth cross-sectional regression:
        # forward_beta = a + b1*text_beta + b2*trailing_beta + eps
        X = np.column_stack([np.ones(len(common_tickers)), tb, tr])
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, fb, rcond=None)
            y_hat = X @ coeffs
            ss_res = np.sum((fb - y_hat) ** 2)
            ss_tot = np.sum((fb - fb.mean()) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Also run univariate regressions for incremental R²
            # Text-only
            X_text = np.column_stack([np.ones(len(common_tickers)), tb])
            c_text, _, _, _ = np.linalg.lstsq(X_text, fb, rcond=None)
            ss_res_text = np.sum((fb - X_text @ c_text) ** 2)
            r2_text = 1.0 - ss_res_text / ss_tot if ss_tot > 0 else 0.0

            # Trailing-only
            X_trail = np.column_stack([np.ones(len(common_tickers)), tr])
            c_trail, _, _, _ = np.linalg.lstsq(X_trail, fb, rcond=None)
            ss_res_trail = np.sum((fb - X_trail @ c_trail) ** 2)
            r2_trail = 1.0 - ss_res_trail / ss_tot if ss_tot > 0 else 0.0

            all_fm_results.append({
                "date": str(rebal_date),
                "intercept": float(coeffs[0]),
                "b_text": float(coeffs[1]),
                "b_trailing": float(coeffs[2]),
                "r_squared_both": float(r_squared),
                "r_squared_text_only": float(r2_text),
                "r_squared_trailing_only": float(r2_trail),
                "incremental_r2_text": float(r_squared - r2_trail),
                "incremental_r2_trailing": float(r_squared - r2_text),
            })
        except Exception as e:
            logger.warning("Regression failed at %s: %s", rebal_date, e)

        # --- Step 5: Subgroup analysis by return history length ---
        history_days = pd.Series(return_history_days)
        for t in common_tickers:
            h = history_days.get(t, 504)
            if h < 63:
                group = "short_<63d"
            elif h < 126:
                group = "medium_63-126d"
            elif h < 252:
                group = "medium_126-252d"
            else:
                group = "long_>252d"

            all_subgroup[group].append({
                "date": str(rebal_date),
                "text_beta": float(text_beta.loc[t]),
                "trailing_beta": float(trailing_beta.loc[t]),
                "forward_beta": float(forward_beta.loc[t]),
            })

    # --- Aggregate results ---
    logger.info("Computing aggregate statistics over %d quarters", len(all_fm_results))

    # Fama-MacBeth: average coefficients, Newey-West t-stats
    fm_df = pd.DataFrame(all_fm_results)
    fm_summary = {}
    for col in ["intercept", "b_text", "b_trailing", "r_squared_both",
                "r_squared_text_only", "r_squared_trailing_only",
                "incremental_r2_text", "incremental_r2_trailing"]:
        vals = fm_df[col].values
        fm_summary[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "t_stat": _nw_tstat(vals),
            "n_quarters": len(vals),
        }

    # Rank correlations: average
    rc_df = pd.DataFrame(all_rank_corrs)
    rank_summary = {
        "avg_spearman_text_forward": float(rc_df["spearman_text_forward"].mean()),
        "avg_spearman_trailing_forward": float(rc_df["spearman_trailing_forward"].mean()),
        "std_spearman_text_forward": float(rc_df["spearman_text_forward"].std()),
        "std_spearman_trailing_forward": float(rc_df["spearman_trailing_forward"].std()),
        "pct_text_significant": float((rc_df["p_text"] < 0.05).mean()),
        "pct_trailing_significant": float((rc_df["p_trailing"] < 0.05).mean()),
        "n_quarters": len(rc_df),
        # Which predictor wins more often?
        "text_beats_trailing_pct": float(
            (rc_df["spearman_text_forward"].abs() > rc_df["spearman_trailing_forward"].abs()).mean()
        ),
    }

    # Subgroup analysis — per-quarter Spearman, then average (Fama-MacBeth style)
    subgroup_summary = {}
    for group, records in all_subgroup.items():
        df = pd.DataFrame(records)
        n = len(df)
        if n < 10:
            continue
        # Compute Spearman within each quarter, then average
        rho_text_list = []
        rho_trail_list = []
        for _, qdf in df.groupby("date"):
            if len(qdf) < 5:
                continue
            rt, _ = stats.spearmanr(qdf["text_beta"], qdf["forward_beta"])
            rr, _ = stats.spearmanr(qdf["trailing_beta"], qdf["forward_beta"])
            if np.isfinite(rt) and np.isfinite(rr):
                rho_text_list.append(rt)
                rho_trail_list.append(rr)
        if len(rho_text_list) < 3:
            continue
        avg_rho_text = float(np.mean(rho_text_list))
        avg_rho_trail = float(np.mean(rho_trail_list))
        subgroup_summary[group] = {
            "n_observations": n,
            "n_quarters": len(rho_text_list),
            "spearman_text_forward": avg_rho_text,
            "std_text": float(np.std(rho_text_list)),
            "spearman_trailing_forward": avg_rho_trail,
            "std_trailing": float(np.std(rho_trail_list)),
            "text_better": bool(abs(avg_rho_text) > abs(avg_rho_trail)),
        }

    results = {
        "experiment": "text_beta",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "p": p,
            "text_source": text_source,
            "start": start,
            "end": end,
            "trailing_lookback": trailing_lookback,
            "forward_horizon": forward_horizon,
        },
        "fama_macbeth": fm_summary,
        "rank_correlations": rank_summary,
        "per_quarter_rank_corr": all_rank_corrs,
        "per_quarter_regression": all_fm_results,
        "subgroup_analysis": subgroup_summary,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Text-based beta experiment")
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--text-source", default="10k", choices=["10k", "transcript", "news", "combined", "all"])
    parser.add_argument("--start", default="2007-04-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    t0 = time.time()
    results = run_beta_experiment(
        p=args.p,
        text_source=args.text_source,
        start=args.start,
        end=args.end,
    )

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / "results" / f"beta_experiment_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info("Done in %.1f seconds. Results saved to %s", elapsed, out_path)

    # Print summary
    rc = results["rank_correlations"]
    fm = results["fama_macbeth"]
    print(f"\n{'='*60}")
    print(f"TEXT-BASED BETA EXPERIMENT (p={args.p}, {args.text_source})")
    print(f"{'='*60}")
    print(f"\nRank Correlations (avg over {rc['n_quarters']} quarters):")
    print(f"  Text-beta → Forward beta:     ρ = {rc['avg_spearman_text_forward']:.4f}")
    print(f"  Trailing-beta → Forward beta:  ρ = {rc['avg_spearman_trailing_forward']:.4f}")
    print(f"  Text beats trailing: {rc['text_beats_trailing_pct']:.1%} of quarters")
    print(f"  Text significant: {rc['pct_text_significant']:.1%} | Trailing significant: {rc['pct_trailing_significant']:.1%}")

    print(f"\nFama-MacBeth Regression (forward_beta ~ text_beta + trailing_beta):")
    print(f"  b_text:     {fm['b_text']['mean']:.4f} (t={fm['b_text']['t_stat']:.2f})")
    print(f"  b_trailing: {fm['b_trailing']['mean']:.4f} (t={fm['b_trailing']['t_stat']:.2f})")
    print(f"  R² (both):  {fm['r_squared_both']['mean']:.4f}")
    print(f"  R² (text only):     {fm['r_squared_text_only']['mean']:.4f}")
    print(f"  R² (trailing only): {fm['r_squared_trailing_only']['mean']:.4f}")
    print(f"  Incremental R² (text | trailing): {fm['incremental_r2_text']['mean']:.4f}")

    print(f"\nSubgroup Analysis:")
    for group, sg in sorted(results["subgroup_analysis"].items()):
        winner = "TEXT" if sg["text_better"] else "TRAIL"
        print(f"  {group}: n={sg['n_observations']}, "
              f"text ρ={sg['spearman_text_forward']:.3f}, "
              f"trail ρ={sg['spearman_trailing_forward']:.3f} → {winner}")


if __name__ == "__main__":
    main()
