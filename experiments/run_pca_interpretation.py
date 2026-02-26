"""
PCA Component Interpretation: Name the text factors.

Extracts principal component loadings from the firm embedding matrix at each
rebalance date, correlates them with firm characteristics (size, volatility,
momentum), maps to SIC sectors, and tests stability across time.

Key questions:
- What does PC1 (93% of variance) represent? Market/size? Tech vs Financials?
- Are the text factors stable over time (Kendall tau of loading rankings)?
- Do they correlate with known firm characteristics?

Usage:
    uv run python experiments/run_pca_interpretation.py
    uv run python experiments/run_pca_interpretation.py --n-components 10 --text-source 10k
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

from pnlp.config import (
    DATA_DIR,
    EmbeddingConfig,
)
from pnlp.data.embeddings_loader import load_doc_embeddings
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


def load_sic_codes() -> dict[str, dict]:
    """Load SIC code cache if available."""
    path = DATA_DIR / "sic_codes.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    logger.warning("SIC codes not found at %s. Run fetch_sic_codes.py first.", path)
    return {}


# ---------------------------------------------------------------------------
# PCA extraction at a single rebalance date
# ---------------------------------------------------------------------------

def extract_pca_loadings(
    firm_embeddings: dict,
    tickers: list[str],
    n_components: int,
) -> dict:
    """SVD of the firm embedding matrix. Returns loadings, scores, explained variance."""
    E = np.array([firm_embeddings[t].embedding for t in tickers])  # (n_firms, embed_dim)
    n, d = E.shape

    # SVD: E = U @ diag(S) @ Vt
    U, S, Vt = np.linalg.svd(E, full_matrices=False)

    # Explained variance from singular values
    total_var = (S ** 2).sum()
    explained_var = (S ** 2) / total_var
    cumulative_var = np.cumsum(explained_var)

    k = min(n_components, len(S))

    # Firm scores on each PC: U[:, j] * S[j] (or just U[:, j] for direction)
    # Using U[:, :k] as firm loadings (orthonormal directions in firm space)
    scores = U[:, :k]  # (n_firms, k)

    return {
        "tickers": tickers,
        "scores": scores,  # (n_firms, k)
        "singular_values": S[:k],
        "explained_variance": explained_var[:k],
        "cumulative_variance": cumulative_var[:k],
        "Vt": Vt[:k],  # (k, embed_dim) — embedding-space directions
    }


def sign_align_pca(
    current_scores: np.ndarray,
    prev_scores: np.ndarray | None,
    current_tickers: list[str],
    prev_tickers: list[str] | None,
) -> np.ndarray:
    """Sign-align PC scores to maintain consistent interpretation across time.

    For each component, flip sign if correlation with previous quarter is negative.
    """
    if prev_scores is None or prev_tickers is None:
        return current_scores

    # Find common tickers
    common = [t for t in current_tickers if t in prev_tickers]
    if len(common) < 20:
        return current_scores

    curr_idx = [current_tickers.index(t) for t in common]
    prev_idx = [prev_tickers.index(t) for t in common]

    aligned = current_scores.copy()
    k = min(current_scores.shape[1], prev_scores.shape[1])

    for j in range(k):
        corr = np.corrcoef(current_scores[curr_idx, j], prev_scores[prev_idx, j])[0, 1]
        if corr < 0:
            aligned[:, j] *= -1

    return aligned


# ---------------------------------------------------------------------------
# Firm characteristic computation
# ---------------------------------------------------------------------------

def compute_firm_characteristics(
    tickers: list[str],
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    rebal_date: date,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Compute per-firm characteristics: log(ADV), trailing vol, 12m momentum."""
    rebal_str = str(rebal_date)
    returns_before = daily_returns.loc[daily_returns.index < rebal_str]
    dv_before = dollar_volume.loc[dollar_volume.index < rebal_str]

    chars = []
    for t in tickers:
        entry = {"ticker": t}

        # Log ADV (median dollar volume over last 21 days)
        if t in dollar_volume.columns:
            adv = dv_before[t].iloc[-21:].median()
            entry["log_adv"] = np.log(adv) if adv > 0 else np.nan
        else:
            entry["log_adv"] = np.nan

        # Trailing volatility (annualized, last 252 days)
        if t in daily_returns.columns:
            rets = returns_before[t].iloc[-lookback_days:]
            entry["trailing_vol"] = float(rets.std() * np.sqrt(252))
        else:
            entry["trailing_vol"] = np.nan

        # 12-month momentum (cumulative return)
        if t in daily_returns.columns:
            rets = returns_before[t].iloc[-lookback_days:]
            entry["momentum_12m"] = float((1 + rets.fillna(0)).prod() - 1)
        else:
            entry["momentum_12m"] = np.nan

        chars.append(entry)

    return pd.DataFrame(chars).set_index("ticker")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_pca_interpretation(
    all_doc_embeddings: dict,
    daily_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    rebalance_dates: list[date],
    aggregator: FirmAggregator,
    liq_config: LiquidityConfig,
    lookback_days: int,
    n_components: int = 10,
    max_firms: int = 500,
) -> dict:
    """Run PCA interpretation across all rebalance dates."""
    sic_codes = load_sic_codes()

    liq_p = LiquidityConfig(
        adv_threshold=liq_config.adv_threshold,
        adv_lookback_days=liq_config.adv_lookback_days,
        min_return_completeness=liq_config.min_return_completeness,
        covariance_lookback_days=liq_config.covariance_lookback_days,
        max_firms=max_firms,
    )

    # Storage across quarters
    all_pc_data: list[dict] = []
    prev_scores = None
    prev_tickers = None

    # Per-PC aggregators
    pc_char_corrs = defaultdict(lambda: defaultdict(list))  # pc_j -> char_name -> [corr values]
    pc_sector_loadings = defaultdict(lambda: defaultdict(list))  # pc_j -> sic_2digit -> [mean loading]
    pc_top_bottom = defaultdict(lambda: {"top": defaultdict(int), "bottom": defaultdict(int)})
    pc_kendall_taus = defaultdict(list)  # pc_j -> [tau values across consecutive quarters]

    for rebal_date in tqdm(rebalance_dates, desc="PCA interpretation"):
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=rebal_date, min_documents=1,
        )
        eligible = filter_investable_universe(
            firm_embeddings, daily_returns, dollar_volume,
            rebal_date, config=liq_p,
        )
        if len(eligible) < 20:
            continue

        # Standard universe filtering
        eligible = [t for t in eligible if t in daily_returns.columns]
        returns_before_T = daily_returns.loc[daily_returns.index < str(rebal_date)]
        lookback_rets = returns_before_T[eligible].iloc[-lookback_days:]
        completeness = lookback_rets.notna().sum() / len(lookback_rets)
        eligible = [t for t in eligible if completeness.get(t, 0) >= 0.8]
        if len(eligible) < 20:
            continue

        sub_embeddings = {t: firm_embeddings[t] for t in eligible}

        # --- Phase B: Extract PC loadings ---
        pca = extract_pca_loadings(sub_embeddings, eligible, n_components)
        scores = pca["scores"]  # (n_firms, k)

        # Sign-align with previous quarter
        scores = sign_align_pca(scores, prev_scores, eligible, prev_tickers)

        # Stability: Kendall tau with previous quarter
        if prev_scores is not None and prev_tickers is not None:
            common = [t for t in eligible if t in prev_tickers]
            if len(common) >= 20:
                curr_idx = [eligible.index(t) for t in common]
                prev_idx_map = {t: i for i, t in enumerate(prev_tickers)}
                prev_idx = [prev_idx_map[t] for t in common]

                k = min(scores.shape[1], prev_scores.shape[1])
                for j in range(k):
                    tau, _ = stats.kendalltau(scores[curr_idx, j], prev_scores[prev_idx, j])
                    pc_kendall_taus[j].append(tau)

        # --- Correlate with firm characteristics ---
        chars = compute_firm_characteristics(
            eligible, daily_returns, dollar_volume, rebal_date, lookback_days,
        )

        k = scores.shape[1]
        for j in range(k):
            pc_scores = pd.Series(scores[:, j], index=eligible)

            for char_name in ["log_adv", "trailing_vol", "momentum_12m"]:
                char_vals = chars[char_name].reindex(eligible)
                valid = char_vals.notna() & pc_scores.notna()
                if valid.sum() >= 20:
                    corr, _ = stats.spearmanr(pc_scores[valid], char_vals[valid])
                    pc_char_corrs[j][char_name].append(corr)

            # Top/bottom 10 firms
            sorted_idx = np.argsort(scores[:, j])
            for rank_pos in range(min(10, len(eligible))):
                top_ticker = eligible[sorted_idx[-(rank_pos + 1)]]
                bot_ticker = eligible[sorted_idx[rank_pos]]
                pc_top_bottom[j]["top"][top_ticker] += 1
                pc_top_bottom[j]["bottom"][bot_ticker] += 1

            # SIC sector analysis
            if sic_codes:
                sector_scores = defaultdict(list)
                for i_firm, t in enumerate(eligible):
                    sic_entry = sic_codes.get(t, {})
                    sic = sic_entry.get("sic")
                    if sic:
                        sic_2 = sic[:2]
                        sector_scores[sic_2].append(scores[i_firm, j])
                for sic_2, s_list in sector_scores.items():
                    pc_sector_loadings[j][sic_2].append(float(np.mean(s_list)))

        # Store per-quarter data
        quarter_data = {
            "date": str(rebal_date),
            "n_firms": len(eligible),
            "explained_variance": pca["explained_variance"].tolist(),
            "cumulative_variance": pca["cumulative_variance"].tolist(),
        }

        # Top/bottom 5 for each PC this quarter
        for j in range(min(k, 5)):
            sorted_idx = np.argsort(scores[:, j])
            quarter_data[f"PC{j+1}_top5"] = [
                [eligible[sorted_idx[-(i + 1)]], round(float(scores[sorted_idx[-(i + 1)], j]), 4)]
                for i in range(min(5, len(eligible)))
            ]
            quarter_data[f"PC{j+1}_bottom5"] = [
                [eligible[sorted_idx[i]], round(float(scores[sorted_idx[i], j]), 4)]
                for i in range(min(5, len(eligible)))
            ]

        all_pc_data.append(quarter_data)

        prev_scores = scores
        prev_tickers = eligible

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    pc_summaries = {}
    for j in range(n_components):
        summary: dict = {
            "component": j + 1,
        }

        # Average explained variance
        all_exp_var = [q["explained_variance"][j] for q in all_pc_data if len(q["explained_variance"]) > j]
        if all_exp_var:
            summary["avg_explained_var"] = round(float(np.mean(all_exp_var)), 4)
            summary["std_explained_var"] = round(float(np.std(all_exp_var)), 4)

        # Characteristic correlations
        for char_name in ["log_adv", "trailing_vol", "momentum_12m"]:
            corrs = pc_char_corrs[j].get(char_name, [])
            if corrs:
                summary[f"avg_corr_{char_name}"] = round(float(np.mean(corrs)), 4)
                summary[f"std_corr_{char_name}"] = round(float(np.std(corrs)), 4)

        # Stability (Kendall tau)
        taus = pc_kendall_taus.get(j, [])
        if taus:
            summary["kendall_tau_mean"] = round(float(np.mean(taus)), 4)
            summary["kendall_tau_std"] = round(float(np.std(taus)), 4)
            summary["kendall_tau_min"] = round(float(np.min(taus)), 4)

        # Most common top/bottom firms
        top_counts = pc_top_bottom[j]["top"]
        bot_counts = pc_top_bottom[j]["bottom"]
        summary["typical_top_10"] = sorted(top_counts.items(), key=lambda x: -x[1])[:10]
        summary["typical_bottom_10"] = sorted(bot_counts.items(), key=lambda x: -x[1])[:10]

        # Sector loadings
        sector_avgs = {}
        for sic_2, loadings in pc_sector_loadings[j].items():
            sector_avgs[sic_2] = round(float(np.mean(loadings)), 4)
        if sector_avgs:
            # Sort by absolute loading
            sorted_sectors = sorted(sector_avgs.items(), key=lambda x: -abs(x[1]))
            summary["top_sectors_positive"] = [(s, v) for s, v in sorted_sectors if v > 0][:5]
            summary["top_sectors_negative"] = [(s, v) for s, v in sorted_sectors if v < 0][:5]

        pc_summaries[f"PC{j+1}"] = summary

    # Stability summary
    stability = {}
    for j in range(n_components):
        taus = pc_kendall_taus.get(j, [])
        if taus:
            stability[f"PC{j+1}_kendall_tau_mean"] = round(float(np.mean(taus)), 4)

    return {
        "n_components": n_components,
        "n_quarters": len(all_pc_data),
        "pc_summaries": pc_summaries,
        "stability": stability,
        "per_quarter": all_pc_data,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA Component Interpretation: name the text factors",
    )
    parser.add_argument("--n-components", type=int, default=10,
                        help="Number of PCs to analyze (default: 10)")
    parser.add_argument("--max-firms", type=int, default=500,
                        help="Universe size (default: 500)")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2007, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
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

    # Load data
    all_doc_embeddings = load_doc_embeddings(args.text_source)
    _, daily_returns, dollar_volume = load_price_data(list(all_doc_embeddings.keys()))

    rebalance_dates = generate_quarterly_dates(args.start, args.end)
    logger.info("%d quarterly rebalance dates from %s to %s",
                len(rebalance_dates), rebalance_dates[0], rebalance_dates[-1])

    embedding_config = EmbeddingConfig()
    liq_config = LiquidityConfig(
        adv_threshold=args.adv_threshold,
        adv_lookback_days=args.adv_lookback,
        covariance_lookback_days=args.lookback_days,
    )
    aggregator = FirmAggregator(embedding_config)

    t0 = time.time()
    results = run_pca_interpretation(
        all_doc_embeddings, daily_returns, dollar_volume,
        rebalance_dates, aggregator, liq_config, args.lookback_days,
        n_components=args.n_components, max_firms=args.max_firms,
    )
    results["runtime_seconds"] = round(time.time() - t0, 1)

    # Print summary
    print("\n" + "=" * 90)
    print("PCA COMPONENT INTERPRETATION")
    print("=" * 90)

    for pc_name, summary in results["pc_summaries"].items():
        exp_var = summary.get("avg_explained_var", 0)
        tau = summary.get("kendall_tau_mean", 0)
        corr_adv = summary.get("avg_corr_log_adv", 0)
        corr_vol = summary.get("avg_corr_trailing_vol", 0)
        corr_mom = summary.get("avg_corr_momentum_12m", 0)

        print(f"\n  {pc_name} (avg explained var: {exp_var:.1%}, stability tau: {tau:.3f})")
        print(f"    Corr with log(ADV): {corr_adv:+.3f}  |  vol: {corr_vol:+.3f}  |  mom: {corr_mom:+.3f}")

        top = summary.get("typical_top_10", [])
        bot = summary.get("typical_bottom_10", [])
        if top:
            top_str = ", ".join(f"{t[0]}({t[1]})" for t in top[:5])
            print(f"    Top firms:    {top_str}")
        if bot:
            bot_str = ", ".join(f"{t[0]}({t[1]})" for t in bot[:5])
            print(f"    Bottom firms: {bot_str}")

        pos_sectors = summary.get("top_sectors_positive", [])
        neg_sectors = summary.get("top_sectors_negative", [])
        if pos_sectors:
            sec_str = ", ".join(f"SIC{s[0]}({s[1]:+.3f})" for s in pos_sectors[:3])
            print(f"    + sectors:    {sec_str}")
        if neg_sectors:
            sec_str = ", ".join(f"SIC{s[0]}({s[1]:+.3f})" for s in neg_sectors[:3])
            print(f"    - sectors:    {sec_str}")

    print(f"\n{'─' * 90}")
    print("STABILITY (Kendall Tau across consecutive quarters)")
    print(f"{'─' * 90}")
    for pc_name, tau_val in results["stability"].items():
        print(f"  {pc_name}: {tau_val:.4f}")

    print(f"\n  Runtime: {results['runtime_seconds']:.1f}s")
    print("=" * 90)

    # Save results
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"pca_interpretation_{ts}.json"

    output = {
        "experiment": "pca_interpretation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_components": args.n_components,
            "max_firms": args.max_firms,
            "start": str(args.start),
            "end": str(args.end),
            "lookback_days": args.lookback_days,
            "text_source": args.text_source,
        },
        **results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Results saved to %s", out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
