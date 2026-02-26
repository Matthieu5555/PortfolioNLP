"""
Correlation study: does embedding similarity predict return correlation?

Runs the foundational validation across multiple quarterly dates.
If cosine_sim(e_i, e_j) does NOT rank-correlate with forward realized
return correlation, the entire semantic covariance approach has no value.

Usage:
    uv run python scripts/run_correlation_study.py
    uv run python scripts/run_correlation_study.py --start 2015-01-01 --end 2022-12-31
    uv run python scripts/run_correlation_study.py --horizon 126  # semiannual
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR, EmbeddingConfig
from pnlp.data.embeddings_loader import load_doc_embeddings
from pnlp.data.prices import PriceFetcher
from pnlp.data.store import DocumentStore
from pnlp.embeddings.document_embedder import DocumentEmbedding
from pnlp.embeddings.firm_aggregator import FirmAggregator
from pnlp.research.correlation_study import run_correlation_study
from pnlp.data.dates import generate_quarterly_dates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_all_doc_embeddings(store: DocumentStore) -> dict[str, list[DocumentEmbedding]]:
    """Load all ticker embeddings from .npz files into DocumentEmbedding objects."""
    emb_dir = DATA_DIR / "embeddings"
    all_docs: dict[str, list[DocumentEmbedding]] = {}
    npz_files = sorted(emb_dir.glob("*.npz"))
    for npz_file in tqdm(npz_files, desc="Loading embeddings"):
        ticker = npz_file.stem
        docs = store.load_document_embeddings(ticker)
        if docs:
            all_docs[ticker] = docs
    total_docs = sum(len(v) for v in all_docs.values())
    logger.info("Loaded %d tickers, %d total documents", len(all_docs), total_docs)
    return all_docs


def load_all_returns(tickers: list[str]) -> pd.DataFrame:
    """Load cached price data and compute daily log returns."""
    fetcher = PriceFetcher()
    prices = fetcher.fetch(tickers, start=date(2000, 1, 1), end=date(2025, 12, 31))
    logger.info("Loaded prices for %d / %d tickers", len(prices), len(tickers))
    returns = PriceFetcher.compute_returns(prices, frequency="daily")
    logger.info("Returns matrix: %d days x %d tickers", *returns.shape)
    return returns



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding vs return correlation study")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2007, 1, 1),
                        help="First study date (default: 2007-01-01)")
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 6, 30),
                        help="Last study date (default: 2024-06-30)")
    parser.add_argument("--horizon", type=int, default=252,
                        help="Forward return horizon in trading days (default: 252)")
    parser.add_argument("--gap-days", type=int, default=5,
                        help="Gap days before forward window (default: 5)")
    parser.add_argument("--text-source", default="10k",
                        choices=["10k", "transcript", "news", "combined", "all"],
                        help="Text source for embeddings (default: 10k)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: data/results/correlation_study.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load all document embeddings
    all_doc_embeddings = load_doc_embeddings(args.text_source)

    # 2. Load daily returns for all tickers with embeddings
    daily_returns = load_all_returns(list(all_doc_embeddings.keys()))

    # 3. Generate quarterly study dates
    study_dates = generate_quarterly_dates(args.start, args.end)
    logger.info("Running correlation study for %d quarterly dates (%s to %s)",
                len(study_dates), study_dates[0], study_dates[-1])

    # 4. Aggregate firm embeddings and run study at each date
    aggregator = FirmAggregator(EmbeddingConfig())
    results = []

    for study_date in tqdm(study_dates, desc="Correlation study"):
        firm_embeddings = aggregator.aggregate_universe(
            all_doc_embeddings, as_of_date=study_date, min_documents=1,
        )

        if len(firm_embeddings) < 30:
            logger.warning("Only %d firms at %s, skipping", len(firm_embeddings), study_date)
            continue

        result = run_correlation_study(
            firm_embeddings=firm_embeddings,
            returns=daily_returns,
            horizon_days=args.horizon,
            gap_days=args.gap_days,
        )

        results.append({
            "date": str(study_date),
            "spearman_rho": result.spearman_rho,
            "spearman_pvalue": result.spearman_pvalue,
            "n_pairs": result.n_pairs,
            "n_firms": result.n_firms,
        })

        logger.info(
            "  %s: rho=%.4f, p=%.6f, n_firms=%d, n_pairs=%d",
            study_date, result.spearman_rho, result.spearman_pvalue,
            result.n_firms, result.n_pairs,
        )

    # 5. Summarize
    if not results:
        logger.error("No valid study dates produced results")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("CORRELATION STUDY SUMMARY")
    print("=" * 70)
    print(f"Quarters analyzed:    {len(df)}")
    print(f"Average Spearman rho: {df['spearman_rho'].mean():.4f}")
    print(f"Median Spearman rho:  {df['spearman_rho'].median():.4f}")
    print(f"Std Spearman rho:     {df['spearman_rho'].std():.4f}")
    print(f"Min rho:              {df['spearman_rho'].min():.4f}")
    print(f"Max rho:              {df['spearman_rho'].max():.4f}")
    sig_count = (df["spearman_pvalue"] < 0.05).sum()
    print(f"Significant (p<0.05): {sig_count}/{len(df)} ({100 * sig_count / len(df):.0f}%)")
    print(f"Avg firms per date:   {df['n_firms'].mean():.0f}")
    print(f"Avg pairs per date:   {df['n_pairs'].mean():.0f}")
    print()

    avg_rho = df["spearman_rho"].mean()
    if avg_rho > 0.10:
        print("VERDICT: Strong signal. Embedding similarity predicts return correlation.")
    elif avg_rho > 0.05:
        print("VERDICT: Moderate signal. Proceed with caution.")
    else:
        print("VERDICT: Weak or no signal. Reconsider the approach.")
    print("=" * 70)

    # 6. Save results
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output or str(output_dir / "correlation_study.json")

    output = {
        "config": {
            "start": str(args.start),
            "end": str(args.end),
            "horizon": args.horizon,
            "gap_days": args.gap_days,
        },
        "summary": {
            "mean_rho": float(df["spearman_rho"].mean()),
            "median_rho": float(df["spearman_rho"].median()),
            "std_rho": float(df["spearman_rho"].std()),
            "n_quarters": len(df),
            "pct_significant": float(100 * sig_count / len(df)),
        },
        "per_quarter": df.to_dict(orient="records"),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
