"""
Block size sensitivity analysis for statistical tests.

Re-runs the permutation test and bootstrap CI at block_sizes=[42, 63, 126]
using saved daily return parquets. No walk-forward re-run needed.

Usage:
    uv run python experiments/run_block_sensitivity.py
    uv run python experiments/run_block_sensitivity.py --returns-file data/results/returns_XXXX.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnlp.config import DATA_DIR
from pnlp.validation.statistical_tests import block_bootstrap_ci, block_permutation_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_latest_returns_parquet() -> Path | None:
    results_dir = DATA_DIR / "results"
    parquets = sorted(results_dir.glob("returns_*.parquet"))
    return parquets[-1] if parquets else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Block size sensitivity for statistical tests")
    parser.add_argument("--returns-file", type=str, default=None)
    parser.add_argument("--block-sizes", type=str, default="42,63,126")
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(",")]

    returns_path = args.returns_file
    if returns_path is None:
        found = find_latest_returns_parquet()
        if found:
            returns_path = str(found)
        else:
            logger.error("No saved returns parquet found.")
            sys.exit(1)

    logger.info("Loading returns from %s", returns_path)
    returns_df = pd.read_parquet(returns_path)
    logger.info("Returns: %d days, strategies: %s", len(returns_df), list(returns_df.columns))

    strategies = list(returns_df.columns)
    results = {"returns_file": returns_path, "block_sizes": block_sizes, "tests": {}}

    # Pairwise permutation tests at each block size
    print("\n" + "=" * 80)
    print("BLOCK SIZE SENSITIVITY — PERMUTATION TESTS")
    print("=" * 80)

    for bs in block_sizes:
        print(f"\n  Block size = {bs}")
        bs_results = {}
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                name_a, name_b = strategies[i], strategies[j]
                aligned = returns_df[[name_a, name_b]].dropna()
                a_vals = aligned[name_a].values
                b_vals = aligned[name_b].values

                test = block_permutation_test(a_vals, b_vals, block_size=bs, n_permutations=10000)
                key = f"{name_a}_vs_{name_b}"
                bs_results[key] = {
                    "diff_means": float(test.statistic),
                    "p_value": float(test.p_value),
                }
                print(f"    {key}: diff={test.statistic:.6f}, p={test.p_value:.4f}")

        results["tests"][f"block_{bs}"] = bs_results

    # Bootstrap CIs for mean return difference at each block size
    print("\n" + "=" * 80)
    print("BLOCK SIZE SENSITIVITY — BOOTSTRAP CIs")
    print("=" * 80)

    ci_results = {}
    for bs in block_sizes:
        print(f"\n  Block size = {bs}")
        bs_ci = {}
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                name_a, name_b = strategies[i], strategies[j]
                aligned = returns_df[[name_a, name_b]].dropna()
                diffs = (aligned[name_a] - aligned[name_b]).values

                ci = block_bootstrap_ci(diffs, block_size=bs, n_bootstrap=10000)
                key = f"{name_a}_vs_{name_b}"
                bs_ci[key] = {
                    "mean_diff": float(ci.statistic),
                    "ci_low": float(ci.ci_low),
                    "ci_high": float(ci.ci_high),
                }
                print(f"    {key}: mean_diff={ci.statistic:.6f}, 95% CI=[{ci.ci_low:.6f}, {ci.ci_high:.6f}]")

        ci_results[f"block_{bs}"] = bs_ci

    results["bootstrap_ci"] = ci_results

    # Save
    output_dir = DATA_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = str(output_dir / f"block_sensitivity_{timestamp}.json")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", path)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
