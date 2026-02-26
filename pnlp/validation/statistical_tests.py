"""
Statistical tests for comparing portfolio strategies.

Uses block-based methods to respect temporal autocorrelation in
portfolio returns. Includes Lo (2002) Sharpe ratio correction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["TestResult", "block_permutation_test", "block_bootstrap_ci", "lo_sharpe_correction"]


@dataclass(frozen=True)
class TestResult:
    """Result of a statistical test."""

    statistic: float
    p_value: float
    ci_low: float = float("-inf")
    ci_high: float = float("inf")
    description: str = ""


def block_permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    block_size: int = 63,
    n_permutations: int = 10000,
    seed: int = 42,
) -> TestResult:
    """Block sign-flip test for difference in means.

    H0: mean(values_a) == mean(values_b)

    Uses non-overlapping block sign-flip (cf. Carlstein, 1986 for block
    structure) to respect temporal autocorrelation. Block size should match
    the rebalancing frequency (63 trading days = quarterly).

    Args:
        values_a: first sample (e.g., daily returns of strategy A).
        values_b: second sample (e.g., daily returns of strategy B).
        block_size: length of circular blocks.
        n_permutations: number of random permutations.

    Returns:
        TestResult with observed difference and p-value.
    """
    rng = np.random.RandomState(seed)
    observed_diff = float(np.mean(values_a) - np.mean(values_b))

    # Use paired differences if same length (more powerful)
    if len(values_a) == len(values_b):
        diffs = values_a - values_b
        n = len(diffs)

        count = 0
        # Need enough blocks to cover all n elements
        n_blocks = (n + block_size - 1) // block_size
        for _ in range(n_permutations):
            # Circular block sign-flip: randomly flip signs of entire blocks
            signs = rng.choice([-1, 1], size=n_blocks)
            # Expand signs to full length
            sign_vec = np.repeat(signs, block_size)[:n]
            perm_diff = float(np.mean(diffs * sign_vec))
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
    else:
        # Unpaired: circular block permutation of combined sample
        pooled = np.concatenate([values_a, values_b])
        n = len(pooled)
        n_a = len(values_a)

        count = 0
        for _ in range(n_permutations):
            # Random circular shift by blocks
            shift = rng.randint(0, n)
            shifted = np.roll(pooled, shift)
            perm_diff = shifted[:n_a].mean() - shifted[n_a:].mean()
            if abs(perm_diff) >= abs(observed_diff):
                count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return TestResult(
        statistic=observed_diff,
        p_value=p_value,
        description=f"Block permutation test (block={block_size}): diff={observed_diff:.6f}, p={p_value:.4f}",
    )


def block_bootstrap_ci(
    values: np.ndarray,
    block_size: int = 63,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Block bootstrap confidence interval for the mean.

    Respects temporal autocorrelation by resampling blocks of
    consecutive observations rather than individual values.

    Block size of 63 matches quarterly rebalancing frequency.

    Args:
        values: time series of values.
        block_size: length of each block (63 = quarterly).
        n_bootstrap: number of bootstrap replications.
        alpha: significance level (0.05 = 95% CI).

    Returns:
        TestResult with mean and confidence interval.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    if n < block_size:
        logger.warning("n=%d < block_size=%d; bootstrap will use a single repeated block", n, block_size)
    n_blocks = max(1, n // block_size)

    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.randint(0, max(1, n - block_size + 1), size=n_blocks)
        sample = np.concatenate([values[s : s + block_size] for s in starts])[:n]
        means[i] = sample.mean()

    ci_low = float(np.percentile(means, 100 * alpha / 2))
    ci_high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    observed_mean = float(np.mean(values))

    return TestResult(
        statistic=observed_mean,
        p_value=0.0,  # CI-based, no p-value
        ci_low=ci_low,
        ci_high=ci_high,
        description=f"Block bootstrap (block={block_size}): mean={observed_mean:.6f}, {100*(1-alpha):.0f}% CI=[{ci_low:.6f}, {ci_high:.6f}]",
    )


def lo_sharpe_correction(
    returns: np.ndarray,
    max_lag: int = 63,
) -> float:
    """Lo (2002) correction factor for annualized Sharpe ratio.

    Adjusts for serial autocorrelation in portfolio returns:
        SR_adj = SR / eta
    where:
        eta = sqrt((1 + 2 * sum_{k=1}^{q} (1 - k/(q+1)) * rho_k) / 252)

    This uses the Bartlett kernel (triangular weights) to ensure
    a positive semi-definite spectral density estimate.

    Args:
        returns: daily portfolio returns.
        max_lag: maximum lag for autocorrelation (default: 63 = quarterly).

    Returns:
        Correction factor eta. Multiply annualized SR by 1/eta to get
        the corrected SR. Returns 1.0 if correction is not meaningful.
    """
    n = len(returns)
    if n < max_lag + 10:
        return 1.0

    # Compute autocorrelations with Bartlett kernel weights
    mean_r = returns.mean()
    var_r = returns.var()
    if var_r < 1e-15:
        return 1.0

    correction_sum = 0.0
    for k in range(1, max_lag + 1):
        # Autocorrelation at lag k
        rho_k = np.corrcoef(returns[k:], returns[:-k])[0, 1]
        if not np.isfinite(rho_k):
            continue
        # Bartlett kernel weight
        weight = 1.0 - k / (max_lag + 1)
        correction_sum += weight * rho_k

    # eta^2 = 1 + 2 * correction_sum
    eta_sq = 1.0 + 2.0 * correction_sum
    if eta_sq <= 0:
        return 1.0

    return float(np.sqrt(eta_sq))
