"""
Value-at-Risk backtesting tests.

Implements Kupiec (1995) unconditional coverage test and
Christoffersen (1998) conditional coverage test for VaR
model evaluation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


__all__ = ["kupiec_test", "christoffersen_test", "parametric_var"]


def parametric_var(
    portfolio_vol: float,
    alpha: float,
    portfolio_mu: float = 0.0,
) -> float:
    """Compute one-day parametric (Gaussian) VaR.

    Args:
        portfolio_vol: daily portfolio volatility (std dev).
        alpha: significance level (e.g., 0.05 for 5% VaR).
        portfolio_mu: expected daily return (default 0).

    Returns:
        VaR as a positive number (loss threshold).
    """
    z = stats.norm.ppf(alpha)
    return -(portfolio_mu + z * portfolio_vol)


def kupiec_test(
    n_observations: int,
    n_violations: int,
    alpha: float,
) -> dict:
    """Kupiec (1995) proportion-of-failures (POF) test.

    Tests H0: the true violation rate equals the nominal level alpha.

    Args:
        n_observations: total number of out-of-sample days.
        n_violations: number of days where loss exceeded VaR.
        alpha: nominal VaR significance level.

    Returns:
        dict with violation_rate, lr_statistic, p_value, reject_5pct.
    """
    n = n_observations
    v = n_violations

    if n == 0:
        return {"violation_rate": float("nan"), "lr_statistic": float("nan"),
                "p_value": float("nan"), "reject_5pct": False}

    p_hat = v / n if n > 0 else 0.0

    # Handle edge cases for log
    if v == 0:
        lr = -2.0 * (n * np.log(1 - alpha))
    elif v == n:
        lr = -2.0 * (n * np.log(alpha))
    else:
        lr = -2.0 * (
            v * np.log(alpha) + (n - v) * np.log(1 - alpha)
            - v * np.log(p_hat) - (n - v) * np.log(1 - p_hat)
        )

    p_value = 1.0 - stats.chi2.cdf(lr, df=1)

    return {
        "violation_rate": p_hat,
        "expected_rate": alpha,
        "n_observations": n,
        "n_violations": v,
        "lr_statistic": float(lr),
        "p_value": float(p_value),
        "reject_5pct": p_value < 0.05,
    }


def christoffersen_test(
    violations: np.ndarray,
    alpha: float,
) -> dict:
    """Christoffersen (1998) conditional coverage test.

    Tests both unconditional coverage (Kupiec) AND independence
    of violations. Clustered violations indicate model failure.

    Args:
        violations: boolean array (True = violation on that day).
        alpha: nominal VaR significance level.

    Returns:
        dict with lr_cc (conditional coverage statistic), p_value,
        and the independence component.
    """
    violations = np.asarray(violations, dtype=bool)
    n = len(violations)

    if n < 2:
        return {"lr_cc": float("nan"), "p_value": float("nan"),
                "reject_5pct": False}

    v = violations.sum()

    # Kupiec (unconditional coverage) component
    kupiec = kupiec_test(n, int(v), alpha)
    lr_uc = kupiec["lr_statistic"]

    # Independence component: count transitions
    # n00: no-violation followed by no-violation
    # n01: no-violation followed by violation
    # n10: violation followed by no-violation
    # n11: violation followed by violation
    n00 = n01 = n10 = n11 = 0
    for i in range(n - 1):
        if not violations[i] and not violations[i + 1]:
            n00 += 1
        elif not violations[i] and violations[i + 1]:
            n01 += 1
        elif violations[i] and not violations[i + 1]:
            n10 += 1
        else:
            n11 += 1

    # Transition probabilities
    n0 = n00 + n01  # days starting with no-violation
    n1 = n10 + n11  # days starting with violation

    if n0 == 0 or n1 == 0 or n01 == 0 or n10 == 0:
        # Cannot compute independence test — degenerate case
        return {
            "lr_cc": lr_uc,
            "lr_uc": lr_uc,
            "lr_ind": 0.0,
            "p_value": kupiec["p_value"],
            "reject_5pct": kupiec["reject_5pct"],
            "n_violations": int(v),
            "violation_rate": v / n,
        }

    pi01 = n01 / n0  # P(violation | no violation yesterday)
    pi11 = n11 / n1  # P(violation | violation yesterday)
    pi = (n01 + n11) / (n0 + n1)  # unconditional P(violation)

    # Independence LR
    lr_ind = -2.0 * (
        n00 * np.log(1 - pi) + n01 * np.log(pi)
        + n10 * np.log(1 - pi) + n11 * np.log(pi)
        - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
        - n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )

    lr_cc = lr_uc + lr_ind
    p_value = 1.0 - stats.chi2.cdf(lr_cc, df=2)

    return {
        "lr_cc": float(lr_cc),
        "lr_uc": float(lr_uc),
        "lr_ind": float(lr_ind),
        "p_value": float(p_value),
        "reject_5pct": p_value < 0.05,
        "n_violations": int(v),
        "violation_rate": v / n,
        "pi01": float(pi01),
        "pi11": float(pi11),
    }
