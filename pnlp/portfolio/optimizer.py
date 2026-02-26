"""
Portfolio optimization using text-derived inputs.

Takes mu, sigma, and the covariance matrix (all potentially from
embedding geometry) and produces optimal portfolio weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pnlp.config import PortfolioConfig

logger = logging.getLogger(__name__)

__all__ = ["PortfolioWeights", "PortfolioOptimizer"]


@dataclass(frozen=True)
class PortfolioWeights:
    """Result of portfolio optimization."""

    weights: pd.Series  # ticker -> weight
    objective_value: float
    metadata: dict[str, Any]


class PortfolioOptimizer:
    """Mean-variance portfolio optimization.

    Takes text-derived (or traditional) mu and covariance matrix,
    produces portfolio weights via constrained optimization.

    Objectives:
        'min_variance' — minimize w'Sigma w (no mu needed).
        'max_sharpe' — maximize (w'mu) / sqrt(w'Sigma w).
        'mean_variance' — maximize w'mu - (lambda/2) * w'Sigma w.
        'risk_parity' — equal risk contribution.
    """

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()

    def _optimize_qp(
        self,
        Sigma: np.ndarray,
        mu_vec: np.ndarray,
        tickers: list[str],
        sector_constraints: list[tuple[list[int], float]] | None = None,
    ) -> PortfolioWeights:
        """Solve min_variance or mean_variance as a QP via cvxpy.

        Args:
            sector_constraints: optional list of (ticker_indices, max_weight) tuples.
                Each entry constrains sum(w[indices]) <= max_weight.
        """
        n = len(tickers)
        w = cp.Variable(n)
        objective = self.config.objective

        if objective == "min_variance":
            obj = cp.Minimize(cp.quad_form(w, Sigma))
        else:  # mean_variance
            lam = self.config.risk_aversion
            obj = cp.Maximize(mu_vec @ w - (lam / 2) * cp.quad_form(w, Sigma))

        constraints = [cp.sum(w) == 1]
        if self.config.long_only:
            constraints += [w >= 0, w <= self.config.max_weight]
        else:
            constraints += [w >= -self.config.max_weight, w <= self.config.max_weight]

        # Sector-level weight constraints
        if sector_constraints:
            for indices, max_sector_wt in sector_constraints:
                constraints.append(cp.sum(w[indices]) <= max_sector_wt)

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            logger.warning("QP solver status: %s, falling back to SLSQP", prob.status)
            return self._optimize_slsqp(Sigma, mu_vec, tickers)

        weights = pd.Series(w.value, index=tickers)
        weights[weights.abs() < 1e-6] = 0.0
        weights = weights / weights.sum()
        weights = weights.clip(upper=self.config.max_weight)
        weights = weights / weights.sum()

        return PortfolioWeights(
            weights=weights,
            objective_value=float(prob.value),
            metadata={
                "success": prob.status in ("optimal", "optimal_inaccurate"),
                "message": prob.status,
                "solver": "cvxpy/OSQP",
                "objective": objective,
            },
        )

    def _optimize_slsqp(
        self,
        Sigma: np.ndarray,
        mu_vec: np.ndarray,
        tickers: list[str],
    ) -> PortfolioWeights:
        """Solve non-QP objectives (max_sharpe, risk_parity) via SLSQP."""
        n = len(tickers)
        w0 = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if self.config.long_only:
            bounds = [(0.0, self.config.max_weight) for _ in range(n)]
        else:
            bounds = [(-self.config.max_weight, self.config.max_weight) for _ in range(n)]

        objective = self.config.objective
        jac_fn = None

        if objective == "min_variance":
            def obj(w):
                return w @ Sigma @ w
            def jac_fn(w):
                return 2.0 * (Sigma @ w)

        elif objective == "max_sharpe":
            def obj(w):
                port_return = w @ mu_vec
                port_vol = np.sqrt(max(w @ Sigma @ w, 1e-12))
                return -port_return / port_vol

        elif objective == "mean_variance":
            lam = self.config.risk_aversion
            def obj(w):
                return -(w @ mu_vec - (lam / 2) * w @ Sigma @ w)
            def jac_fn(w):
                return -(mu_vec - lam * (Sigma @ w))

        elif objective == "risk_parity":
            def obj(w):
                Sw = Sigma @ w
                risk_contrib = w * Sw
                total_risk = w @ Sw
                if total_risk < 1e-12:
                    return 0.0
                rc_pct = risk_contrib / total_risk
                target = 1.0 / n
                return np.sum((rc_pct - target) ** 2)

        else:
            raise ValueError(f"Unknown objective: {objective}")

        result = minimize(
            obj,
            w0,
            method="SLSQP",
            jac=jac_fn,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning("Optimization did not converge: %s", result.message)

        weights = pd.Series(result.x, index=tickers)
        weights[weights.abs() < 1e-6] = 0.0
        weights = weights / weights.sum()
        weights = weights.clip(upper=self.config.max_weight)
        weights = weights / weights.sum()

        return PortfolioWeights(
            weights=weights,
            objective_value=float(result.fun),
            metadata={
                "success": result.success,
                "message": result.message,
                "n_iterations": result.nit,
                "solver": "SLSQP",
                "objective": objective,
            },
        )

    def optimize(
        self,
        cov: pd.DataFrame,
        mu: pd.Series | None = None,
        *,
        ticker_sectors: dict[str, str] | None = None,
        sector_limits: dict[str, float] | None = None,
        vol_target: float | None = None,
    ) -> PortfolioWeights:
        """Optimize portfolio weights.

        Args:
            cov: (n, n) covariance matrix with ticker index/columns.
            mu: (n,) expected returns with ticker index. Required for
                max_sharpe and mean_variance objectives.
            ticker_sectors: ticker -> sector name mapping (e.g. 2-digit SIC).
            sector_limits: sector name -> max total weight for that sector.
                Requires ticker_sectors.
            vol_target: target annualized portfolio volatility. Weights are
                scaled post-optimization to match (may require leverage).

        Returns:
            PortfolioWeights with optimal allocations.
        """
        tickers = list(cov.index)
        n = len(tickers)
        Sigma = cov.values

        if mu is not None:
            mu_vec = np.array([mu.get(t, 0.0) for t in tickers])
        else:
            mu_vec = np.zeros(n)

        # Build sector constraints for QP
        sector_constraints = None
        if sector_limits and ticker_sectors:
            sector_constraints = []
            # Group tickers by sector
            sector_to_idx: dict[str, list[int]] = {}
            for i, t in enumerate(tickers):
                sec = ticker_sectors.get(t)
                if sec:
                    sector_to_idx.setdefault(sec, []).append(i)
            for sector, max_wt in sector_limits.items():
                if sector in sector_to_idx:
                    sector_constraints.append((sector_to_idx[sector], max_wt))

        # Use cvxpy for QP-solvable objectives (orders of magnitude faster at large n)
        if self.config.objective in ("min_variance", "mean_variance"):
            result = self._optimize_qp(Sigma, mu_vec, tickers, sector_constraints)
        else:
            # Fallback to SLSQP for non-convex objectives
            result = self._optimize_slsqp(Sigma, mu_vec, tickers)

        # Vol targeting: scale weights to match target annualized vol
        if vol_target is not None:
            w = result.weights.values
            predicted_var = w @ Sigma @ w
            predicted_vol = np.sqrt(predicted_var * 252)
            if predicted_vol > 1e-10:
                scale = vol_target / predicted_vol
                result = PortfolioWeights(
                    weights=result.weights * scale,
                    objective_value=result.objective_value,
                    metadata={**result.metadata, "vol_scale": float(scale),
                              "predicted_vol": float(predicted_vol)},
                )

        return result
