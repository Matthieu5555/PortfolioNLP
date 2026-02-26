"""
Portfolio performance metrics.

Standard risk-adjusted performance measures for comparing strategies.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["compute_portfolio_metrics", "fama_french_regression"]

TRADING_DAYS_PER_YEAR = 252


def compute_portfolio_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute standard portfolio performance metrics.

    Args:
        returns: daily portfolio returns (simple returns).
        benchmark_returns: daily benchmark returns (e.g., SPY).
        risk_free_rate: annualized risk-free rate.

    Returns:
        dict with performance metrics.
    """
    if len(returns) == 0:
        return {}

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf

    ann_excess_return = float(excess.mean() * TRADING_DAYS_PER_YEAR)
    ann_return = float(returns.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = float(excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    sharpe = ann_excess_return / ann_vol if ann_vol > 0 else 0.0

    # Sortino: downside deviation includes zero contributions from positive returns
    downside_sq = np.minimum(excess.values, 0.0) ** 2
    downside_dev = float(np.sqrt(downside_sq.mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino = ann_excess_return / downside_dev if downside_dev > 0 else 0.0

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

    metrics = {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "n_observations": len(returns),
    }

    if benchmark_returns is not None:
        # Align dates
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 10:
            r = returns.loc[common_idx]
            b = benchmark_returns.loc[common_idx]

            # Information ratio
            active = r - b
            tracking_error = float(active.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            info_ratio = float(active.mean() * TRADING_DAYS_PER_YEAR) / tracking_error if tracking_error > 0 else 0.0

            # CAPM alpha/beta with HAC standard errors
            import statsmodels.api as sm

            X_bench = sm.add_constant(b.values)
            capm_ols = sm.OLS(r.values, X_bench).fit(
                cov_type="HAC", cov_kwds={"maxlags": 10},
            )
            alpha_daily = float(capm_ols.params[0])
            beta = float(capm_ols.params[1])
            alpha = alpha_daily * TRADING_DAYS_PER_YEAR

            metrics.update({
                "information_ratio": info_ratio,
                "tracking_error": tracking_error,
                "capm_alpha": alpha,
                "capm_beta": beta,
                "capm_alpha_tstat": float(capm_ols.tvalues[0]),
                "capm_alpha_pvalue": float(capm_ols.pvalues[0]),
            })

    return metrics


def fama_french_regression(
    returns: pd.Series,
    factors: pd.DataFrame | None = None,
    model: str = "ff5_mom",
    max_lag: int = 10,
) -> dict[str, float | dict]:
    """Regress portfolio returns on Fama-French factors.

    Uses OLS with Newey-West (HAC) standard errors to account for
    serial correlation and heteroskedasticity in return residuals.

    Args:
        returns: daily portfolio returns (not excess — RF is subtracted here).
        factors: DataFrame with columns [Mkt-RF, SMB, HML, RMW, CMA, Mom, RF].
            If None, downloads from Kenneth French's data library.
        model: which factor model to use:
            'ff3' — Mkt-RF, SMB, HML
            'ff5' — Mkt-RF, SMB, HML, RMW, CMA
            'ff5_mom' — Mkt-RF, SMB, HML, RMW, CMA, Mom (Carhart 6-factor)
        max_lag: Newey-West bandwidth for HAC standard errors.

    Returns:
        dict with:
            alpha_annual: annualized alpha (intercept * 252)
            alpha_tstat: t-statistic for alpha
            r_squared: R² of the regression
            r_squared_adj: adjusted R²
            betas: dict of factor_name -> beta
            tstats: dict of factor_name -> t-stat
            n_obs: number of observations
    """
    import statsmodels.api as sm

    if factors is None:
        from pnlp.data.fama_french import load_ff_factors
        start_str = str(returns.index.min().date())
        end_str = str(returns.index.max().date())
        factors = load_ff_factors(start=start_str, end=end_str)

    # Select factor columns based on model
    if model == "ff3":
        factor_cols = ["Mkt-RF", "SMB", "HML"]
    elif model == "ff5":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    elif model == "ff5_mom":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    else:
        raise ValueError(f"Unknown model: {model}")

    missing = [c for c in factor_cols if c not in factors.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}. Available: {list(factors.columns)}")

    # Align dates
    common_idx = returns.index.intersection(factors.index)
    if len(common_idx) < 60:
        logger.warning("Only %d common dates for FF regression", len(common_idx))
        return {"error": "insufficient_data", "n_obs": len(common_idx)}

    ret = returns.loc[common_idx]
    ff = factors.loc[common_idx]

    # Compute excess returns (subtract risk-free rate)
    excess_ret = ret - ff["RF"]

    # Build regressor matrix
    X = ff[factor_cols]
    X = sm.add_constant(X)
    y = excess_ret

    # OLS with Newey-West standard errors
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})

    alpha_daily = ols.params["const"]
    alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR

    result = {
        "alpha_annual": float(alpha_annual),
        "alpha_daily": float(alpha_daily),
        "alpha_tstat": float(ols.tvalues["const"]),
        "alpha_pvalue": float(ols.pvalues["const"]),
        "r_squared": float(ols.rsquared),
        "r_squared_adj": float(ols.rsquared_adj),
        "n_obs": int(ols.nobs),
        "betas": {col: float(ols.params[col]) for col in factor_cols},
        "tstats": {col: float(ols.tvalues[col]) for col in factor_cols},
        "pvalues": {col: float(ols.pvalues[col]) for col in factor_cols},
    }

    logger.info(
        "FF regression (%s): alpha=%.4f%% (t=%.2f), R²=%.3f, n=%d",
        model,
        alpha_annual * 100,
        result["alpha_tstat"],
        result["r_squared"],
        result["n_obs"],
    )

    return result
