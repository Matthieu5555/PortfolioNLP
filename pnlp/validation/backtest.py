"""
Walk-forward portfolio backtest engine.

Simulates a realistic portfolio strategy with quarterly rebalancing,
transaction costs, and proper look-ahead bias prevention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from pnlp.config import BacktestConfig
from pnlp.portfolio.optimizer import PortfolioWeights
from pnlp.validation.metrics import compute_portfolio_metrics

logger = logging.getLogger(__name__)

__all__ = ["BacktestResult", "BacktestEngine"]


@dataclass
class BacktestResult:
    """Result of a portfolio backtest."""

    strategy_name: str
    config: BacktestConfig
    portfolio_returns: pd.Series  # daily portfolio returns
    weights_history: dict[date, PortfolioWeights]  # rebalance date -> weights
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """Walk-forward portfolio backtest.

    Given a series of portfolio weights at rebalancing dates and daily
    returns for the universe, computes the realized portfolio return
    series including transaction costs.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        weights_history: dict[date, PortfolioWeights],
        daily_returns: pd.DataFrame,
        strategy_name: str = "strategy",
        benchmark_returns: pd.Series | None = None,
        tc_rates_history: dict[date, pd.Series] | None = None,
    ) -> BacktestResult:
        """Run backtest from a sequence of portfolio weights.

        Args:
            weights_history: rebalance_date -> PortfolioWeights.
            daily_returns: DataFrame with tickers as columns, dates as index.
            strategy_name: name for this strategy.
            benchmark_returns: optional benchmark daily returns.
            tc_rates_history: optional per-ticker TC rates at each rebalance
                date. If None, uses flat rate from config.

        Returns:
            BacktestResult with realized returns and metrics.
        """
        rebal_dates = sorted(weights_history.keys())
        flat_tc_rate = self.config.transaction_cost_bps / 10000.0

        # Convert log returns to simple returns (prices.py produces log returns)
        daily_returns = np.exp(daily_returns) - 1

        portfolio_returns: list[tuple[date, float]] = []
        prev_weights: pd.Series | None = None

        for i, rebal_date in enumerate(rebal_dates):
            # Determine period: from this rebal to next (or end)
            period_start = rebal_date
            if i + 1 < len(rebal_dates):
                period_end = rebal_dates[i + 1]
            else:
                period_end = self.config.end_date

            target_weights = weights_history[rebal_date].weights

            # Transaction costs at rebalancing
            per_ticker_rates = tc_rates_history.get(rebal_date) if tc_rates_history else None
            if prev_weights is not None:
                all_tickers = set(target_weights.index) | set(prev_weights.index)
                if per_ticker_rates is not None:
                    tc = sum(
                        abs(target_weights.get(t, 0) - prev_weights.get(t, 0))
                        * per_ticker_rates.get(t, flat_tc_rate)
                        for t in all_tickers
                    )
                else:
                    turnover = sum(
                        abs(target_weights.get(t, 0) - prev_weights.get(t, 0))
                        for t in all_tickers
                    )
                    tc = turnover * flat_tc_rate
            else:
                tc = flat_tc_rate  # Initial investment cost

            # Get daily returns for the holding period
            period_mask = (daily_returns.index >= str(period_start)) & (daily_returns.index < str(period_end))
            period_rets = daily_returns.loc[period_mask]

            if len(period_rets) == 0:
                continue

            # Apply transaction cost on first day
            first_day = True
            current_weights = target_weights.copy()

            for ret_date, row in period_rets.iterrows():
                # Portfolio return = weighted sum of individual returns
                port_ret = 0.0
                for ticker, weight in current_weights.items():
                    if ticker in row.index and np.isfinite(row[ticker]):
                        port_ret += weight * row[ticker]

                if first_day:
                    port_ret -= tc
                    first_day = False

                portfolio_returns.append((ret_date.date() if hasattr(ret_date, 'date') else ret_date, port_ret))

                # Drift weights (buy-and-hold between rebalancing)
                for ticker in current_weights.index:
                    if ticker in row.index and np.isfinite(row[ticker]):
                        current_weights[ticker] *= (1 + row[ticker])
                total = current_weights.sum()
                if total > 0:
                    current_weights = current_weights / total

            # Use DRIFTED weights (not target) for next rebalance turnover calc
            prev_weights = current_weights.copy()

        if not portfolio_returns:
            return BacktestResult(
                strategy_name=strategy_name,
                config=self.config,
                portfolio_returns=pd.Series(dtype=float),
                weights_history=weights_history,
            )

        dates, rets = zip(*portfolio_returns)
        port_ret_series = pd.Series(rets, index=pd.DatetimeIndex(dates), name=strategy_name)

        # Compute metrics
        metrics = compute_portfolio_metrics(
            port_ret_series,
            benchmark_returns=benchmark_returns,
        )

        # Turnover statistics (using drifted weights for accuracy)
        # Already captured during the main loop via prev_weights (drifted)
        # But we can also compute turnover from target_weights vs prev drifted
        # The main loop's tc calculation already uses drifted weights for cost
        turnovers = []
        prev_w = None
        for d in rebal_dates:
            w = weights_history[d].weights
            if prev_w is not None:
                all_t = set(w.index) | set(prev_w.index)
                to = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t)
                turnovers.append(to)
            prev_w = w
        if turnovers:
            metrics["avg_turnover"] = float(np.mean(turnovers))

        return BacktestResult(
            strategy_name=strategy_name,
            config=self.config,
            portfolio_returns=port_ret_series,
            weights_history=weights_history,
            metrics=metrics,
        )

    @staticmethod
    def compare(results: list[BacktestResult]) -> pd.DataFrame:
        """Compare multiple backtest results in a summary table."""
        rows = []
        for r in results:
            row = {"strategy": r.strategy_name}
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows).set_index("strategy")
