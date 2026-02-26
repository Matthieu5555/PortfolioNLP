"""
Stratified transaction cost model.

Assigns per-trade cost based on stock liquidity tier and market
volatility regime. Replaces the naive flat-rate assumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["TransactionCostModel", "TCTier"]


@dataclass(frozen=True)
class TCTier:
    """A single liquidity tier for transaction costs."""

    name: str
    adv_floor: float  # minimum median ADV to qualify (USD)
    cost_bps: float   # base cost in basis points


DEFAULT_TIERS = [
    TCTier("large_cap", adv_floor=500_000_000, cost_bps=10.0),
    TCTier("mid_cap", adv_floor=50_000_000, cost_bps=20.0),
    TCTier("small_cap", adv_floor=0, cost_bps=50.0),
]


class TransactionCostModel:
    """Per-ticker transaction costs based on ADV tier and vol regime.

    Usage:
        tc_model = TransactionCostModel()
        tc_rates = tc_model.get_tc_rates(median_adv, daily_returns, rebal_date)
        # tc_rates: ticker -> cost rate as decimal fraction (0.001 = 10 bps)
    """

    def __init__(
        self,
        tiers: list[TCTier] | None = None,
        vol_lookback: int = 63,
        vol_multiplier_cap: float = 2.0,
    ) -> None:
        self.tiers = sorted(tiers or DEFAULT_TIERS, key=lambda t: t.adv_floor, reverse=True)
        self.vol_lookback = vol_lookback
        self.vol_multiplier_cap = vol_multiplier_cap

    def _assign_tier(self, adv: float) -> TCTier:
        for tier in self.tiers:
            if adv >= tier.adv_floor:
                return tier
        return self.tiers[-1]

    def _vol_regime_multiplier(
        self,
        daily_returns: pd.DataFrame,
        rebal_date_str: str,
    ) -> float:
        """Compute market-wide volatility regime multiplier.

        Uses cross-sectional median of per-stock realized vol as a VIX
        proxy. When recent vol exceeds long-term baseline, TC costs
        increase (spreads widen during stress).
        """
        returns_before = daily_returns.loc[daily_returns.index < rebal_date_str]
        if len(returns_before) < self.vol_lookback * 2:
            return 1.0

        recent_vol = returns_before.iloc[-self.vol_lookback :].std().median()
        longterm_vol = returns_before.std().median()

        if longterm_vol <= 0 or not np.isfinite(longterm_vol):
            return 1.0

        ratio = recent_vol / longterm_vol
        return float(max(1.0, min(ratio, self.vol_multiplier_cap)))

    def get_tc_rates(
        self,
        median_adv: pd.Series,
        daily_returns: pd.DataFrame,
        rebal_date: str,
    ) -> pd.Series:
        """Get per-ticker TC rates for a rebalance date.

        Args:
            median_adv: ticker -> trailing median ADV in USD.
            daily_returns: full daily returns matrix (for vol regime).
            rebal_date: rebalance date as string (YYYY-MM-DD).

        Returns:
            Series: ticker -> TC rate as decimal fraction.
        """
        vol_mult = self._vol_regime_multiplier(daily_returns, rebal_date)

        rates = {}
        for ticker, adv in median_adv.items():
            if not np.isfinite(adv):
                tier = self.tiers[-1]
            else:
                tier = self._assign_tier(adv)
            rates[ticker] = (tier.cost_bps / 10_000.0) * vol_mult

        return pd.Series(rates)
