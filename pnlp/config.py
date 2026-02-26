"""
Centralized configuration for the PortfolioNLP system.

All magic numbers, model paths, and tunable parameters live here.
Each constant documents what it controls and the effect of changing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "UniverseConfig",
    "EmbeddingConfig",
    "MuConfig",
    "SigmaConfig",
    "CovarianceConfig",
    "PortfolioConfig",
    "BacktestConfig",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# Structured data subdirectories (migration pending — currently files live
# directly in DATA_DIR for backward compatibility):
#   RAW_DIR:       immutable downloads (filings, transcripts, prices, SIC codes)
#   PROCESSED_DIR: derived artifacts (embeddings, store metadata)
#   RESULTS_DIR:   experiment outputs (backtests, ablations, figures)
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UniverseConfig:
    """Which firms to include and how to identify them.

    top_n: number of most-liquid US equities to include.
    liquidity_lookback_days: trailing window for average daily dollar volume.
    min_price: minimum stock price to avoid penny stocks.
    start_date / end_date: overall date range for the study.
    """

    top_n: int = 2000
    liquidity_lookback_days: int = 252
    min_price: float = 5.0
    start_date: date = date(2000, 1, 1)
    end_date: date = date(2025, 12, 31)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingConfig:
    """Text embedding model and aggregation parameters.

    model_name: HuggingFace sentence-transformers model id.
        'BAAI/bge-base-en-v1.5' -> 768-dim, strong retrieval quality.
    text_dim: output embedding dimension (must match model).
    max_chunk_tokens: maximum tokens per chunk when embedding long documents.
        Documents exceeding this are split at sentence boundaries, embedded
        independently, and mean-pooled.
    aggregation: how to combine multiple documents into a firm representation.
        'mean' — simple mean pooling (baseline, Hoberg-Phillips spirit).
        'time_weighted' — exponential decay, recent filings weighted more.
    time_decay_halflife_days: for time_weighted aggregation, half-life in days.
        365 means a filing from one year ago gets 50% weight vs today's.
    """

    model_name: str = "BAAI/bge-base-en-v1.5"
    text_dim: int = 768
    max_chunk_tokens: int = 512
    aggregation: str = "mean"
    time_decay_halflife_days: int = 365


# ---------------------------------------------------------------------------
# Portfolio Primitives
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MuConfig:
    """Expected return proxy construction from embedding geometry.

    method:
        'returns_direction' — supervised: ridge regression to find the
            direction in R^768 maximally correlated with forward returns.
        'cluster_distance' — semi-supervised: k-means clusters labeled by
            average realized return; score by distance to high vs low clusters.
        'temporal_drift' — unsupervised: direction and magnitude of embedding
            movement between consecutive filing periods.
    horizon_days: forward return horizon for supervised fitting (~63 = quarterly).
    regularization: L2 penalty for returns_direction ridge regression.
        Critical: 768 dims, ~2000 firms. Without strong penalty = overfitting.
    n_clusters: number of semantic clusters for cluster_distance method.
    """

    method: str = "returns_direction"
    horizon_days: int = 63
    regularization: float = 100.0
    n_clusters: int = 10


@dataclass(frozen=True)
class SigmaConfig:
    """Volatility proxy construction from embedding dispersion.

    method:
        'dispersion' — std dev of a firm's document embeddings around its
            centroid. Inconsistent language across filings = higher risk.
        'centroid_distance' — distance from sector centroid. Semantically
            unusual firms are harder to price = riskier.
        'entropy' — differential entropy of the local embedding distribution.
    window_days: lookback window for computing dispersion.
    """

    method: str = "dispersion"
    window_days: int = 365


@dataclass(frozen=True)
class CovarianceConfig:
    """Covariance matrix proxy construction from embedding similarity.

    method:
        'cosine' — cosine similarity of firm embeddings as correlation proxy.
            PSD by construction (Gram matrix). Combined with sigma estimates.
        'energy_distance' — Gawronsky & Huang (2024). Treats each firm's
            document set as a distribution; computes pairwise energy distance.
        'shrinkage' — EMNLP 2023: shrink sample covariance toward semantic
            similarity target. Hybrid: uses both text and prices.
    shrinkage_intensity: for 'shrinkage' method. 'auto' = Ledoit-Wolf oracle.
    min_eigenvalue: floor for eigenvalues to ensure positive definiteness.
    """

    method: str = "shrinkage"
    shrinkage_intensity: float | str = "auto"
    min_eigenvalue: float = 1e-4  # relative floor applied as max_eigenvalue * min_eigenvalue


# ---------------------------------------------------------------------------
# Portfolio Optimization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio optimization parameters.

    objective:
        'min_variance' — minimum variance portfolio (avoids needing mu).
        'max_sharpe' — maximize Sharpe ratio (tangency portfolio).
        'risk_parity' — equal risk contribution.
        'mean_variance' — standard Markowitz with risk aversion parameter.
    long_only: whether to impose long-only (w >= 0) constraint.
    max_weight: maximum position size per asset. 0.05 = 5%.
    turnover_penalty: proportional transaction cost for rebalancing.
    rebalance_frequency: 'monthly', 'quarterly', 'annually'.
    risk_aversion: lambda for mean_variance objective.
    """

    objective: str = "min_variance"
    long_only: bool = True
    max_weight: float = 0.05
    turnover_penalty: float = 0.001
    rebalance_frequency: str = "quarterly"
    risk_aversion: float = 1.0


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward backtest parameters.

    start_date / end_date: backtest period (must be within UniverseConfig range).
    initial_capital: starting portfolio value in USD.
    transaction_cost_bps: per-trade cost in basis points (10 = 0.10%).
    warmup_years: years of data required before first rebalancing.
    """

    start_date: date = date(2005, 1, 1)
    end_date: date = date(2025, 12, 31)
    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 10.0
    warmup_years: int = 3
