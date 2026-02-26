"""
Covariance matrix proxies from embedding similarity.

Replaces the traditional sample covariance matrix with estimates derived
from the geometric structure of firm embeddings in text space.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from pnlp.config import CovarianceConfig
from pnlp.embeddings.firm_aggregator import FirmEmbedding

logger = logging.getLogger(__name__)

__all__ = [
    "CovarianceEstimator",
    "CosineSimilarityCovariance",
    "EnergyDistanceCovariance",
    "MultiTargetShrinkageCovariance",
    "PCAFactorCovariance",
    "SemanticShrinkageCovariance",
    "cross_validate_alpha",
]


def _ensure_psd(matrix: np.ndarray, min_eigenvalue: float = 1e-4) -> np.ndarray:
    """Ensure a matrix is positive semi-definite by relative eigenvalue flooring.

    min_eigenvalue is treated as relative to the largest eigenvalue:
        floor = max_eigenvalue * min_eigenvalue
    This prevents degenerate condition numbers regardless of matrix scale.
    Uses GPU when available (~800x faster at p>=2000).
    """
    from pnlp.primitives.gpu_accel import ensure_psd
    return ensure_psd(matrix, min_eigenvalue)


class CovarianceEstimator(ABC):
    """Abstract base for covariance estimation from embeddings."""

    @abstractmethod
    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        sigma_estimates: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Estimate covariance matrix from firm embeddings.

        Args:
            firm_embeddings: ticker -> FirmEmbedding.
            sigma_estimates: optional ticker -> volatility estimates.
                If None, the estimator uses unit volatility (correlation-only).

        Returns:
            DataFrame with shape (n_firms, n_firms), index and columns = tickers.
        """
        ...


class CosineSimilarityCovariance(CovarianceEstimator):
    """Covariance from cosine similarity of firm embeddings.

    C_ij = sigma_i * sigma_j * cos(e_i, e_j)

    Cosine similarity serves as a correlation proxy. The resulting
    similarity matrix is PSD by construction (it's a Gram matrix:
    S = E @ E.T where E is the L2-normalized embedding matrix).

    Combined with per-firm sigma estimates, produces a full covariance
    matrix. If no sigma estimates are provided, returns the correlation
    proxy directly.
    """

    def __init__(self, config: CovarianceConfig | None = None) -> None:
        self.config = config or CovarianceConfig()

    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        sigma_estimates: pd.Series | None = None,
    ) -> pd.DataFrame:
        tickers = sorted(firm_embeddings.keys())
        n = len(tickers)

        # Build embedding matrix (n_firms, embed_dim)
        E = np.array([firm_embeddings[t].embedding for t in tickers])

        # Cosine similarity = dot product of L2-normalized vectors
        corr_proxy = E @ E.T  # (n, n), values in [-1, 1]

        # Clip to valid correlation range
        np.clip(corr_proxy, -1.0, 1.0, out=corr_proxy)

        if sigma_estimates is not None:
            # Scale correlation proxy to covariance
            sigmas = np.array([sigma_estimates.get(t, 1.0) for t in tickers])
            sigma_outer = np.outer(sigmas, sigmas)
            cov = corr_proxy * sigma_outer
        else:
            cov = corr_proxy

        # Ensure PSD
        cov = _ensure_psd(cov, self.config.min_eigenvalue)

        return pd.DataFrame(cov, index=tickers, columns=tickers)


class EnergyDistanceCovariance(CovarianceEstimator):
    """Covariance from Energy Distance (Gawronsky & Huang, 2024).

    Treats each firm's set of document embeddings as a distribution.
    Energy Distance between distributions X and Y:
        D(X,Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]

    Converted to similarity via: sim = exp(-D / scale), then to
    covariance by multiplying with sigma estimates.

    Requires the raw per-document embeddings, not just the aggregated
    firm embedding. Pass these via the doc_embeddings parameter.
    """

    def __init__(
        self,
        config: CovarianceConfig | None = None,
        doc_embeddings: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.config = config or CovarianceConfig()
        self.doc_embeddings = doc_embeddings or {}

    def _energy_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute energy distance between two sets of vectors."""
        # E[||X - Y||]
        cross = cdist(X, Y, metric="euclidean").mean()
        # E[||X - X'||]
        if len(X) > 1:
            within_x = cdist(X, X, metric="euclidean").mean()
        else:
            within_x = 0.0
        # E[||Y - Y'||]
        if len(Y) > 1:
            within_y = cdist(Y, Y, metric="euclidean").mean()
        else:
            within_y = 0.0

        return 2 * cross - within_x - within_y

    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        sigma_estimates: pd.Series | None = None,
    ) -> pd.DataFrame:
        tickers = sorted(firm_embeddings.keys())
        n = len(tickers)

        if not self.doc_embeddings:
            logger.warning("No doc_embeddings provided; falling back to cosine similarity")
            return CosineSimilarityCovariance(self.config).estimate(firm_embeddings, sigma_estimates)

        # Compute pairwise energy distances
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            Xi = self.doc_embeddings.get(tickers[i])
            if Xi is None or len(Xi) == 0:
                continue
            for j in range(i + 1, n):
                Xj = self.doc_embeddings.get(tickers[j])
                if Xj is None or len(Xj) == 0:
                    continue
                d = self._energy_distance(Xi, Xj)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Convert distance to similarity
        scale = np.median(dist_matrix[dist_matrix > 0]) if (dist_matrix > 0).any() else 1.0
        sim_matrix = np.exp(-dist_matrix / max(scale, 1e-8))
        np.fill_diagonal(sim_matrix, 1.0)

        if sigma_estimates is not None:
            sigmas = np.array([sigma_estimates.get(t, 1.0) for t in tickers])
            cov = sim_matrix * np.outer(sigmas, sigmas)
        else:
            cov = sim_matrix

        cov = _ensure_psd(cov, self.config.min_eigenvalue)
        return pd.DataFrame(cov, index=tickers, columns=tickers)


def cross_validate_alpha(
    returns: pd.DataFrame,
    semantic_corr: np.ndarray,
    alpha_grid: list[float] | None = None,
    train_frac: float = 0.7,
    min_eigenvalue: float = 1e-4,
) -> float:
    """Cross-validate shrinkage intensity using realized portfolio variance.

    Splits the return window into train/validation. For each candidate alpha,
    builds shrunk covariance on train, computes min-variance weights, evaluates
    realized variance on validation. Selects alpha minimizing validation variance.

    This addresses the LW oracle miscalibration: LW minimizes Frobenius loss
    to the population covariance, but we care about portfolio loss. The two
    objectives diverge when the shrinkage target is informative (text) rather
    than agnostic (identity/constant-correlation).

    Args:
        returns: (n_obs, p) return matrix, already aligned to common tickers.
        semantic_corr: (p, p) correlation proxy from cosine similarity.
        alpha_grid: candidate alpha values. Default: fine grid from 0.01 to 1.0.
        train_frac: fraction of observations for training.
        min_eigenvalue: relative eigenvalue floor for PSD enforcement.

    Returns:
        Optimal alpha (float in alpha_grid).
    """
    if alpha_grid is None:
        alpha_grid = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    n_obs, p = returns.shape
    split = int(n_obs * train_frac)
    if split < max(10, p // 2) or n_obs - split < 10:
        logger.warning("CV-alpha: insufficient data (n=%d, p=%d), defaulting to alpha=0.5", n_obs, p)
        return 0.5

    train_rets = returns.iloc[:split].values
    val_rets = returns.iloc[split:].values

    # Sample cov on train
    S_train = np.cov(train_rets, rowvar=False)
    train_stds = np.sqrt(np.diag(S_train))
    T_train = semantic_corr * np.outer(train_stds, train_stds)

    best_alpha = 0.5
    best_var = float("inf")

    for alpha in alpha_grid:
        Sigma = (1 - alpha) * S_train + alpha * T_train
        Sigma = _ensure_psd(Sigma, min_eigenvalue)

        # Min-variance weights (closed-form: w ∝ Σ⁻¹ 1)
        try:
            Sigma_inv = np.linalg.solve(Sigma, np.ones(p))
            w = Sigma_inv / Sigma_inv.sum()
            # Enforce long-only (simple projection)
            w = np.maximum(w, 0.0)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                continue
        except np.linalg.LinAlgError:
            continue

        # Realized portfolio variance on validation set
        port_rets = val_rets @ w
        realized_var = np.var(port_rets)

        if realized_var < best_var:
            best_var = realized_var
            best_alpha = alpha

    return best_alpha


class SemanticShrinkageCovariance(CovarianceEstimator):
    """Shrink sample covariance toward semantic similarity target.

    Following EMNLP 2023 "Semantic Similarity Covariance Matrix Shrinkage":
        Sigma_hat = (1 - alpha) * S_sample + alpha * T_semantic

    This is a HYBRID method: uses both price history (sample covariance)
    and text (semantic target). The text provides structural prior about
    which firms should co-move; the price data provides sample statistics.

    Requires historical_returns as additional input.

    Shrinkage intensity modes:
        - "auto": Ledoit-Wolf oracle (Frobenius-loss optimal).
        - "cv": Cross-validated on realized portfolio variance.
        - float: Fixed intensity.
    """

    def __init__(
        self,
        config: CovarianceConfig | None = None,
        historical_returns: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or CovarianceConfig()
        self.historical_returns = historical_returns
        self.last_alpha_: float | None = None

    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        sigma_estimates: pd.Series | None = None,
    ) -> pd.DataFrame:
        tickers = sorted(firm_embeddings.keys())

        # Build semantic target (cosine similarity)
        E = np.array([firm_embeddings[t].embedding for t in tickers])
        semantic_corr = np.clip(E @ E.T, -1.0, 1.0)

        if self.historical_returns is None:
            logger.warning("No historical returns; using semantic-only covariance")
            cov = semantic_corr
            if sigma_estimates is not None:
                sigmas = np.array([sigma_estimates.get(t, 1.0) for t in tickers])
                cov = cov * np.outer(sigmas, sigmas)
            cov = _ensure_psd(cov, self.config.min_eigenvalue)
            return pd.DataFrame(cov, index=tickers, columns=tickers)

        # Compute sample covariance from returns
        common_tickers = [t for t in tickers if t in self.historical_returns.columns]
        if len(common_tickers) < 2:
            logger.warning("Not enough return data; using semantic-only")
            return CosineSimilarityCovariance(self.config).estimate(firm_embeddings, sigma_estimates)

        # Fill sparse NaNs with zero. The universe filter enforces >=80% return
        # completeness, so remaining gaps are rare. Zero-fill is conservative
        # (slightly biases covariance downward) but preserves matrix dimensions.
        returns_subset = self.historical_returns[common_tickers].fillna(0.0)
        S_sample = returns_subset.cov().values

        # Build semantic target at the same scale as S_sample
        idx = [tickers.index(t) for t in common_tickers]
        semantic_sub = semantic_corr[np.ix_(idx, idx)]
        sample_stds = np.sqrt(np.diag(S_sample))
        T_semantic = semantic_sub * np.outer(sample_stds, sample_stds)

        # Shrinkage intensity
        if self.config.shrinkage_intensity == "auto":
            from pnlp.primitives.gpu_accel import ledoit_wolf_shrinkage

            n_obs = len(returns_subset)
            X_centered = returns_subset.values - returns_subset.values.mean(axis=0)

            alpha = ledoit_wolf_shrinkage(X_centered, assume_centered=True)
            logger.info("LW-calibrated alpha=%.4f (n=%d, p=%d)", alpha, n_obs, len(common_tickers))
        elif self.config.shrinkage_intensity == "cv":
            alpha = cross_validate_alpha(
                returns_subset, semantic_sub,
                min_eigenvalue=self.config.min_eigenvalue,
            )
            logger.info("CV-calibrated alpha=%.4f (n=%d, p=%d)", alpha, len(returns_subset), len(common_tickers))
        else:
            alpha = float(self.config.shrinkage_intensity)

        self.last_alpha_ = alpha

        # Shrink
        Sigma_hat = (1 - alpha) * S_sample + alpha * T_semantic
        Sigma_hat = _ensure_psd(Sigma_hat, self.config.min_eigenvalue)

        return pd.DataFrame(Sigma_hat, index=common_tickers, columns=common_tickers)


class PCAFactorCovariance(CovarianceEstimator):
    """Factor-model covariance from PCA on the embedding space.

    Builds a rank-k approximation of the cosine similarity matrix by
    retaining the top-k singular components of the firm embedding matrix.
    The resulting low-rank correlation proxy serves as a shrinkage target
    that captures the dominant "text factors" (industry clusters, business
    model similarity) while discarding noise in the small eigenvalues.

    At n_factors = rank(E) this is mathematically identical to using the
    full cosine matrix (SemanticShrinkageCovariance). Reducing n_factors
    tests how much of the spectral structure is needed for effective
    regularization.

    When historical_returns are provided, calibrates shrinkage intensity
    using the same Ledoit-Wolf oracle formula as SemanticShrinkageCovariance.
    """

    def __init__(
        self,
        config: CovarianceConfig | None = None,
        n_factors: int = 10,
        historical_returns: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or CovarianceConfig()
        self.n_factors = n_factors
        self.historical_returns = historical_returns
        self.last_alpha_: float | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        sigma_estimates: pd.Series | None = None,
    ) -> pd.DataFrame:
        tickers = sorted(firm_embeddings.keys())
        n = len(tickers)

        # Build embedding matrix (n_firms, embed_dim)
        E = np.array([firm_embeddings[t].embedding for t in tickers])

        # SVD (uncentered — preserves cosine structure of L2-normalized embeddings)
        U, s, _ = np.linalg.svd(E, full_matrices=False)
        k = min(self.n_factors, len(s))

        # Store explained variance ratio
        s_sq = s ** 2
        self.explained_variance_ratio_ = s_sq[:k] / s_sq.sum()

        # Rank-k approximation of Gram matrix: B @ B.T where B = U_k * s_k
        B = U[:, :k] * s[:k]
        corr_k = B @ B.T

        # Normalize diagonal to 1 (make it a correlation proxy)
        diag = np.sqrt(np.diag(corr_k))
        diag = np.maximum(diag, 1e-10)
        corr_k = corr_k / np.outer(diag, diag)
        np.clip(corr_k, -1.0, 1.0, out=corr_k)

        logger.info(
            "PCA factor model: k=%d, explained_var=%.3f, n_firms=%d",
            k, float(self.explained_variance_ratio_.sum()), n,
        )

        # Without returns: use as pure text prior
        if self.historical_returns is None:
            if sigma_estimates is not None:
                sigmas = np.array([sigma_estimates.get(t, 1.0) for t in tickers])
                cov = corr_k * np.outer(sigmas, sigmas)
            else:
                cov = corr_k
            cov = _ensure_psd(cov, self.config.min_eigenvalue)
            return pd.DataFrame(cov, index=tickers, columns=tickers)

        # With returns: shrink sample covariance toward PCA target
        common_tickers = [t for t in tickers if t in self.historical_returns.columns]
        if len(common_tickers) < 2:
            logger.warning("Not enough return data; using PCA-only covariance")
            cov = _ensure_psd(corr_k, self.config.min_eigenvalue)
            return pd.DataFrame(cov, index=tickers, columns=tickers)

        # Fill sparse NaNs with zero. The universe filter enforces >=80% return
        # completeness, so remaining gaps are rare. Zero-fill is conservative
        # (slightly biases covariance downward) but preserves matrix dimensions.
        returns_subset = self.historical_returns[common_tickers].fillna(0.0)
        S_sample = returns_subset.cov().values

        # Build PCA target at the same scale as S_sample
        idx = [tickers.index(t) for t in common_tickers]
        pca_sub = corr_k[np.ix_(idx, idx)]
        sample_stds = np.sqrt(np.diag(S_sample))
        T_pca = pca_sub * np.outer(sample_stds, sample_stds)

        # Shrinkage intensity (same LW oracle formula)
        if self.config.shrinkage_intensity == "auto":
            from pnlp.primitives.gpu_accel import ledoit_wolf_shrinkage

            X_centered = returns_subset.values - returns_subset.values.mean(axis=0)
            alpha = ledoit_wolf_shrinkage(X_centered, assume_centered=True)
            logger.info("PCA-LW alpha=%.4f (n=%d, p=%d, k=%d)", alpha, len(returns_subset), len(common_tickers), k)
        else:
            alpha = float(self.config.shrinkage_intensity)

        self.last_alpha_ = alpha

        Sigma_hat = (1 - alpha) * S_sample + alpha * T_pca
        Sigma_hat = _ensure_psd(Sigma_hat, self.config.min_eigenvalue)

        return pd.DataFrame(Sigma_hat, index=common_tickers, columns=common_tickers)


class MultiTargetShrinkageCovariance(CovarianceEstimator):
    """Multi-target shrinkage: Σ = c₀·S + c₁·T_text + c₂·T_SIC + c₃·I.

    Instead of shrinking toward a single target, combines the sample
    covariance with multiple structured targets.

    Calibration modes:
        'frobenius': Lancewicki & Aladjem (2014) closed-form weights
            minimizing Frobenius loss. NOTE: this almost always gives
            c₁=c₂=0 for text/SIC targets because the oracle optimizes
            matrix distance, not portfolio loss.
        'cv': Cross-validate weights on realized portfolio variance.
            Splits lookback into train/validation, tests a grid of
            weight combinations, selects the one minimizing validation
            portfolio variance. This is the recommended mode.

    Targets:
        T_text: cosine similarity from firm embeddings (Gram matrix).
        T_SIC: block-diagonal from 2-digit SIC sector codes.
        I: scaled identity (diagonal of S_sample).
    """

    def __init__(
        self,
        config: CovarianceConfig | None = None,
        historical_returns: pd.DataFrame | None = None,
        sic_codes: dict[str, str] | None = None,
        within_sector_corr: float = 0.5,
        cross_sector_corr: float = 0.2,
        calibration: str = "cv",
    ) -> None:
        self.config = config or CovarianceConfig()
        self.historical_returns = historical_returns
        self.sic_codes = sic_codes or {}
        self.within_sector_corr = within_sector_corr
        self.cross_sector_corr = cross_sector_corr
        self.calibration = calibration
        self.last_weights_: dict[str, float] | None = None

    def _build_sic_target(self, tickers: list[str]) -> np.ndarray:
        """Build SIC block-diagonal correlation matrix."""
        n = len(tickers)
        sectors = []
        for t in tickers:
            sic = self.sic_codes.get(t)
            sectors.append(sic if sic else f"_unknown_{t}")

        corr = np.full((n, n), self.cross_sector_corr)
        for i in range(n):
            corr[i, i] = 1.0
            for j in range(i + 1, n):
                if sectors[i] == sectors[j]:
                    corr[i, j] = self.within_sector_corr
                    corr[j, i] = self.within_sector_corr
        return corr

    def _calibrate_frobenius(
        self,
        S: np.ndarray,
        targets: list[np.ndarray],
    ) -> np.ndarray:
        """Calibrate via Frobenius loss (Lancewicki & Aladjem 2014)."""
        n_targets = len(targets)

        A = np.zeros((n_targets, n_targets))
        b = np.zeros(n_targets)
        for i in range(n_targets):
            b[i] = np.sum((targets[i] - S) * S)
            for j in range(i, n_targets):
                aij = np.sum((targets[i] - S) * (targets[j] - S))
                A[i, j] = aij
                A[j, i] = aij

        try:
            A_reg = A + 1e-8 * np.eye(n_targets)
            alphas = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            alphas = np.ones(n_targets) / (n_targets + 1)

        alphas = np.maximum(alphas, 0.0)
        if alphas.sum() > 1.0:
            alphas = alphas / alphas.sum()
        return alphas

    def _calibrate_cv(
        self,
        returns: pd.DataFrame,
        S_full: np.ndarray,
        targets: list[np.ndarray],
        train_frac: float = 0.7,
    ) -> np.ndarray:
        """Calibrate via cross-validated portfolio variance.

        Tests a grid of weight combinations (simplex over targets + sample).
        Selects the combo minimizing realized min-variance portfolio variance
        on the validation period.
        """
        n_obs, p = returns.shape
        n_targets = len(targets)
        split = int(n_obs * train_frac)
        if split < max(10, p // 2) or n_obs - split < 10:
            # Not enough data for CV; use equal mixture
            return np.ones(n_targets) / (n_targets + 1)

        train_rets = returns.iloc[:split].values
        val_rets = returns.iloc[split:].values

        S_train = np.cov(train_rets, rowvar=False)
        train_stds = np.sqrt(np.diag(S_train))

        # Rescale targets to train scale
        full_stds = np.sqrt(np.diag(S_full))
        scale_ratio = np.outer(train_stds, train_stds) / np.maximum(
            np.outer(full_stds, full_stds), 1e-20,
        )
        targets_train = [T * scale_ratio for T in targets]

        # Grid: enumerate weight combos on a coarse simplex
        # For 3 targets: alpha_text, alpha_sic, alpha_identity
        # sample_weight = 1 - sum(alphas)
        grid_vals = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        best_alphas = np.zeros(n_targets)
        best_var = float("inf")

        for a_text in grid_vals:
            for a_sic in grid_vals:
                for a_id in grid_vals:
                    total = a_text + a_sic + a_id
                    if total > 1.0:
                        continue
                    alphas = np.array([a_text, a_sic, a_id][:n_targets])
                    c_sample = 1.0 - alphas.sum()

                    Sigma = c_sample * S_train
                    for a, T in zip(alphas, targets_train):
                        Sigma += a * T
                    Sigma = _ensure_psd(Sigma, self.config.min_eigenvalue)

                    # Min-variance weights (closed-form)
                    try:
                        Sigma_inv_ones = np.linalg.solve(Sigma, np.ones(p))
                        w = Sigma_inv_ones / Sigma_inv_ones.sum()
                        w = np.maximum(w, 0.0)
                        if w.sum() > 0:
                            w = w / w.sum()
                        else:
                            continue
                    except np.linalg.LinAlgError:
                        continue

                    port_rets = val_rets @ w
                    realized_var = np.var(port_rets)

                    if realized_var < best_var:
                        best_var = realized_var
                        best_alphas = alphas.copy()

        return best_alphas

    def estimate(
        self,
        firm_embeddings: dict[str, FirmEmbedding],
        sigma_estimates: pd.Series | None = None,
    ) -> pd.DataFrame:
        tickers = sorted(firm_embeddings.keys())

        if self.historical_returns is None:
            logger.warning("MultiTarget requires returns; falling back to cosine")
            return CosineSimilarityCovariance(self.config).estimate(
                firm_embeddings, sigma_estimates,
            )

        common_tickers = [t for t in tickers if t in self.historical_returns.columns]
        if len(common_tickers) < 2:
            return CosineSimilarityCovariance(self.config).estimate(
                firm_embeddings, sigma_estimates,
            )

        returns_subset = self.historical_returns[common_tickers].fillna(0.0)
        S_sample = returns_subset.cov().values
        sample_stds = np.sqrt(np.diag(S_sample))

        # Build targets at sample scale
        E = np.array([firm_embeddings[t].embedding for t in common_tickers])
        text_corr = np.clip(E @ E.T, -1.0, 1.0)
        T_text = text_corr * np.outer(sample_stds, sample_stds)

        sic_corr = self._build_sic_target(common_tickers)
        T_sic = sic_corr * np.outer(sample_stds, sample_stds)

        T_identity = np.diag(np.diag(S_sample))

        targets = [T_text, T_sic, T_identity]
        target_names = ["text", "sic", "identity"]

        # Calibrate weights
        if self.calibration == "cv":
            alphas = self._calibrate_cv(returns_subset, S_sample, targets)
        else:
            alphas = self._calibrate_frobenius(S_sample, targets)

        c_sample = 1.0 - alphas.sum()

        self.last_weights_ = {
            "sample": float(c_sample),
            **{name: float(a) for name, a in zip(target_names, alphas)},
        }

        logger.info(
            "Multi-target [%s] weights: sample=%.3f, text=%.3f, sic=%.3f, identity=%.3f (p=%d, n=%d)",
            self.calibration, c_sample, alphas[0], alphas[1], alphas[2],
            len(common_tickers), len(returns_subset),
        )

        # Build combined estimate
        Sigma_hat = c_sample * S_sample
        for alpha, T in zip(alphas, targets):
            Sigma_hat += alpha * T

        Sigma_hat = _ensure_psd(Sigma_hat, self.config.min_eigenvalue)
        return pd.DataFrame(Sigma_hat, index=common_tickers, columns=common_tickers)
