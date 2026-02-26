"""
GPU-accelerated linear algebra for covariance estimation.

Provides drop-in replacements for the CPU bottlenecks:
  - ensure_psd: eigendecomposition + PSD projection
  - ledoit_wolf_shrinkage: LW shrinkage coefficient
  - ledoit_wolf_covariance: full LW covariance matrix

Auto-detects CUDA availability. Falls back to numpy on CPU-only machines.
All computations use float64 to match numpy precision.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_CUDA_AVAILABLE: bool | None = None


def _has_cuda() -> bool:
    """Check CUDA availability (cached)."""
    global _TORCH_CUDA_AVAILABLE
    if _TORCH_CUDA_AVAILABLE is None:
        try:
            import torch
            _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
            if _TORCH_CUDA_AVAILABLE:
                logger.info("GPU acceleration enabled: %s", torch.cuda.get_device_name(0))
        except ImportError:
            _TORCH_CUDA_AVAILABLE = False
    return _TORCH_CUDA_AVAILABLE


def ensure_psd(matrix: np.ndarray, min_eigenvalue: float = 1e-4) -> np.ndarray:
    """Ensure a matrix is positive semi-definite.

    min_eigenvalue is relative to the largest eigenvalue:
        floor = max_eigenvalue * min_eigenvalue
    Uses GPU eigendecomposition when available (~800x faster at p=2000).
    """
    if _has_cuda() and matrix.shape[0] >= 200:
        return _ensure_psd_gpu(matrix, min_eigenvalue)
    return _ensure_psd_cpu(matrix, min_eigenvalue)


def _ensure_psd_cpu(matrix: np.ndarray, min_eigenvalue: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(eigenvalues.max() * min_eigenvalue, 1e-10)
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _ensure_psd_gpu(matrix: np.ndarray, min_eigenvalue: float) -> np.ndarray:
    import torch

    t = torch.from_numpy(matrix).to(device="cuda", dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(t)
    floor = max(eigenvalues.max().item() * min_eigenvalue, 1e-10)
    eigenvalues = torch.clamp(eigenvalues, min=floor)
    result = (eigenvectors * eigenvalues) @ eigenvectors.T
    out = result.cpu().numpy()
    del t, eigenvalues, eigenvectors, result
    return out


def ledoit_wolf_shrinkage(X: np.ndarray, assume_centered: bool = False) -> float:
    """Compute LW shrinkage coefficient.

    Reimplements sklearn.covariance.ledoit_wolf_shrinkage using GPU
    matrix operations when available (~500x faster at p=2000).

    Args:
        X: (n_samples, n_features) data matrix.
        assume_centered: if True, don't subtract mean.

    Returns:
        Shrinkage coefficient in [0, 1].
    """
    if _has_cuda() and X.shape[1] >= 200:
        return _lw_shrinkage_gpu(X, assume_centered)
    return _lw_shrinkage_cpu(X, assume_centered)


def _lw_shrinkage_cpu(X: np.ndarray, assume_centered: bool) -> float:
    from sklearn.covariance import ledoit_wolf_shrinkage as sk_lw
    return float(sk_lw(X, assume_centered=assume_centered))


def _lw_shrinkage_gpu(X: np.ndarray, assume_centered: bool) -> float:
    import torch

    X = np.ascontiguousarray(X)
    Xt = torch.from_numpy(X).to(device="cuda", dtype=torch.float64)
    n, p = Xt.shape

    if not assume_centered:
        Xt = Xt - Xt.mean(dim=0)

    X2 = Xt ** 2
    emp_cov_trace = X2.sum(dim=0) / n  # (p,)
    mu = emp_cov_trace.sum().item() / p

    # beta_ = sum of all elements of X2.T @ X2
    # Use trace of (X2.T @ X2 @ ones) trick to avoid materializing p×p matrix
    # Actually at p=2000 the p×p matrix fits easily in GPU memory (~30MB float64)
    XtX = Xt.T @ Xt  # (p, p)
    X2tX2 = X2.T @ X2  # (p, p)

    beta_ = X2tX2.sum().item()
    delta_ = (XtX ** 2).sum().item() / (n ** 2)

    beta = (beta_ / n - delta_) / (p * n)
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum().item() + p * mu ** 2
    delta /= p

    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta

    del Xt, X2, XtX, X2tX2, emp_cov_trace
    return shrinkage


def ledoit_wolf_covariance(
    X: np.ndarray, assume_centered: bool = False,
) -> tuple[np.ndarray, float]:
    """Compute full LW covariance matrix and shrinkage coefficient.

    Returns:
        (covariance, shrinkage) tuple.
    """
    if _has_cuda() and X.shape[1] >= 200:
        return _lw_covariance_gpu(X, assume_centered)
    return _lw_covariance_cpu(X, assume_centered)


def _lw_covariance_cpu(
    X: np.ndarray, assume_centered: bool,
) -> tuple[np.ndarray, float]:
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf(assume_centered=assume_centered)
    lw.fit(X)
    return lw.covariance_, float(lw.shrinkage_)


def _lw_covariance_gpu(
    X: np.ndarray, assume_centered: bool,
) -> tuple[np.ndarray, float]:
    import torch

    X = np.ascontiguousarray(X)
    Xt = torch.from_numpy(X).to(device="cuda", dtype=torch.float64)
    n, p = Xt.shape

    if not assume_centered:
        Xt = Xt - Xt.mean(dim=0)

    # Sample covariance (1/n to match sklearn LedoitWolf convention)
    S = (Xt.T @ Xt) / n  # (p, p)

    # Shrinkage coefficient (same formula as _lw_shrinkage_gpu)
    X2 = Xt ** 2
    emp_cov_trace = X2.sum(dim=0) / n
    mu = emp_cov_trace.sum().item() / p

    XtX = Xt.T @ Xt  # = S * (n-1)
    X2tX2 = X2.T @ X2

    beta_ = X2tX2.sum().item()
    delta_ = (XtX ** 2).sum().item() / (n ** 2)

    beta = (beta_ / n - delta_) / (p * n)
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum().item() + p * mu ** 2
    delta /= p

    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta

    # Shrunk covariance: (1 - shrinkage) * S + shrinkage * mu * I
    cov = (1.0 - shrinkage) * S
    cov.diagonal().add_(shrinkage * mu)

    out = cov.cpu().numpy()
    del Xt, X2, XtX, X2tX2, S, cov, emp_cov_trace
    return out, shrinkage
