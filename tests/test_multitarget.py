"""Tests for MultiTargetShrinkageCovariance and cross_validate_alpha."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pnlp.config import CovarianceConfig
from pnlp.embeddings.firm_aggregator import FirmEmbedding
from pnlp.primitives.covariance import (
    MultiTargetShrinkageCovariance,
    SemanticShrinkageCovariance,
    cross_validate_alpha,
)


def _make_embeddings(n: int, dim: int = 32) -> dict[str, FirmEmbedding]:
    """Create random L2-normalized firm embeddings."""
    rng = np.random.default_rng(42)
    emb = {}
    for i in range(n):
        vec = rng.standard_normal(dim)
        vec = vec / np.linalg.norm(vec)
        emb[f"T{i:03d}"] = FirmEmbedding(
            ticker=f"T{i:03d}",
            as_of_date=date(2023, 1, 1),
            embedding=vec,
            n_documents=5,
        )
    return emb


def _make_returns(n: int, n_obs: int = 252) -> pd.DataFrame:
    """Create random return matrix."""
    rng = np.random.default_rng(42)
    tickers = [f"T{i:03d}" for i in range(n)]
    dates = pd.bdate_range("2021-01-01", periods=n_obs)
    data = rng.standard_normal((n_obs, n)) * 0.02
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_sic_codes(n: int) -> dict[str, str]:
    """Create SIC codes assigning firms to 3 sectors."""
    sic = {}
    sectors = ["36", "28", "73"]
    for i in range(n):
        sic[f"T{i:03d}"] = sectors[i % len(sectors)]
    return sic


class TestMultiTargetShrinkage:
    """Tests for MultiTargetShrinkageCovariance."""

    def test_produces_psd_matrix(self) -> None:
        n = 30
        emb = _make_embeddings(n)
        rets = _make_returns(n)
        sic = _make_sic_codes(n)

        est = MultiTargetShrinkageCovariance(
            historical_returns=rets,
            sic_codes=sic,
        )
        cov = est.estimate(emb)

        assert cov.shape == (n, n)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-8), f"Not PSD: min eigenvalue={eigenvalues.min()}"

    def test_symmetric(self) -> None:
        n = 20
        emb = _make_embeddings(n)
        rets = _make_returns(n)
        sic = _make_sic_codes(n)

        est = MultiTargetShrinkageCovariance(
            historical_returns=rets,
            sic_codes=sic,
        )
        cov = est.estimate(emb)

        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)

    def test_weights_sum_leq_one(self) -> None:
        n = 30
        emb = _make_embeddings(n)
        rets = _make_returns(n)
        sic = _make_sic_codes(n)

        est = MultiTargetShrinkageCovariance(
            historical_returns=rets,
            sic_codes=sic,
        )
        est.estimate(emb)

        assert est.last_weights_ is not None
        total = sum(est.last_weights_.values())
        assert total <= 1.0 + 1e-6, f"Weights sum to {total}"

    def test_weights_non_negative(self) -> None:
        n = 30
        emb = _make_embeddings(n)
        rets = _make_returns(n)
        sic = _make_sic_codes(n)

        est = MultiTargetShrinkageCovariance(
            historical_returns=rets,
            sic_codes=sic,
        )
        est.estimate(emb)

        assert est.last_weights_ is not None
        for name, w in est.last_weights_.items():
            assert w >= -1e-6, f"Weight {name}={w} is negative"

    def test_fallback_without_returns(self) -> None:
        n = 20
        emb = _make_embeddings(n)

        est = MultiTargetShrinkageCovariance(
            historical_returns=None,
            sic_codes=_make_sic_codes(n),
        )
        cov = est.estimate(emb)

        assert cov.shape == (n, n)
        # Should fall back to cosine similarity
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-8)

    def test_ticker_alignment(self) -> None:
        """Test that tickers in output match embedding tickers."""
        n = 25
        emb = _make_embeddings(n)
        rets = _make_returns(n)
        sic = _make_sic_codes(n)

        est = MultiTargetShrinkageCovariance(
            historical_returns=rets,
            sic_codes=sic,
        )
        cov = est.estimate(emb)

        assert list(cov.index) == list(cov.columns)
        assert set(cov.index).issubset(set(emb.keys()))


class TestCrossValidateAlpha:
    """Tests for cross_validate_alpha."""

    def test_returns_valid_alpha(self) -> None:
        n = 30
        rets = _make_returns(n, n_obs=252)
        rng = np.random.default_rng(42)
        corr = np.eye(n) + 0.3 * (np.ones((n, n)) - np.eye(n))

        alpha = cross_validate_alpha(rets, corr)
        assert 0.0 <= alpha <= 1.0

    def test_returns_from_grid(self) -> None:
        n = 30
        rets = _make_returns(n, n_obs=252)
        corr = np.eye(n) + 0.3 * (np.ones((n, n)) - np.eye(n))
        grid = [0.1, 0.5, 0.9]

        alpha = cross_validate_alpha(rets, corr, alpha_grid=grid)
        assert alpha in grid

    def test_insufficient_data_returns_default(self) -> None:
        n = 30
        rets = _make_returns(n, n_obs=15)  # Too few observations
        corr = np.eye(n)

        alpha = cross_validate_alpha(rets, corr)
        assert alpha == 0.5  # Default fallback

    def test_cv_mode_in_semantic_shrinkage(self) -> None:
        """Test that shrinkage_intensity='cv' works end-to-end."""
        n = 30
        emb = _make_embeddings(n)
        rets = _make_returns(n, n_obs=252)

        config = CovarianceConfig(
            method="shrinkage",
            shrinkage_intensity="cv",
        )
        est = SemanticShrinkageCovariance(
            config=config,
            historical_returns=rets,
        )
        cov = est.estimate(emb)

        assert cov.shape == (n, n)
        assert est.last_alpha_ is not None
        assert 0.0 <= est.last_alpha_ <= 1.0
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-8)
