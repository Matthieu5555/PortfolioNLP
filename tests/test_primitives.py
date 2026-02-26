"""Tests for the covariance primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pnlp.config import CovarianceConfig
from pnlp.primitives.covariance import (
    CosineSimilarityCovariance,
    PCAFactorCovariance,
    SemanticShrinkageCovariance,
    _ensure_psd,
)


class TestEnsurePSD:
    def test_already_psd(self):
        A = np.eye(3)
        result = _ensure_psd(A)
        np.testing.assert_array_almost_equal(result, A)

    def test_fixes_negative_eigenvalues(self):
        # Create a matrix with a negative eigenvalue
        A = np.array([[1, 2], [2, 1]], dtype=float)  # eigenvalues: 3, -1
        result = _ensure_psd(A, min_eigenvalue=0.01)
        eigenvalues = np.linalg.eigvalsh(result)
        assert eigenvalues.min() >= 0.01 - 1e-10


class TestCosineSimilarityCovariance:
    def test_output_shape(self, firm_embeddings):
        estimator = CosineSimilarityCovariance()
        cov = estimator.estimate(firm_embeddings)
        n = len(firm_embeddings)
        assert cov.shape == (n, n)

    def test_symmetric(self, firm_embeddings):
        cov = CosineSimilarityCovariance().estimate(firm_embeddings)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_psd(self, firm_embeddings):
        cov = CosineSimilarityCovariance().estimate(firm_embeddings)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert eigenvalues.min() >= 0, "Covariance matrix should be PSD"

    def test_diagonal_is_one_without_sigma(self, firm_embeddings):
        """Without sigma estimates, diagonal should be ~1 (correlation proxy)."""
        cov = CosineSimilarityCovariance().estimate(firm_embeddings)
        np.testing.assert_array_almost_equal(np.diag(cov.values), np.ones(len(cov)), decimal=1)

    def test_with_sigma_estimates(self, firm_embeddings):
        tickers = sorted(firm_embeddings.keys())
        sigma = pd.Series(np.random.rand(len(tickers)) * 0.3 + 0.1, index=tickers)
        cov = CosineSimilarityCovariance().estimate(firm_embeddings, sigma)
        # Diagonal should be sigma_i^2
        for t in tickers:
            assert abs(cov.loc[t, t] - sigma[t] ** 2) < 0.1 * sigma[t] ** 2


class TestSemanticShrinkageCovariance:
    def test_with_returns(self, firm_embeddings, daily_returns):
        estimator = SemanticShrinkageCovariance(
            historical_returns=daily_returns,
        )
        cov = estimator.estimate(firm_embeddings)
        assert cov.shape[0] > 0
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_without_returns_falls_back(self, firm_embeddings):
        estimator = SemanticShrinkageCovariance()
        cov = estimator.estimate(firm_embeddings)
        assert cov.shape[0] == len(firm_embeddings)


class TestPCAFactorCovariance:
    def test_output_shape(self, firm_embeddings):
        estimator = PCAFactorCovariance(n_factors=5)
        cov = estimator.estimate(firm_embeddings)
        n = len(firm_embeddings)
        assert cov.shape == (n, n)

    def test_symmetric(self, firm_embeddings):
        cov = PCAFactorCovariance(n_factors=5).estimate(firm_embeddings)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_psd(self, firm_embeddings):
        cov = PCAFactorCovariance(n_factors=3).estimate(firm_embeddings)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert eigenvalues.min() >= 0, "PCA covariance should be PSD"

    def test_diagonal_one_without_returns(self, firm_embeddings):
        """Without returns/sigma, diagonal should be ~1 (correlation proxy)."""
        cov = PCAFactorCovariance(n_factors=5).estimate(firm_embeddings)
        np.testing.assert_array_almost_equal(np.diag(cov.values), np.ones(len(cov)), decimal=1)

    def test_full_rank_matches_cosine(self, firm_embeddings):
        """At k=full rank, PCA should match cosine similarity (up to PSD floor)."""
        pca_cov = PCAFactorCovariance(n_factors=768).estimate(firm_embeddings)
        cos_cov = CosineSimilarityCovariance().estimate(firm_embeddings)
        np.testing.assert_array_almost_equal(pca_cov.values, cos_cov.values, decimal=3)

    def test_explained_variance(self, firm_embeddings):
        estimator = PCAFactorCovariance(n_factors=5)
        estimator.estimate(firm_embeddings)
        ev = estimator.explained_variance_ratio_
        assert ev is not None
        assert len(ev) == 5
        assert ev.sum() <= 1.0 + 1e-6
        # Should be in decreasing order
        assert all(ev[i] >= ev[i + 1] - 1e-10 for i in range(len(ev) - 1))

    def test_with_returns_shrinkage(self, firm_embeddings, daily_returns):
        estimator = PCAFactorCovariance(n_factors=5, historical_returns=daily_returns)
        cov = estimator.estimate(firm_embeddings)
        assert cov.shape[0] > 0
        assert estimator.last_alpha_ is not None
        assert 0 <= estimator.last_alpha_ <= 1
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_low_rank(self, firm_embeddings):
        """At k=1, matrix should have effective rank ~1 (before PSD floor)."""
        estimator = PCAFactorCovariance(n_factors=1)
        cov = estimator.estimate(firm_embeddings)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        # After PSD floor, all eigenvalues are positive, but the ratio of
        # largest to second-largest should be very large
        sorted_ev = np.sort(eigenvalues)[::-1]
        assert sorted_ev[0] / sorted_ev[1] > 10
