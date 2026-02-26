"""Tests for the portfolio primitives (mu, sigma, covariance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pnlp.config import CovarianceConfig, MuConfig, SigmaConfig
from pnlp.primitives.covariance import (
    CosineSimilarityCovariance,
    PCAFactorCovariance,
    SemanticShrinkageCovariance,
    _ensure_psd,
)
from pnlp.primitives.mu import ClusterDistanceMu, ReturnsDirectionMu
from pnlp.primitives.sigma import CentroidDistanceSigma, DispersionSigma, TemporalPCASigma


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


class TestDispersionSigma:
    def test_output_length(self, firm_embeddings, doc_embeddings):
        sigma = DispersionSigma().estimate(firm_embeddings, doc_embeddings)
        assert len(sigma) == len(firm_embeddings)

    def test_non_negative(self, firm_embeddings, doc_embeddings):
        sigma = DispersionSigma().estimate(firm_embeddings, doc_embeddings)
        assert (sigma >= 0).all()

    def test_without_docs_returns_uniform(self, firm_embeddings):
        sigma = DispersionSigma().estimate(firm_embeddings)
        assert (sigma == 1.0).all()


class TestCentroidDistanceSigma:
    def test_output_length(self, firm_embeddings):
        sigma = CentroidDistanceSigma().estimate(firm_embeddings)
        assert len(sigma) == len(firm_embeddings)

    def test_normalized_mean_one(self, firm_embeddings):
        sigma = CentroidDistanceSigma().estimate(firm_embeddings)
        assert abs(sigma.mean() - 1.0) < 0.01


class TestReturnsDirectionMu:
    def test_fit_predict(self, firm_embeddings, daily_returns):
        mu_est = ReturnsDirectionMu(MuConfig(regularization=1.0))
        # Use cumulative returns as training signal
        historical = daily_returns.sum()
        mu_est.fit(firm_embeddings, historical)
        predictions = mu_est.predict(firm_embeddings)
        assert len(predictions) == len(firm_embeddings)
        assert predictions.isna().sum() == 0

    def test_unfitted_returns_zeros(self, firm_embeddings):
        mu_est = ReturnsDirectionMu()
        predictions = mu_est.predict(firm_embeddings)
        assert (predictions == 0.0).all()


class TestClusterDistanceMu:
    def test_fit_predict(self, firm_embeddings, daily_returns):
        config = MuConfig(method="cluster_distance", n_clusters=3)
        mu_est = ClusterDistanceMu(config)
        historical = daily_returns.sum()
        mu_est.fit(firm_embeddings, historical)
        predictions = mu_est.predict(firm_embeddings)
        assert len(predictions) == len(firm_embeddings)


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


class TestTemporalPCASigma:
    def test_output_length(self, firm_embeddings, doc_embeddings):
        sigma = TemporalPCASigma().estimate(firm_embeddings, doc_embeddings)
        assert len(sigma) == len(firm_embeddings)

    def test_positive(self, firm_embeddings, doc_embeddings):
        sigma = TemporalPCASigma().estimate(firm_embeddings, doc_embeddings)
        assert (sigma > 0).all()

    def test_few_docs_fallback(self, firm_embeddings):
        """Firms with < 3 docs should get sigma ~ 1.0."""
        from pnlp.embeddings.document_embedder import DocumentEmbedding
        from datetime import date

        # Create doc_embeddings with only 1 doc per firm
        sparse_docs = {}
        for ticker in firm_embeddings:
            sparse_docs[ticker] = [
                DocumentEmbedding(
                    ticker=ticker,
                    doc_date=date(2020, 1, 1),
                    doc_type="10-K",
                    embedding=firm_embeddings[ticker].embedding,
                    norm=1.0,
                    n_chunks=1,
                )
            ]
        sigma = TemporalPCASigma().estimate(firm_embeddings, sparse_docs)
        # All should be 1.0 (fallback)
        np.testing.assert_array_almost_equal(sigma.values, np.ones(len(sigma)))

    def test_stable_vs_volatile(self):
        """Firm with identical docs should have lower sigma than firm with varied docs."""
        from pnlp.embeddings.document_embedder import DocumentEmbedding
        from datetime import date

        rng = np.random.RandomState(123)
        base = rng.randn(768).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Stable firm: all docs have nearly identical embeddings
        stable_docs = []
        for i in range(10):
            emb = base + rng.randn(768).astype(np.float32) * 0.001
            emb = emb / np.linalg.norm(emb)
            stable_docs.append(DocumentEmbedding(
                ticker="STABLE", doc_date=date(2020, 1, 1) + pd.Timedelta(days=i * 90),
                doc_type="10-K", embedding=emb, norm=1.0, n_chunks=1,
            ))

        # Volatile firm: docs vary wildly
        volatile_docs = []
        for i in range(10):
            emb = rng.randn(768).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            volatile_docs.append(DocumentEmbedding(
                ticker="VOLATILE", doc_date=date(2020, 1, 1) + pd.Timedelta(days=i * 90),
                doc_type="10-K", embedding=emb, norm=1.0, n_chunks=1,
            ))

        from pnlp.embeddings.firm_aggregator import FirmEmbedding
        firm_embs = {
            "STABLE": FirmEmbedding(ticker="STABLE", as_of_date=date(2024, 1, 1), embedding=base, n_documents=10),
            "VOLATILE": FirmEmbedding(ticker="VOLATILE", as_of_date=date(2024, 1, 1), embedding=base, n_documents=10),
        }
        doc_embs = {"STABLE": stable_docs, "VOLATILE": volatile_docs}

        sigma = TemporalPCASigma().estimate(firm_embs, doc_embs)
        assert sigma["STABLE"] < sigma["VOLATILE"]

    def test_concentration_ratios_stored(self, firm_embeddings, doc_embeddings):
        estimator = TemporalPCASigma()
        estimator.estimate(firm_embeddings, doc_embeddings)
        assert len(estimator.concentration_ratios_) == len(firm_embeddings)
        # All concentration ratios should be between 0 and 1
        for ticker, cr in estimator.concentration_ratios_.items():
            if not np.isnan(cr):
                assert 0 <= cr <= 1
