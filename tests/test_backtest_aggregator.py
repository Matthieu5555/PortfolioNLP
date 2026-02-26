"""Tests for BacktestEngine (backtest.py) and FirmAggregator (firm_aggregator.py)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pnlp.config import BacktestConfig, EmbeddingConfig
from pnlp.embeddings.document_embedder import DocumentEmbedding
from pnlp.embeddings.firm_aggregator import FirmAggregator, FirmEmbedding
from pnlp.portfolio.optimizer import PortfolioWeights
from pnlp.validation.backtest import BacktestEngine

from tests.conftest import SYNTHETIC_FIRMS, make_doc_embeddings, make_returns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weights(weight_dict: dict[str, float]) -> PortfolioWeights:
    return PortfolioWeights(
        weights=pd.Series(weight_dict),
        objective_value=0.0,
        metadata={},
    )


# ===================================================================
# BacktestEngine tests
# ===================================================================


class TestBacktestEngine:

    def test_log_to_simple_conversion(self):
        # Two stocks, one day of log returns
        tickers = ["A", "B"]
        log_ret_a = 0.05
        log_ret_b = -0.03
        rebal_date = date(2022, 1, 3)  # Monday
        day = pd.bdate_range(start=rebal_date, periods=1)
        log_returns = pd.DataFrame(
            [[log_ret_a, log_ret_b]], index=day, columns=tickers,
        )

        weights_history = {rebal_date: _make_weights({"A": 0.6, "B": 0.4})}
        config = BacktestConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 2, 1),
            transaction_cost_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run(weights_history, log_returns, strategy_name="test")

        simple_a = np.exp(log_ret_a) - 1
        simple_b = np.exp(log_ret_b) - 1
        expected = 0.6 * simple_a + 0.4 * simple_b
        # TC on first rebal is flat_tc_rate = 0 bps
        np.testing.assert_almost_equal(result.portfolio_returns.iloc[0], expected, decimal=10)

    def test_weight_drift_between_rebalances(self):
        tickers = ["A", "B"]
        rebal_date = date(2022, 1, 3)
        # Two days of simple returns (feed as log returns so engine converts)
        simple_day1 = np.array([0.10, -0.05])
        log_day1 = np.log(1 + simple_day1)
        simple_day2 = np.array([0.02, 0.01])
        log_day2 = np.log(1 + simple_day2)

        days = pd.bdate_range(start=rebal_date, periods=2)
        log_returns = pd.DataFrame(
            np.vstack([log_day1, log_day2]), index=days, columns=tickers,
        )

        weights_history = {rebal_date: _make_weights({"A": 0.5, "B": 0.5})}
        config = BacktestConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 2, 1),
            transaction_cost_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run(weights_history, log_returns, strategy_name="test")

        # After day 1, weights drift from (0.5, 0.5):
        # A: 0.5 * 1.10 = 0.55, B: 0.5 * 0.95 = 0.475
        # Total: 1.025 -> drifted: A=0.55/1.025, B=0.475/1.025
        w_a_drifted = (0.5 * 1.10) / (0.5 * 1.10 + 0.5 * 0.95)
        w_b_drifted = (0.5 * 0.95) / (0.5 * 1.10 + 0.5 * 0.95)

        # Day 2 portfolio return should use drifted weights
        expected_day2 = w_a_drifted * simple_day2[0] + w_b_drifted * simple_day2[1]
        np.testing.assert_almost_equal(result.portfolio_returns.iloc[1], expected_day2, decimal=10)

    def test_transaction_costs_applied_first_day_only(self):
        tickers = ["A", "B"]
        rebal_date = date(2022, 1, 3)
        days = pd.bdate_range(start=rebal_date, periods=3)
        # Constant zero log returns so portfolio return is purely from TC
        log_returns = pd.DataFrame(
            np.zeros((3, 2)), index=days, columns=tickers,
        )

        tc_bps = 50.0  # 50 bps = 0.005
        flat_tc = tc_bps / 10000.0
        weights_history = {rebal_date: _make_weights({"A": 0.5, "B": 0.5})}
        config = BacktestConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 2, 1),
            transaction_cost_bps=tc_bps,
        )
        engine = BacktestEngine(config)
        result = engine.run(weights_history, log_returns, strategy_name="test")

        # First rebalance with no prior weights -> tc = flat_tc_rate
        np.testing.assert_almost_equal(result.portfolio_returns.iloc[0], -flat_tc, decimal=10)
        # Subsequent days: no TC deducted, zero returns
        np.testing.assert_almost_equal(result.portfolio_returns.iloc[1], 0.0, decimal=10)
        np.testing.assert_almost_equal(result.portfolio_returns.iloc[2], 0.0, decimal=10)

    def test_turnover_uses_drifted_weights(self):
        tickers = ["A", "B"]
        rebal1 = date(2022, 1, 3)   # Monday
        rebal2 = date(2022, 1, 5)   # Wednesday

        # bdate_range: Jan3(Mon), Jan4(Tue), Jan5(Wed), Jan6(Thu)
        days = pd.bdate_range(start=rebal1, periods=4)
        # Period 1 (Jan3-Jan4): day1 has big returns, day2 zero
        simple_day1 = np.array([0.20, -0.10])
        log_day1 = np.log(1 + simple_day1)
        # Period 2 (Jan5-Jan6): zero returns to isolate TC effect
        log_returns = pd.DataFrame(
            np.vstack([log_day1, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            index=days, columns=tickers,
        )

        # Both rebalances target equal weight
        weights_history = {
            rebal1: _make_weights({"A": 0.5, "B": 0.5}),
            rebal2: _make_weights({"A": 0.5, "B": 0.5}),
        }

        tc_bps = 100.0  # 1% to make TC visible
        flat_tc = tc_bps / 10000.0
        config = BacktestConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 2, 1),
            transaction_cost_bps=tc_bps,
        )
        engine = BacktestEngine(config)
        result = engine.run(weights_history, log_returns, strategy_name="test")

        # Period 1 has Jan3 and Jan4. After Jan3 returns, weights drift:
        # A: 0.5*1.20 = 0.60, B: 0.5*0.90 = 0.45, total = 1.05
        # Jan4 has zero returns so drift stays the same through end of period 1.
        w_a_drifted = 0.60 / 1.05
        w_b_drifted = 0.45 / 1.05

        # At rebal2, turnover from drifted -> target (0.5, 0.5):
        turnover = abs(0.5 - w_a_drifted) + abs(0.5 - w_b_drifted)
        expected_tc = turnover * flat_tc

        # Series: iloc[0]=Jan3, iloc[1]=Jan4, iloc[2]=Jan5(rebal2), iloc[3]=Jan6
        # Jan5 is first day of period 2, zero returns minus TC
        np.testing.assert_almost_equal(result.portfolio_returns.iloc[2], -expected_tc, decimal=10)

        # If turnover were computed from target weights (0.5,0.5) -> (0.5,0.5),
        # TC would be zero. The nonzero TC proves drifted weights were used.
        assert abs(result.portfolio_returns.iloc[2]) > 1e-6

    def test_empty_weights_history(self):
        tickers = [f.ticker for f in SYNTHETIC_FIRMS[:3]]
        returns = make_returns(tickers, n_days=50)
        engine = BacktestEngine()
        result = engine.run({}, returns, strategy_name="empty")

        assert len(result.portfolio_returns) == 0
        assert isinstance(result.portfolio_returns, pd.Series)


# ===================================================================
# FirmAggregator tests
# ===================================================================


def _make_doc(ticker, doc_date, embedding):
    return DocumentEmbedding(
        ticker=ticker,
        doc_date=doc_date,
        doc_type="10-K",
        embedding=np.array(embedding, dtype=np.float32),
        norm=float(np.linalg.norm(embedding)),
        n_chunks=1,
    )


class TestFirmAggregator:

    def test_temporal_filtering(self):
        docs = [
            _make_doc("AAPL", date(2020, 3, 1), [1.0, 0.0, 0.0]),
            _make_doc("AAPL", date(2021, 3, 1), [0.0, 1.0, 0.0]),
            _make_doc("AAPL", date(2022, 3, 1), [0.0, 0.0, 1.0]),
            _make_doc("AAPL", date(2023, 3, 1), [1.0, 1.0, 0.0]),
        ]
        agg = FirmAggregator()
        result = agg.aggregate(docs, as_of_date=date(2021, 6, 1))

        assert result is not None
        assert result.n_documents == 2
        assert result.as_of_date == date(2021, 6, 1)

    def test_l2_normalized_at_firm_level(self):
        docs = [
            _make_doc("AAPL", date(2020, 1, 1), [3.0, 4.0, 0.0]),
            _make_doc("AAPL", date(2020, 6, 1), [0.0, 3.0, 4.0]),
        ]
        agg = FirmAggregator()
        result = agg.aggregate(docs, as_of_date=date(2021, 1, 1))

        assert result is not None
        np.testing.assert_almost_equal(np.linalg.norm(result.embedding), 1.0, decimal=6)

    def test_raw_embeddings_not_prenormalized(self):
        # Two docs with different norms — mean should be on raw vectors
        v1 = np.array([3.0, 0.0, 0.0], dtype=np.float32)  # norm=3
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # norm=1
        docs = [
            _make_doc("AAPL", date(2020, 1, 1), v1),
            _make_doc("AAPL", date(2020, 6, 1), v2),
        ]
        agg = FirmAggregator()
        result = agg.aggregate(docs, as_of_date=date(2021, 1, 1))

        # Raw mean: (3+0)/2, (0+1)/2, (0+0)/2 = [1.5, 0.5, 0.0]
        raw_mean = (v1 + v2) / 2.0
        expected = raw_mean / np.linalg.norm(raw_mean)

        # If docs were pre-normalized, mean would be ([1,0,0]+[0,1,0])/2 = [0.5,0.5,0]
        # normalized to [1/sqrt(2), 1/sqrt(2), 0] — which is different.
        prenorm_mean = (v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)) / 2.0
        prenorm_result = prenorm_mean / np.linalg.norm(prenorm_mean)

        np.testing.assert_array_almost_equal(result.embedding, expected, decimal=5)
        # Confirm the result differs from what pre-normalization would give
        assert not np.allclose(result.embedding, prenorm_result, atol=1e-3)

    def test_time_weighted_aggregation(self):
        as_of = date(2021, 1, 1)
        halflife = 365  # days
        # Doc 1: 365 days ago -> weight = exp(-ln2 * 365/365) = 0.5
        # Doc 2: 0 days ago   -> weight = exp(0) = 1.0
        d1 = date(2020, 1, 1)  # exactly 366 days ago from 2021-01-01 (leap year)
        d2 = date(2021, 1, 1)  # 0 days ago

        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        docs = [
            _make_doc("AAPL", d1, v1),
            _make_doc("AAPL", d2, v2),
        ]

        config = EmbeddingConfig(aggregation="time_weighted", time_decay_halflife_days=halflife)
        agg = FirmAggregator(config)
        result = agg.aggregate(docs, as_of_date=as_of)

        # Compute expected weights manually
        days_ago_1 = (as_of - d1).days  # 366
        days_ago_2 = (as_of - d2).days  # 0
        w1 = np.exp(-np.log(2) * days_ago_1 / halflife)
        w2 = np.exp(-np.log(2) * days_ago_2 / halflife)
        total_w = w1 + w2
        w1_norm = w1 / total_w
        w2_norm = w2 / total_w

        raw_agg = w1_norm * v1 + w2_norm * v2
        expected = raw_agg / np.linalg.norm(raw_agg)

        assert result is not None
        np.testing.assert_array_almost_equal(result.embedding, expected, decimal=5)
        # Recent doc (d2) should have more influence: embedding[1] > embedding[0]
        assert result.embedding[1] > result.embedding[0]

    def test_min_documents_filter(self):
        # Create universe: AAPL has 4 docs before cutoff, MSFT has 2
        as_of = date(2022, 1, 1)
        all_docs = {
            "AAPL": [
                _make_doc("AAPL", date(2020, 1, 1), [1, 0, 0]),
                _make_doc("AAPL", date(2020, 6, 1), [0, 1, 0]),
                _make_doc("AAPL", date(2021, 1, 1), [0, 0, 1]),
                _make_doc("AAPL", date(2021, 6, 1), [1, 1, 0]),
            ],
            "MSFT": [
                _make_doc("MSFT", date(2020, 1, 1), [1, 0, 0]),
                _make_doc("MSFT", date(2021, 1, 1), [0, 1, 0]),
            ],
            "GOOGL": [
                _make_doc("GOOGL", date(2020, 1, 1), [0, 0, 1]),
                _make_doc("GOOGL", date(2021, 1, 1), [1, 0, 0]),
                _make_doc("GOOGL", date(2021, 6, 1), [0, 1, 0]),
                # One doc after as_of — should be excluded by temporal filter
                _make_doc("GOOGL", date(2023, 1, 1), [1, 1, 1]),
            ],
        }

        agg = FirmAggregator()
        result = agg.aggregate_universe(all_docs, as_of_date=as_of, min_documents=3)

        # AAPL: 4 docs <= as_of -> included
        # MSFT: 2 docs <= as_of -> excluded (< 3)
        # GOOGL: 3 docs <= as_of (the 2023 doc is filtered out) -> included
        assert "AAPL" in result
        assert "MSFT" not in result
        assert "GOOGL" in result
        assert result["AAPL"].n_documents == 4
        assert result["GOOGL"].n_documents == 3
