# PortfolioNLP

Semantic shrinkage for high-dimensional covariance estimation using text similarity priors from corporate disclosures.

Standard covariance estimators like Ledoit-Wolf shrink toward a scaled-identity target that encodes no information about which assets move together. PortfolioNLP replaces that target with a text-similarity matrix derived from SEC 10-K filings, earnings call transcripts, and financial news. Pairwise cosine similarities between firm-level text embeddings serve as a structured prior over the correlation matrix, capturing cross-sectional relationships that would take years of return data to learn.

The full paper is in [`docs/Abstract.md`](docs/Abstract.md).

## Key findings

In a walk-forward quarterly-rebalanced backtest spanning 2007 to 2025 with up to 4,694 US equities:

- **Text provides the correlation structure; returns need only supply volatilities.** Using text-based correlations with just 63 to 252 days of sample volatilities produces Sharpe ratios of 0.71 to 0.77, statistically significantly outperforming Ledoit-Wolf with 504 days of full return history (Sharpe 0.54, p = 0.011).

- **The advantage scales with dimensionality.** At p = 2000 (p/n ~ 4), semantic shrinkage delivers nearly double the Sharpe of Ledoit-Wolf (0.30 vs 0.15). VaR calibration confirms the mechanism: at p = 2000, semantic is the only method whose 2.5% VaR passes the Kupiec coverage test.

- **Signal is structural, not model-dependent.** A TF-IDF baseline using only corpus-internal term frequencies (zero pretrained model knowledge) produces statistically indistinguishable results from neural embeddings (Sharpe 0.592 vs 0.592, p = 0.849), confirming that the signal arises from textual structure rather than a pretrained model's future knowledge.

- **The result replicates across three text sources** (10-K filings, earnings transcripts, financial news) and is robust to transaction cost assumptions, block bootstrap parameters, and sub-period splits.

## Quick start

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/MatthieuSepart/PortfolioNLP.git
cd PortfolioNLP
uv sync
```

Copy the environment template and add your [Tiingo API key](https://www.tiingo.com/):

```bash
cp .env.example .env
# Edit .env with your key
```

Then run the pipeline:

```bash
make data          # Download SEC filings, transcripts, prices (~30 min)
make embeddings    # Embed all text sources (~60 min, GPU recommended)
make backtest      # Core backtest: semantic vs Ledoit-Wolf vs 1/N (~35 min)
make experiments   # Full experiment suite (lookback ablation, scaling, VaR, etc.)
make robustness    # FF regressions, shuffle tests, sector analysis
```

Results are written to `data/results/` as JSON and Parquet files.

## Project structure

```
pnlp/                  Python package
  baselines/            Ledoit-Wolf, equal-weight, SIC sector baselines
  data/                 Data loaders (SEC filings, prices, embeddings, universe filter)
  embeddings/           Text encoding, document and firm aggregation
  portfolio/            Minimum-variance optimizer (cvxpy/OSQP), rebalancer
  primitives/           Covariance estimators (semantic shrinkage, PCA factor model)
  validation/           Backtest engine, performance metrics, statistical tests
scripts/                Experiment and data pipeline scripts
tests/                  Test suite (pytest)
docs/                   Paper and supporting documentation
```

## Running tests

```bash
uv run pytest
uv run pytest -m "not slow"   # Skip tests that load ML models
```

## License

CC BY-NC-SA 4.0. See [LICENSE](LICENSE).
