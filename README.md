# PortfolioNLP

Semantic shrinkage for high-dimensional covariance estimation using text similarity priors from corporate disclosures. Standard covariance estimators like Ledoit-Wolf shrink toward a scaled-identity target that encodes no information about which assets move together. PortfolioNLP replaces that target with a text-similarity matrix derived from SEC 10-K filings, earnings call transcripts, and financial news.

The draft paper is in docs/Abstract.md. The core library lives in pnlp/ (covariance estimators, portfolio optimizer, backtest engine, data loaders, baselines), experiment scripts that produce paper results are in experiments/, data download and embedding scripts are in pipeline/, tests in tests/, and the paper with supporting documentation in docs/.

Requires Python 3.13+ and uv. Clone, run uv sync, copy .env.example to .env with a Tiingo API key, then run scripts from pipeline/ to download and embed data and from experiments/ to reproduce results. Run uv run pytest to run the test suite.

CC BY-NC-SA 4.0.
