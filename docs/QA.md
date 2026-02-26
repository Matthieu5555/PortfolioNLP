# PortfolioNLP: Frequently Asked Questions

---

## 1. What does the model take as input and produce as output?

The system takes as input a list of stock tickers (the investable universe), together with text embeddings derived from corporate filings and historical daily returns. It produces as output a vector of portfolio weights that sum to 100%, with each weight corresponding to one ticker. All weights are non-negative (long-only) and capped at 5% per individual position. The objective is minimum variance, meaning the optimizer seeks to minimize total portfolio risk without requiring any return forecast. The portfolio is rebalanced quarterly, and between rebalance dates, weights drift naturally with market returns.

---

## 2. How many stocks are in the backtest?

The answer depends on the experiment. The universe is capped at *p* firms per quarter, selected as the most liquid stocks that pass all eligibility filters. The tables below give exact counts for each configuration.

### Main backtests (45 quarters, 2013-Q1 to 2024-Q2)

| Text source | p (cap) | Rebalances | Min n | Max n | Mean n |
|-------------|---------|------------|-------|-------|--------|
| 10-K | 500 | 45 | 499 | 500 | 500.0 |
| Transcript | 500 | 45 | 393 | 500 | ~463 |
| Combined (10-K + transcript) | 500 | 45 | 499 | 500 | 500.0 |

### Extended backtests (75 quarters, 2007-Q1 to 2025-Q4)

| Text source | p (cap) | Rebalances | Min n | Max n | Mean n |
|-------------|---------|------------|-------|-------|--------|
| 10-K | 500 | 75 | 499 | 500 | 500.0 |
| Transcript | 500 | 75 | 93 | 500 | ~380 |
| News | 500 | 71 | 499 | 500 | 500.0 |

Transcript coverage is notably lower in the early years of the sample: only 93 firms have transcripts available in 2007-Q2, and that number grows steadily to 500 by 2019-Q2. By contrast, 10-K filings and news articles consistently fill the 500-firm cap across the entire period.

### Scaling experiment (75 quarters, 2007-Q2 to 2025-Q4, 10-K)

| p (cap) | p/n ratio | Min n | Max n | Mean n |
|---------|-----------|-------|-------|--------|
| 50 | 0.10 | 50 | 50 | 50.0 |
| 100 | 0.20 | 100 | 100 | 100.0 |
| 200 | 0.40 | 199 | 200 | 200.0 |
| 500 | 0.99 | 499 | 500 | 500.0 |
| 1,000 | 1.98 | 882 | 1,000 | ~970 |
| 1,500 | 2.98 | 882 | 1,500 | ~1,300 |
| 2,000 | 3.97 | 882 | 2,000 | ~1,500 |

At p >= 1,000, the early quarters (2007-2010) become the binding constraint because fewer firms simultaneously meet the liquidity threshold and have available text embeddings. The universe is variable: at p = 1,000, the first several quarters have only 882-924 eligible firms. All quarters are included, and the portfolio uses whatever firms are available. This means the effective p/n ratio is lower than the nominal ratio in early quarters, which is conservative for evaluating the text-based method's advantage at high dimensionality.

### Other experiments

| Experiment | Text source | p | Rebalances | Period |
|------------|------------|---|------------|--------|
| Lookback ablation | 10-K | 500 | 75 | 2007-Q1 to 2025-Q4 |
| Lookback ablation | Transcript | 500 | 75 | 2007-Q1 to 2025-Q4 |
| Lookback ablation | News | 500 | 75 | 2007-Q1 to 2025-Q4 |
| Alpha sweep | 10-K | 200, 500 | 75 | 2007-Q2 to 2025-Q4 |
| CV-alpha | 10-K | 500 | 71 | 2007-Q1 to 2024-Q4 |
| Multi-target | 10-K | 200, 500 | 71 | 2007-Q1 to 2024-Q4 |
| Constrained | 10-K | 500 | 71 | 2007-Q1 to 2024-Q4 |
| Cold-start | 10-K | 500 | 75 | 2007-Q2 to 2025-Q4 |
| VaR calibration | 10-K | 200, 500, 1000, 2000 | 75 | 2007-Q2 to 2025-Q4 |
| TF-IDF baseline | 10-K | 500 | 75 | 2007-Q1 to 2025-Q4 |
| Correlation study | 10-K | uncapped | 70 | 2007-Q1 to 2024-Q2 |
| Correlation study | News | uncapped | 70 | 2007-Q1 to 2024-Q2 |
| Sector analysis | 10-K | 200 | 75 | 2007-Q2 to 2025-Q4 |
| Shuffle placebo | 10-K | 500 | 75 | 2007-Q2 to 2025-Q4 |
| Scaling | 10-K | 50--2000 | 75 | 2007-Q2 to 2025-Q4 |
| PCA k-sweep | 10-K | 500 | 75 | 2007-Q2 to 2025-Q4 |
| Beta experiment | 10-K | 500 | 75 | 2007-Q2 to 2025-Q4 |
| Temporal sensitivity | 10-K | 500 | 75 | 2007-Q1 to 2025-Q4 |

---

## 3. How is the universe filtered?

At each quarterly rebalance date, the investable universe is constructed by applying four criteria in strict point-in-time fashion, meaning no future information is used.

First, a firm must meet a **liquidity** threshold: its trailing 21-day median average daily dollar volume (ADV) must be at least $1,000,000. Second, it must have sufficient **return history**: at least 80% non-missing daily returns over the preceding 504 trading days (approximately two years). Third, the chosen **text source** (10-K, transcript, or news) must have a pre-computed embedding available for the firm on or before the rebalance date. Fourth, the firm must satisfy all three conditions simultaneously. Among eligible firms, the top *p* by ADV rank are selected.

This filter is re-applied from scratch at every rebalance, using only data available up to the day before the rebalance date.

---

## 4. Where does the text data come from?

The project draws on three distinct text sources, plus a price feed.

**10-K filings** come from the HuggingFace dataset `PleIAs/SEC`, which contains pre-parsed 10-K annual reports filed with the SEC. The dataset covers 60,258 filings across 4,815 tickers from 2000 to 2024. The section used is Item 1 (Business Description), which is the narrative portion describing the company's products, markets, and competitive positioning. Post-2020 filings often contain XBRL machine-readable markup before the narrative text; this is automatically detected and stripped.

**Earnings call transcripts** come from the HuggingFace dataset `kurry/sp500_earnings_transcripts`. It contains 30,596 transcripts, primarily covering S&P 500 constituents from 2006 to 2024. Each transcript includes the operator introduction, management prepared remarks, and the analyst Q&A session.

**Financial news** comes from the HuggingFace dataset `Brianferrell787/financial-news-multisource`, comprising articles from NYT, CNBC, Reuters, Yahoo Finance, and other outlets. After matching articles to tickers and aggregating quarterly, the dataset contains 111,062 quarterly documents across 4,024 tickers.

**Stock prices and returns** are sourced from the Tiingo REST API (adjusted OHLCV data) and cached locally as one parquet file per ticker.

---

## 5. What do the business descriptions look like?

Below are real excerpts from 2023 10-K filings, after automated XBRL cleaning. This is the raw text that gets embedded.

**Apple Inc. (AAPL):**

> The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's fiscal year is the 52- or 53-week period that ends on the last Saturday of September. iPhone is the Company's line of smartphones based on its iOS operating system. The iPhone line includes iPhone 15 Pro, iPhone 15, iPhone 14, iPhone 13 and iPhone SE. Mac is the Company's line of personal computers based on its macOS operating system.

**ExxonMobil (XOM):**

> Exxon Mobil Corporation was incorporated in the State of New Jersey in 1882. Divisions and affiliated companies of ExxonMobil operate or market products in the United States and most other countries of the world. Our principal business involves exploration for, and production of, crude oil and natural gas; manufacture, trade, transport and sale of crude oil, natural gas, petroleum products, petrochemicals, and a wide variety of specialty products; and pursuit of lower-emission business opportunities including carbon capture and storage, hydrogen, lower-emission fuels, and lithium.

**Microsoft (MSFT):**

> Embracing Our Future. Microsoft is a technology company whose mission is to empower every person and every organization on the planet to achieve more. We strive to create local opportunity, growth, and impact in every country around the world. We are creating the platforms and tools, powered by artificial intelligence, that deliver better, faster, and more effective solutions to support small and large business competitiveness, improve educational and health outcomes, grow public-sector efficiency, and empower human ingenuity.

The average filing contains roughly 50 KB of clean text, with a range from 20 KB to 800 KB depending on the company's complexity. Each filing is split into approximately 512-token chunks, which are independently embedded and then mean-pooled into a single 768-dimensional document vector.

---

## 6. How are text embeddings computed?

The embedding pipeline has three stages: chunking, encoding, and aggregation.

In the first stage, each document is split at sentence boundaries into chunks of up to 512 tokens (roughly 1,800 characters for SEC prose). This respects the context window of the encoding model while preserving sentence integrity.

In the second stage, each chunk is independently encoded by the BAAI/bge-base-en-v1.5 model, which produces a 768-dimensional vector. The model L2-normalizes each chunk embedding at output.

In the third stage, all chunk embeddings for a given document are mean-pooled into a single document vector. Importantly, this mean-pooled vector is *not* re-normalized at the document level; its L2 norm is retained as a "coherence signal," since a focused business description yields a higher norm than an incoherent one. When a firm has multiple documents (e.g., several years of 10-K filings), the document embeddings are themselves mean-pooled at the firm level, and only then is a single L2-normalization applied.

Each firm's embedding history is stored as a compressed `.npz` file containing the embedding matrix (n_docs x 768), filing dates, chunk counts, and L2 norms.

Walk-forward discipline is strictly enforced: at each rebalance date T, only documents with `doc_date <= T` contribute to the firm embedding.

---

## 7. How is the covariance matrix estimated?

The core method is **semantic shrinkage**, which blends the sample covariance estimated from returns with a structured target derived from text similarity:

> Sigma_hat = (1 - alpha) * S_sample + alpha * T_text

Here S_sample is the realized sample covariance from daily returns over the lookback window. T_text is constructed by computing the cosine similarity matrix of the firm-level embeddings and then scaling it by sample standard deviations: T = cos_sim(E) * outer(sigma, sigma). This ensures that the text-based target captures the correlation structure from text while inheriting the variance levels from returns. The parameter alpha controls the shrinkage intensity, ranging from 0 (pure sample covariance) to 1 (pure text-based covariance).

Three approaches are available for calibrating alpha. The Ledoit-Wolf oracle ("auto") minimizes Frobenius loss to an identity target and typically selects alpha around 4%; however, because the oracle is designed for identity targets, it systematically underestimates the optimal intensity for informative structured targets like the text prior. Cross-validation ("cv") minimizes the realized minimum-variance portfolio variance on held-out returns, and typically selects alpha around 75% (with a median of 100%), confirming that the text target carries substantial information. Fixed values (e.g., 0.25, 0.50, 0.75) can also be specified directly.

Positive semi-definiteness of the resulting covariance matrix is enforced by flooring all eigenvalues at 1e-4 times the largest eigenvalue.

---

## 8. How does portfolio optimization work?

The optimizer solves a convex quadratic program: minimize w'*Sigma*w subject to the constraints that all weights are non-negative (long-only), sum to one (fully invested), and do not exceed 5% per individual position. The solver is cvxpy with the OSQP backend. Optional constraints include sector concentration limits (e.g., no more than 25% in any single SIC sector) and volatility targeting (scaling weights post-optimization to match a desired portfolio volatility level).

Because the objective is purely risk-based, no return forecast is needed. The entire value of the model comes from better covariance estimation. The portfolio is rebalanced quarterly.

---

## 9. How are transaction costs modeled?

Transaction costs follow a stratified tiered model based on market capitalization. Large-cap stocks (ADV >= $500M) are assigned a base cost of 10 basis points, mid-cap stocks ($50M <= ADV < $500M) a base cost of 20 basis points, and small-cap stocks (ADV < $50M) a base cost of 50 basis points.

These base costs are further adjusted by a volatility regime multiplier. When recent realized volatility (measured over a 63-day window) exceeds the long-term average, spreads widen proportionally, up to a cap of 2x the base cost.

Costs are deducted on the first day of each holding period, immediately after rebalancing. Turnover is computed as half the sum of absolute weight changes across all positions.

---

## 10. How does the backtest work?

The backtest follows a strict walk-forward methodology in which no information from the future is used at any step.

At each quarterly rebalance date, the system first filters the investable universe using only data available up to the day before the rebalance. It then aggregates text embeddings using only documents filed on or before the rebalance date, estimates the covariance matrix using only returns observed before the rebalance date, and optimizes the portfolio weights. Between rebalance dates, portfolio weights drift naturally with daily returns, and transaction costs are applied on the first day after each rebalance.

Returns are computed using simple (not log) compounding for portfolio aggregation. Performance is evaluated using annualized Sharpe ratio (mean excess return over the risk-free rate divided by volatility), Sortino ratio (using downside deviation only), maximum drawdown (largest peak-to-trough decline), Calmar ratio (annualized return divided by maximum drawdown), and CAPM alpha with HAC (Newey-West) standard errors to account for serial correlation.

Statistical significance is assessed via block bootstrap on Sharpe ratio differences between strategies, with Benjamini-Hochberg correction for multiple comparisons.

---

## 11. What are the headline backtest results?

### Extended backtests (75 quarters, 2007-2025, p=500)

| Text source | Semantic Sharpe | LW Sharpe | EW Sharpe | Sem AnnRet | Sem Vol | Sem MaxDD |
|-------------|----------------|-----------|-----------|------------|---------|-----------|
| 10-K | 0.592 | 0.540 | 0.582 | 6.6% | 11.1% | -34.6% |
| Transcript | 0.672 | 0.663 | 0.624 | 8.5% | 13.6% | -36.8% |
| News | 0.537 | 0.507 | 0.581 | n/a | n/a | n/a |

Over the full 75-quarter window, semantic shrinkage beats both Ledoit-Wolf and equal weight on Lo-corrected Sharpe for 10-K and transcript sources. Maximum drawdown runs around 35% for the optimized strategies, compared to roughly 57% for equal weight.

### Lookback ablation headline (10-K, p=500, 75 quarters)

| Strategy | Lookback | Sharpe | vs LW-504 |
|----------|----------|--------|-----------|
| text_vols (cosine corr + 63-day sample stds) | 63 days | 0.713 | +0.173 |
| text_vols (cosine corr + 252-day sample stds) | 252 days | 0.771 | +0.231 |
| LW (full history) | 504 days | 0.540 | baseline |
| Equal weight | n/a | 0.582 | +0.042 |

The text_vols strategy, which combines text-derived correlations with short-window sample volatilities, significantly outperforms Ledoit-Wolf at L=252 (p=0.011) and L=504 (p=0.012). In other words, text plus 252 days of variance data beats Ledoit-Wolf with 504 days of full return data.

### Scaling experiment (10-K, 75 quarters, 2007--2025)

| p | Semantic Sharpe | LW Sharpe | Gap | p-value |
|---|----------------|-----------|-----|---------|
| 50 | 0.572 | 0.572 | +0.001 | 0.956 |
| 500 | 0.592 | 0.540 | +0.052 | 0.239 |
| 1,000 | 0.375 | 0.271 | +0.105 | 0.012 |
| 2,000 | 0.380 | 0.202 | +0.178 | 0.009 |

The semantic advantage grows monotonically with the number of assets. At p=1,000 and above, the gap becomes statistically significant (p=0.012 and p=0.009 respectively, surviving BH correction at 0.029). At p=2,000, semantic shrinkage delivers nearly double the Sharpe of Ledoit-Wolf (0.38 vs 0.20).

---

## 12. Is there a look-ahead bias from using a pretrained language model?

No. Three independent lines of evidence rule this out.

First, a **TF-IDF baseline** encoder, which uses zero pretrained knowledge and is refitted at each rebalance using only documents available up to that date, produces a Sharpe ratio indistinguishable from the neural model (0.592 vs 0.592, p=0.849). The signal therefore comes entirely from textual similarity structure, not from the model's training data.

Second, a **temporal sensitivity** analysis splits the sample into a pre-training period (2007-2016), a training-contemporaneous period (2017-2022), and a post-training period (2023-2025). The semantic advantage is largest in the pre-training period (gap +0.061, p=0.027) and essentially zero during the training-contemporaneous period (gap -0.006). If the pretrained model were leaking future knowledge, one would expect the advantage to peak during the period overlapping with its training data. Instead, the opposite pattern holds.

Third, **walk-forward discipline** is strictly enforced throughout. At every rebalance, only documents with `doc_date <= rebalance_date` and returns with `date < rebalance_date` are used. The universe filter relies exclusively on lagged data.

---

## 13. Why does equal weight often beat optimized strategies?

Equal weight (1/N) is a strong baseline for several reasons. It requires no estimation at all, neither a covariance matrix nor a return forecast, and therefore is immune to estimation error. It implicitly holds a diversified portfolio with a small-cap tilt, since it assigns the same weight to small and large firms alike. It also exhibits very low turnover (roughly 15% quarterly vs. roughly 50% for optimized strategies), which reduces the drag from transaction costs.

That said, equal weight comes with significantly worse tail risk. Over the 2007-2025 window, its maximum drawdown is approximately 57%, compared to roughly 35% for the optimized strategies. Under institutional constraints such as sector concentration limits or drawdown caps, semantic shrinkage beats equal weight. For instance, with a 15% sector limit, semantic shrinkage achieves a Sharpe of 0.599 vs. 0.570 for equal weight.

---

## 14. How does semantic shrinkage compare to SIC sector classification?

In cross-sectional regressions, text-based cosine similarity predicts realized return correlations roughly twice as well as SIC code matching (R-squared 6.2% vs. 3.1%). Text captures within-sector heterogeneity that SIC codes miss: for example, Moderna is classified in SIC 28 ("Chemicals") alongside Procter & Gamble, yet their business descriptions are very different. Text also captures cross-sector similarity: KLAC and LRCX sit in different SIC codes but have a cosine similarity of 0.990 and a realized return correlation of 0.903.

However, a simple SIC sector covariance baseline (block-diagonal by 2-digit SIC code) outperforms both semantic shrinkage and Ledoit-Wolf at all tested universe sizes, with a Sharpe of 0.720 vs. 0.592 for semantic at p=500. This is an important caveat: the industry classification prior, despite being coarser, is empirically competitive.

---

## 15. What is the correlation between text similarity and return co-movement?

| Text source | Avg Spearman rho | % quarters significant | N quarters |
|-------------|-----------------|----------------------|------------|
| 10-K filings | 0.104 | 100% | 70 |
| Financial news | 0.254 | 100% | 70 |

This is measured as the rank correlation between pairwise cosine similarity of firm embeddings and pairwise realized return correlation over the subsequent year. Every single quarter shows a statistically significant positive relationship (p < 0.001). News embeddings yield more than double the correlation strength of 10-K embeddings, likely because news articles capture more timely, market-relevant information about firms' relationships.

---

## 16. What software and hardware are used?

The project is written in Python 3.13 and managed with **uv** as the package manager. Text embeddings are computed using the BAAI/bge-base-en-v1.5 model via the sentence-transformers library. Portfolio optimization relies on cvxpy with the OSQP solver backend. Data manipulation uses pandas, numpy, and pyarrow, while statistical inference uses scipy and statsmodels (for HAC standard errors). Text and price data are sourced from HuggingFace datasets (filings, transcripts, news) and the Tiingo API (prices), respectively. GPU acceleration (via PyTorch CUDA) is used for the covariance estimation bottlenecks (eigendecomposition and Ledoit-Wolf shrinkage), providing ~800x speedup at p=2000. Embedding inference uses a standard CPU.
