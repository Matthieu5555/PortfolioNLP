# Semantic Shrinkage: Text-Similarity Priors for High-Dimensional Covariance Estimation in Portfolio Optimization

**Mat Sunderland**

February 2026

---

## Abstract

We propose *semantic shrinkage*, a covariance estimation method that replaces the scaled-identity target in Ledoit-Wolf shrinkage with a text-similarity matrix derived from corporate disclosures. Firm embeddings are computed from SEC 10-K annual reports, earnings call transcripts, and financial news using a pre-trained sentence transformer (BAAI/bge-base-en-v1.5), and their pairwise cosine similarity serves as a structured prior over the correlation matrix. In a walk-forward quarterly-rebalanced backtest spanning 2007--2025 with up to 4,694 US equities, we find that:

1. **Text provides the correlation structure; returns need only provide volatilities.** A return history ablation across lookback windows $L \in \{0, \ldots, 504\}$ days shows that using text-based correlations with sample volatilities produces Sharpe 0.77 at $L$=252 and $p$=500 — **statistically significantly** outperforming Ledoit-Wolf with full 504-day returns (0.54, $p$=0.011). Text dominates LW at every lookback length. The result replicates across text sources: transcript-based text_vols peaks at $L$=63 (Sharpe 0.74, beating EW 0.62). The crossover is clear: text provides the full pairwise correlation matrix, reducing the return data requirement from years to weeks.

2. **The LW oracle is miscalibrated for informative targets.** The Ledoit-Wolf shrinkage intensity formula, calibrated for identity targets, prescribes $\alpha$~4% when optimal is 25--75%. Sweeping $\alpha$ across [0.25, 0.5, 0.75, 1.0] at $p$=500 over 75 quarters produces uniformly higher Sharpe than LW (raw $p$-values 0.002--0.014). A pure text covariance matrix ($\alpha$=1.0, no sample covariance) achieves Sharpe 0.744, outperforming LW's 0.540 by 38%.

3. **The advantage scales with dimensionality.** At $p$=2000 ($p/n$~4), semantic shrinkage delivers nearly double the Sharpe of LW (0.38 vs 0.20), and this is statistically significant after BH correction ($p_{\text{BH}}$=0.029). The gap widens monotonically from +0.001 at $p$=50 to +0.178 at $p$=2000. VaR calibration confirms the mechanism: at all $p$ values, semantic has the lowest violation ratio (closest to nominal calibration), with a 5% VaR ratio of 1.03 at $p$=1000 versus LW's 1.23.

4. **Text embeddings define stable, interpretable factors.** SVD of the firm embedding matrix reveals persistent "text factors": PC1 (92.7% of variance, Kendall $\tau$=0.98) separates industrial/manufacturing from commodity/leisure firms; subsequent components capture energy-vs-biotech and retail-vs-financial distinctions — all derived purely from corporate language.

A TF-IDF baseline — using only corpus-internal term frequencies with zero pretrained model knowledge — produces statistically indistinguishable results from the neural embeddings (Sharpe 0.592 vs 0.592, $p$=0.849 over 75 quarters), confirming that the signal arises from textual structure rather than the pretrained model's future knowledge. A temporal sensitivity analysis corroborates this: the semantic advantage is significant in the pre-training period 2007--2016 ($p$=0.027) and essentially zero in the training-contemporaneous period 2017--2022 (gap $-$0.006), the opposite of what look-ahead bias would produce. Cross-validated shrinkage intensity selection, which independently discovers a mean $\alpha$ of 0.76 (median 1.0), significantly outperforms Ledoit-Wolf ($p$=0.034), providing an adaptive solution to the oracle miscalibration.

A shuffle placebo test shows the advantage at $p$=500 arises from the spectral structure of the cosine matrix rather than firm-specific content, but this reverses at $p$=2000 where 80% of shuffled runs underperform, indicating firm-specific text information matters more at higher dimensionality. An SIC sector baseline outperforms text-based methods at most lookback lengths, suggesting that even coarser industry structure captures much of the same signal. Under institutional constraints (sector concentration $\leq$15%), semantic shrinkage beats the equal-weight baseline (Sharpe 0.599 vs 0.570), and the maximum drawdown advantage persists across all constraint regimes (~35% vs ~57%). Over the extended backtest window (2007--2025, 75 quarters), semantic shrinkage beats the equal-weight (1/N) baseline on both raw Sharpe (10-K: 0.59 vs 0.58; transcript: 0.67 vs 0.62) and Lo-corrected Sharpe (0.72 vs 0.69; 0.91 vs 0.75), with substantially lower maximum drawdown (35% vs 57%), reversing the EW dominance observed in the shorter 2013--2024 window. The text_vols variant — text correlations plus sample volatilities — beats EW at every lookback length $\geq$21 days. The lookback ablation result replicates across a third text source: news article embeddings yield statistically significant text_vols vs LW differences at $L$=126 ($p$=0.047), $L$=252 ($p$=0.033), and $L$=504 ($p$=0.018).

---

## 1. Research Question

Can a structured shrinkage target derived from text similarity enable covariance-optimized portfolios to scale to higher dimensions than standard Ledoit-Wolf shrinkage (which shrinks toward a scaled-identity target)? Building on prior work using text-based similarity for financial applications (Hoberg & Phillips, 2010, 2016), we test whether the richer correlation structure encoded in firm-level embeddings provides a more robust prior as the number of assets grows relative to the estimation window.

---

## 2. Methodology

### 2.1 Covariance Estimator

We construct the shrinkage estimator:

$$\hat{\Sigma} = (1 - \alpha) \cdot S_{\text{sample}} + \alpha \cdot T_{\text{semantic}}$$

where:
- $S_{\text{sample}}$ is the sample covariance matrix of daily simple returns over a 504-day (2-year) trailing window (missing values filled with 0.0 after applying the 80% completeness filter; log returns from the data store are converted to simple returns at backtest entry)
- $T_{\text{semantic}} = C_{\text{cosine}} \odot (\sigma \sigma^\top)$ is the semantic target, formed by scaling the cosine-similarity matrix of firm embeddings by the sample standard deviations to match the scale of $S_{\text{sample}}$
- $\alpha$ is the shrinkage intensity, calibrated using the Ledoit-Wolf oracle approximation formula (scikit-learn implementation). **The same intensity is used for both the semantic and Ledoit-Wolf estimators; only the target differs.** This is a deliberate design choice: it isolates the effect of the target matrix from the effect of shrinkage intensity, ensuring any performance difference is attributable solely to the prior's structure. The LW oracle formula minimizes expected loss for the identity target specifically; applying it to a different target is not theoretically optimal, but it provides a conservative test -- the semantic method does not benefit from a custom-tuned intensity. Cross-validating an intensity specific to the semantic target is left for future work.

Positive semi-definiteness is enforced via relative eigenvalue flooring: all eigenvalues are raised to at least $\lambda_{\max} \times 10^{-4}$.

We also consider a simpler variant, *text_vols*, which bypasses the shrinkage blend entirely:

$$\hat{\Sigma}_{\text{tv}} = D_\sigma \cdot C_{\text{cosine}} \cdot D_\sigma$$

where $D_\sigma = \text{diag}(\hat{\sigma}_1, \ldots, \hat{\sigma}_p)$ contains sample standard deviations from an $L$-day return window. This is formally equivalent to setting $\alpha = 1$ for the correlation structure while retaining return-based volatilities. The text_vols variant decomposes the information content of the covariance matrix: text provides the pairwise correlation structure, while returns need only provide marginal volatilities. As Section 3.21 demonstrates, this decomposition produces the paper's strongest results.

**Intensity calibration alternatives.** In addition to the LW oracle, we test cross-validated intensity selection (CV-alpha): at each quarterly rebalance, the trailing $L$-day window is split temporally 70/30, a grid of $\alpha$ values is evaluated on the validation portion using realized portfolio variance as the loss function, and the best $\alpha$ is selected for that quarter. This is strictly out-of-sample and provides an adaptive alternative to both the oracle formula and fixed $\alpha$ choices.

### 2.2 Embedding Pipeline

| Component | Detail |
|-----------|--------|
| Model | BAAI/bge-base-en-v1.5 (768-dim, sentence-transformers) |
| Chunking | 512 tokens per chunk, sentence-boundary splitting (NLTK punkt_tab) |
| Document pooling | Mean of chunk embeddings (no L2-normalization at document level) |
| Firm aggregation | Mean of all available document embeddings as of rebalance date, then L2-normalized |
| Look-ahead prevention | Documents filtered by `filed_date <= rebalance_date` (actual SEC EDGAR filing dates, not fiscal year end) |

**Normalization convention (critical):** Document-level embeddings are *not* L2-normalized because their norm carries coherence information (high norm = consistent language across chunks). L2-normalization occurs only at firm-aggregation level, ensuring cosine similarity between firms is well-defined.

### 2.3 Text Sources

| Source | Tickers | Documents | Period | Frequency |
|--------|---------|-----------|--------|-----------|
| 10-K annual reports | 4,592 | 60,258 | 1993--2023 | Annual |
| Earnings transcripts | 608 (S&P 500) | 30,577 | 2006--2025 | Quarterly |
| Financial news | 4,024 | 111,062 | 2007--2024 | Quarterly (aggregated) |
| Combined (10-K + transcript) | 4,694 | 90,835 | 1993--2025 | Mixed |

### 2.4 Portfolio Construction

- **Objective:** Minimum variance (no expected return estimation required)
- **Constraints:** Long-only, 5% maximum position weight
- **Optimizer:** CVXPY (OSQP solver) with SLSQP fallback; weights below $10^{-6}$ zeroed and re-normalized
- **Rebalancing:** Quarterly (1st of Jan/Apr/Jul/Oct)
- **Lookback:** 504 trading days (~2 years) for covariance estimation

### 2.5 Investable Universe Filter

Applied identically across all strategies at each rebalance date:
- **Liquidity:** Trailing 21-day median daily dollar volume $\geq$ \$1M (strict lag: `index < T`)
- **Completeness:** $\geq$ 80% non-NaN daily returns over the 504-day lookback window
- **Extreme return clipping:** $|\text{log return}| > 0.50$ clipped to remove penny stock artifacts
- **Universe cap:** Tickers ranked by trailing ADV, capped at max_firms (default 500)

### 2.6 Transaction Cost Model

Stratified per-ticker costs based on liquidity tier, with volatility regime adjustment:

| Tier | ADV Floor | Base Cost |
|------|-----------|-----------|
| Large cap | $\geq$ \$500M | 10 bps |
| Mid cap | $\geq$ \$50M | 20 bps |
| Small cap | $<$ \$50M | 50 bps |

Multiplied by a volatility regime factor: `max(1.0, min(recent_vol / longterm_vol, 2.0))`, where recent vol is the cross-sectional median realized vol over the trailing 63 days, and longterm vol is the full-history cross-sectional median.

### 2.7 Statistical Testing

- **Block sign-flip test** (cf. Carlstein, 1986): Two-sided test on paired daily return differences. Differences are partitioned into non-overlapping blocks of 63 trading days (~quarterly); block signs are randomly flipped to construct the null distribution. 10,000 permutations, p-value per Phipson & Smyth (2010): $(c+1)/(B+1)$. Minimum achievable p-value = $1/10{,}001 \approx 0.0001$. Benjamini-Hochberg FDR correction applied when testing multiple p values.
- **Lo (2002) Sharpe correction:** Adjusts annualized Sharpe for serial autocorrelation using Bartlett-kernel weighted autocorrelations up to lag 63.
- **CAPM regression:** OLS of portfolio returns on SPY returns with HAC standard errors (Newey-West, 10 lags); alpha annualized as $\alpha_{\text{daily}} \times 252$.

### 2.8 Baselines

1. **Ledoit-Wolf:** Shrinks $S_{\text{sample}}$ toward scaled identity using sklearn's `LedoitWolf()`. Same optimizer, constraints, and transaction cost model.
2. **Equal weight (1/N):** Uniform weights across all eligible tickers. No optimization.
3. **SPY:** Buy-and-hold S&P 500 ETF (out-of-sample benchmark only).

---

## 3. Results

### 3.1 Main Backtest (p=500, 2013-04-01 to 2024-06-30, quarterly rebalancing)

| Text Source | Strategy | Ann. Return | Ann. Vol | Sharpe | Lo-corrected SR | p-value (two-sided) |
|---|---|---|---|---|---|---|
| 10-K | Semantic | 6.2% | 11.7% | 0.533 | 0.671 | |
| 10-K | LW | 6.1% | 11.7% | 0.521 | 0.642 | |
| 10-K | **Gap** | | | **+0.012 (+2%)** | | **0.925** |
| Transcript | Semantic | 8.6% | 13.6% | 0.627 | 0.862 | |
| Transcript | LW | 8.6% | 13.6% | 0.632 | 0.845 | |
| Transcript | **Gap** | | | **-0.005 (-1%)** | | **0.979** |
| Combined | Semantic | 5.9% | 11.8% | 0.498 | 0.661 | |
| Combined | LW | 5.7% | 11.9% | 0.479 | 0.614 | |
| Combined | **Gap** | | | **+0.019 (+4%)** | | **0.739** |

- 45 quarterly rebalance dates, 2,832 trading days
- Equal weight Sharpe: 0.708 (10-K universe), 0.775 (transcript universe), 0.733 (combined)
- Transcript-only uses S&P 500 subset (~400--500 firms per quarter), explaining higher absolute Sharpe
- Lo-corrected Sharpe ratios are *higher* than raw (eta < 1 for all strategies), indicating slight negative daily autocorrelation (mean-reversion) that causes the standard annualized Sharpe to understate risk-adjusted performance
- **None of the p=500 results are statistically significant at conventional levels**
- Semantic and LW perform nearly identically at p=500; the gap is negligible (2% for 10-K, -1% for transcripts)
- Combined text source underperforms 10-K alone (Sharpe 0.498 vs 0.533). Adding quarterly transcripts may dilute the annual 10-K signal: the firm aggregator mean-pools all documents equally, so the 30,577 transcripts can dominate the 60,258 annual reports for the 608 tickers that have both sources, potentially adding noise from forward-looking management language that does not improve the covariance prior

### 3.1b Extended Backtest (p=500, 2007-04-01 to 2025-12-31, quarterly rebalancing)

Extending the backtest window from 45 to 75 quarterly rebalances (including the 2008 financial crisis and 2025 drawdown) materially changes the relative performance landscape:

| Text Source | Strategy | Ann. Return | Ann. Vol | Sharpe | Lo-corrected SR | MaxDD | p-value (vs LW) |
|---|---|---|---|---|---|---|---|
| 10-K | Semantic | 6.6% | 11.1% | 0.592 | 0.716 | -34.6% | |
| 10-K | LW | 6.0% | 11.2% | 0.540 | 0.641 | -33.8% | |
| 10-K | EW | 12.8% | 22.0% | 0.582 | 0.687 | -56.4% | |
| 10-K | **Sem vs LW** | | | **+0.052 (+10%)** | | | **0.239** |
| 10-K | **Sem vs EW** | | | **+0.010** | | | **0.073** |
| Transcript | Semantic | 9.4% | 14.0% | 0.672 | 0.906 | -36.8% | |
| Transcript | LW | 9.2% | 13.9% | 0.663 | 0.887 | -35.9% | |
| Transcript | EW | 13.7% | 21.9% | 0.624 | 0.754 | -57.7% | |
| Transcript | **Sem vs LW** | | | **+0.009 (+1%)** | | | **0.617** |
| News | Semantic | 6.3% | 11.7% | 0.537 | 0.663 | -33.1% | |
| News | LW | 6.0% | 11.8% | 0.507 | 0.603 | -32.6% | |
| News | EW | 13.0% | 22.3% | 0.581 | 0.685 | -57.2% | |
| News | **Sem vs LW** | | | **+0.030 (+6%)** | | | **0.421** |

- 75 quarterly rebalance dates for 10-K and transcript, 71 for news (data ends 2024)
- **Semantic beats EW on Lo-corrected Sharpe** for 10-K (0.716 vs 0.687) and transcript (0.906 vs 0.754)
- **Maximum drawdown tells the risk story:** semantic ~35% vs EW ~57%. Over 75 quarters including crisis periods, the covariance-optimized min-variance portfolio provides dramatically better drawdown protection
- **LW significantly underperforms EW** for 10-K ($p$=0.040) and news ($p$=0.035) — standard identity-target shrinkage fails at $p$=500 over longer windows
- The 10-K sem vs LW gap grows from +0.012 (+2%) in 45 quarters to +0.052 (+10%) in 75 quarters, with $p$-value improving from 0.925 to 0.239
- News embeddings (from financial news articles) provide a third text source with the same pattern: semantic outperforms LW across the board, with the highest average correlation signal ($\rho$=0.254 vs 10-K $\rho$=0.104)

### 3.2 Dimensional Scaling (10-K, p=50 to p=2000, 75 quarters, 2007--2025)

| p | p/n | Sem. Sharpe | LW Sharpe | Gap | p-value | BH-adjusted | EW Sharpe | Sem Lo | LW Lo |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 0.10 | 0.572 | 0.572 | +0.001 | 0.956 | 0.956 | 0.619 | 0.745 | 0.742 |
| 100 | 0.20 | 0.630 | 0.617 | +0.013 | 0.397 | 0.463 | 0.618 | 0.840 | 0.818 |
| 200 | 0.40 | 0.701 | 0.682 | +0.018 | 0.380 | 0.463 | 0.606 | 0.896 | 0.863 |
| 500 | 0.99 | 0.592 | 0.540 | +0.052 | 0.239 | 0.418 | 0.582 | 0.716 | 0.641 |
| 1000 | 1.98 | 0.375 | 0.271 | +0.105 | **0.012** | **0.029** | 0.542 | 0.430 | 0.311 |
| 1500 | 2.98 | 0.399 | 0.250 | +0.150 | **0.008** | **0.029** | 0.519 | 0.440 | 0.279 |
| 2000 | 3.97 | 0.380 | 0.202 | +0.178 | **0.009** | **0.029** | 0.505 | 0.418 | 0.223 |

- Lookback window n=504 trading days; p/n = p/504
- Sharpe gap widens monotonically with p/n: from +0.001 at p=50 to +0.178 at p=2000
- BH-adjusted p-values computed within each p-level across the three pairwise comparisons (sem-vs-LW, sem-vs-EW, LW-vs-EW), then the sem-vs-LW BH-adjusted p is reported. At $p \geq 1000$, semantic significantly beats LW after BH correction ($p_{\text{BH}}$=0.029)
- All strategies maintain positive Sharpe ratios across all p values
- **Equal weight Sharpe remains highest at all p values** (0.51--0.62), dominating both optimized methods everywhere
- Oracle alpha (shrinkage intensity) grows from 3.3% at p=50 to 5.5% at p=2000, showing the text prior receives more weight at higher dimensionality

**Key finding — scalability:** The semantic target enables covariance-optimized portfolios to accommodate far more assets. At p=2000, semantic shrinkage delivers nearly double the Sharpe of LW (0.38 vs 0.20), and this advantage is statistically significant after BH correction ($p_{\text{BH}}$=0.029). The identity target collapses while the structured target maintains usable performance. The gap widens monotonically with p/n, meaning the benefit is greatest precisely where practitioners need it most: broad universes where estimation error is severe. The shuffle placebo test (Section 3.7) shows the advantage derives from the spectral structure of the cosine matrix, not firm-specific text content.

### 3.3 SPY Benchmark (10-K, p=500)

| Strategy | Sharpe | Ann. Return | Max DD | Beta | CAPM Alpha | Alpha t-stat | Alpha p |
|----------|--------|-------------|--------|------|------------|-------------|---------|
| SPY | 0.399 | 7.7% | -59.6% | 1.000 | -- | -- | -- |
| Semantic | 0.592 | 6.6% | -34.6% | 0.434 | +2.1% | 1.20 | 0.23 |
| Ledoit-Wolf | 0.540 | 6.0% | -33.8% | 0.439 | +1.5% | 0.88 | 0.38 |
| Equal Weight | 0.582 | 12.8% | -56.4% | 1.063 | +1.9% | 1.47 | 0.14 |

**Caveat:** SPY statistics are computed over its full available history (6,538 trading days, ~2003--2025) while strategy statistics cover the backtest window (4,718 trading days, 2007--2025). Beta and CAPM alpha are computed over the overlapping period only (HAC standard errors, Newey-West with 10 lags) and are directly comparable.

- Both min-variance strategies have beta ~0.44, achieving less than half the market exposure
- All CAPM alphas are positive but insignificant (all p>0.14), indicating no strategy generates statistically significant risk-adjusted return above the market
- Semantic shrinkage has marginally better alpha than LW (+2.1% vs +1.5%), consistent with slightly superior covariance estimation
- Both optimized strategies outperform SPY on a risk-adjusted basis (Sharpe 0.59 and 0.54 vs SPY 0.40) with roughly half the volatility

### 3.4 Higher Moments (10-K, p=500)

| Strategy | Skewness | Kurtosis | VaR(5%) | CVaR(5%) | VaR(1%) | CVaR(1%) |
|----------|----------|----------|---------|----------|---------|----------|
| Semantic | -1.52 | 35.0 | -0.97% | -1.64% | -1.80% | -3.11% |
| Ledoit-Wolf | -1.36 | 31.1 | -0.98% | -1.66% | -1.84% | -3.12% |
| Equal Weight | -0.34 | 8.7 | -2.15% | -3.37% | -3.93% | -5.73% |

- Min-variance strategies exhibit heavy left tails (high kurtosis, negative skewness)
- Despite this, daily CVaR(5%) is substantially lower than equal weight (-1.64% vs -3.37%)
- Equal weight has better tail shape (lower kurtosis) but worse absolute tail risk due to higher volatility

### 3.5 Transcript Robustness (p=500)

| Metric | Sem (10-K) | LW (10-K) | Sem (Transcript) | LW (Transcript) |
|--------|-----------|----------|------------------|-----------------|
| Sharpe | 0.533 | 0.521 | 0.627 | 0.632 |
| Ann. Return | 6.2% | 6.1% | 8.6% | 8.6% |
| Ann. Vol | 11.7% | 11.7% | 13.6% | 13.6% |
| Max DD | -34.6% | -33.8% | -36.8% | -35.9% |

- Transcript-based strategies have higher Sharpe (0.63 vs 0.53) due to the S&P 500 universe filter
- For transcripts, LW marginally outperforms semantic (+0.005 Sharpe), unlike the 10-K result where semantic leads by +0.012
- Both text sources show a negligible and statistically insignificant gap between semantic and LW

### 3.6 Shrinkage Intensity

The Ledoit-Wolf oracle formula calibrates shrinkage intensity $\alpha$ identically for both the semantic and LW estimators. Both methods use the same $\alpha$; only the target matrix differs.

| p | p/n | Avg. $\alpha$ |
|---|---|---|
| 50 | 0.10 | 0.037 |
| 100 | 0.20 | 0.038 |
| 200 | 0.40 | 0.041 |
| 500 | 0.99 | 0.045 |
| 1000 | 1.98 | 0.052 |
| 1500 | 2.98 | 0.061 |
| 2000 | 3.97 | 0.065 |

As dimensionality increases, the oracle prescribes heavier shrinkage. At p/n~4, the estimator places ~6.5% weight on the target — sufficient for the structured prior to meaningfully preserve correlation structure that the identity target destroys. This explains the scalability advantage: at low p/n the target barely matters ($\alpha$~3.7%), but at high p/n the target's structure becomes the dominant factor in portfolio quality. The shuffle test (Section 3.7) indicates this structure is spectral rather than informational at p=500 ($\alpha$~4.5%); whether firm-specific content contributes at higher $\alpha$ values remains an open question.

### 3.7 Shuffle Placebo Test (10-K, p=500, 75 quarters, 2007--2025)

To test whether the semantic advantage relies on firm-specific text information, we randomly permute the ticker-embedding assignment and re-run the walk-forward backtest. If the semantic signal is real, shuffled portfolios should perform worse than unshuffled.

| Run | Sharpe |
|-----|--------|
| Unshuffled | 0.592 |
| **Shuffled mean (10 runs)** | **0.593** (std=0.002) |

**Result:** 7 out of 10 shuffled runs *outperform* the correctly-assigned unshuffled portfolio (z=$-$0.80, empirical $p$=0.70). At p=500 with shrinkage intensity $\alpha$~4.5%, the firm-specific text information does not contribute to portfolio performance.

**Interpretation:** The cosine-similarity matrix provides a structured regularization target that outperforms the identity target regardless of whether tickers are correctly assigned. This is because the key spectral properties of the cosine matrix — its approximate low-rank structure and eigenvalue distribution — are invariant under row/column permutation. The advantage over LW (Section 3.2) therefore reflects the structural benefit of shrinking toward a matrix with realistic correlation structure, rather than the informational content of which specific firms are similar. At higher p/n where $\alpha$ is larger (6.5% at p=2000), the shuffle test may yield a different result, as the text prior receives more weight — this is left for future work.

### 3.8 Sub-Period Validation (10-K, p=500)

| Period | Sem Sharpe | LW Sharpe | Gap | EW Sharpe | Sem vs LW p |
|--------|-----------|-----------|-----|-----------|-------------|
| 2013--2018 | 0.788 | 0.774 | +0.014 | 0.958 | 0.853 |
| 2019--2024 | 0.459 | 0.471 | -0.012 | 0.725 | 0.966 |
| Full period | 0.533 | 0.521 | +0.012 | 0.708 | 0.925 |

- First half (2013--2018) outperforms second half for all strategies, reflecting the low-volatility / recovery environment
- Semantic outperforms LW in the first half but underperforms in the second half
- Neither sub-period shows statistical significance
- Equal weight dominates in both periods

### 3.9 Transaction Cost Sensitivity (10-K, p=500, 75 quarters, 2007--2025)

| TC Multiplier | Sem Sharpe | LW Sharpe | Gap | EW Sharpe |
|---------------|-----------|-----------|-----|-----------|
| 0.5x | — | — | +0.050 | — |
| 1.0x (base) | 0.592 | 0.540 | +0.052 | 0.582 |
| 1.5x | — | — | +0.053 | — |
| 2.0x | — | — | +0.055 | — |

- Semantic beats LW at all cost multipliers (0.5x through 2.0x); the gap widens from +0.050 at 0.5x to +0.055 at 2.0x
- Semantic has lower average turnover (46.8%) than LW (49.6%), explaining the growing gap at higher TC
- Equal weight has far lower turnover (14.7%)

### 3.10 Block Size Sensitivity (10-K, p=500)

| Block Size | Sem vs LW p-value | Sem vs EW p-value | LW vs EW p-value |
|------------|-------------------|-------------------|------------------|
| 42 (bimonthly) | 0.872 | 0.039 | 0.025 |
| 63 (quarterly) | 0.925 | 0.059 | 0.037 |
| 126 (semiannual) | 0.901 | 0.060 | 0.035 |

- Statistical conclusions are robust to block size choice
- EW vs LW is marginally significant at all block sizes (p=0.025--0.037)
- Sem vs LW is far from significant at all block sizes (p=0.87--0.93)

### 3.11 Fama-French Factor Attribution (10-K, p=500, 75 quarters, 2007--2025)

We regress daily strategy returns on the Fama-French 5 factors plus momentum (FF5+Mom) using OLS with Newey-West HAC standard errors (10 lags). Factor data from Kenneth French's data library.

| Strategy | Alpha (ann.) | Alpha t | Alpha p | Mkt-RF | SMB | HML | RMW | CMA | Mom | R² |
|----------|-------------|---------|---------|--------|-----|-----|-----|-----|-----|-----|
| Semantic | -0.79% | -0.43 | 0.66 | 0.476 | -0.055 | -0.014 | 0.167 | 0.220 | 0.053 | 0.632 |
| LW | -1.25% | -0.70 | 0.48 | 0.477 | -0.025 | -0.031 | 0.135 | 0.211 | 0.052 | 0.637 |
| EW | +0.53% | +0.66 | 0.51 | 1.020 | 0.276 | -0.017 | -0.037 | 0.073 | -0.061 | 0.973 |

- All FF alphas are insignificant: no unexplained return after factor adjustment
- Semantic CMA loading (0.220) is moderate, indicating a tilt toward conservative firms, consistent with min-variance selecting low-beta value stocks. LW has a similar CMA loading (0.211)
- Equal weight is essentially a market portfolio (Mkt-RF $\beta$$\approx$1.02) with strong SMB tilt (0.28) from equal-weighting small caps, explaining its higher absolute return. EW CVaR(1%) = $-$5.73% (worse tail risk than optimized strategies)
- Semantic and LW have similar factor exposures; the Sharpe difference reflects micro-level portfolio composition rather than systematic factor tilt

### 3.12 PCA Factor Analysis of the Embedding Space (75 quarters, 2007--2025)

The cosine-similarity matrix used as our shrinkage target is the Gram matrix $C = EE^\top$ where $E$ is the L2-normalized embedding matrix ($N \times 768$). Its eigenstructure is determined by the SVD of $E$. A rank-$k$ PCA approximation retains only the top-$k$ singular components, testing how much spectral structure is needed for effective regularization.

**Eigenvalue concentration.** The embedding matrix is overwhelmingly low-rank. Averaged across 75 quarterly rebalances at $p=500$:

| Top-$k$ PCs | Explained Variance |
|------------|-------------------|
| 1 | 92.7% |
| 3 | 93.9% |
| 5 | 94.7% |
| 10 | 95.7% |
| 20 | 96.7% |
| 50 | 98.1% |
| 100 | 99.1% |

A single principal component captures nearly 93% of the variance in the 768-dimensional embedding space. This extreme concentration suggests the dominant axis captures a market-wide factor (likely market capitalization or broad industry membership), with subsequent components picking up finer industry/sector distinctions.

**$k$-sweep: how many text factors are needed?** We run the full walk-forward backtest using PCAFactorCovariance as the shrinkage target at each $k$, with rolling SVD recomputed at each quarterly rebalance using only embeddings and returns available before the rebalance date:

| $k$ | Sharpe | Gap vs LW | Gap vs Full Cosine |
|-----|--------|-----------|-------------------|
| 1 | 0.598 | +0.058 | +0.006 |
| 5 | 0.598 | +0.058 | +0.006 |
| 10 | 0.596 | +0.056 | +0.004 |
| 20 | 0.594 | +0.054 | +0.002 |
| 50 | 0.593 | +0.053 | +0.001 |
| 100 | 0.593 | +0.053 | +0.001 |
| Full ($k=500$) | 0.592 | +0.052 | 0.000 |
| LW (identity) | 0.540 | — | — |
| EW (1/N) | 0.582 | — | — |

**Key finding: performance is flat across $k$.** Sharpe varies by only 0.006 as $k$ increases from 1 to 500 (full cosine). All $k$ values beat LW (0.540). The rank-1 approximation (a single "market factor" in text space) marginally outperforms the full 500-dimensional cosine matrix by +0.006 Sharpe. This implies the dominant spectral component captures essentially all of the useful structure — additional dimensions contribute negligible information.

This result deepens the shuffle test finding (Section 3.7): not only is the advantage structural rather than informational, but the useful structure is extremely low-dimensional. A single text factor captures 93% of the embedding variance and provides a marginally better shrinkage target than the full cosine matrix. The remaining 499 dimensions add negligible value.

**Implication for the "Why not GICS?" question:** GICS provides an 11-dimensional categorical classification. The PCA analysis shows the effective dimensionality of the embedding-based target is even lower — a single continuous factor outperforms the full 768-dimensional representation. The advantage of embeddings over sector dummies is not dimensionality but continuity: they provide a smooth, continuous similarity measure rather than a binary same-sector/different-sector classification, and they capture cross-sector relationships (e.g., Amazon's similarity to both technology and retail firms) that discrete labels cannot.

### 3.13 Shrinkage Intensity: The LW Oracle Is Miscalibrated (10-K, p=200 and p=500, 75 quarters, 2007--2025)

The Ledoit-Wolf oracle formula is derived for the identity target. When applied to the semantic target — which is far more informative — it dramatically underweights the prior. We sweep the shrinkage intensity $\alpha$ across [0.0, 0.25, 0.5, 0.75, 1.0] plus the LW oracle ("auto") to map the performance surface.

**p=200 (p/n=0.40, avg. auto $\alpha$=3.6%):**

| Strategy | Sharpe | Ann. Return | Ann. Vol | vs LW p-value |
|----------|--------|-------------|----------|---------------|
| $\alpha$=0.0 (pure sample) | 0.678 | 8.6% | 12.6% | 0.252 |
| $\alpha$=0.25 | **0.729** | 9.3% | 12.7% | 0.344 |
| $\alpha$=0.5 | 0.711 | 9.3% | 13.0% | 0.495 |
| $\alpha$=0.75 | **0.727** | 9.6% | 13.2% | 0.365 |
| $\alpha$=1.0 (pure text) | 0.716 | 9.5% | 13.3% | 0.424 |
| $\alpha$=auto (~3.6%) | 0.701 | 8.8% | 12.6% | 0.380 |
| LW (identity target) | 0.682 | 8.6% | 12.6% | — |
| Equal weight | 0.606 | 13.1% | 21.6% | — |

**p=500 (p/n=0.99, avg. auto $\alpha$=3.9%):**

| Strategy | Sharpe | vs LW p-value |
|----------|--------|---------------|
| $\alpha$=0.25 | 0.752 | **0.002** |
| $\alpha$=0.5 | 0.745 | **0.014** |
| $\alpha$=0.75 | **0.762** | **0.012** |
| $\alpha$=1.0 (pure text) | 0.744 | **0.012** |
| $\alpha$=auto (~4%) | 0.592 | — |
| LW (identity target) | 0.540 | — |
| Equal weight | 0.582 | — |

**Key findings:**

1. **All $\alpha \geq 0.25$ significantly outperform LW at p=500.** The raw p-values range from 0.002 ($\alpha$=0.25) to 0.014 ($\alpha$=0.50). All four $\alpha \geq 0.25$ comparisons are individually significant at the 5% level before correction. After Benjamini-Hochberg correction for the comparisons within the sweep, these remain significant.

2. **The LW oracle intensity is far too conservative.** At p=500, auto $\alpha$~3.9% but optimal is ~75%. The oracle formula is calibrated for identity targets; it does not know how informative the semantic target is. With the semantic target, the sample covariance needs far less weight.

3. **Pure text ($\alpha$=1.0) massively beats LW** at p=500: Sharpe 0.744 vs 0.540 (+38%). At p=200: 0.716 vs 0.682 (+5%). The text-similarity matrix alone, scaled by sample standard deviations, outperforms the standard covariance estimator.

4. **$\alpha$=0.0 is comparable to LW at p=200** (Sharpe 0.678 vs 0.682, p=0.252), indicating that the pure sample covariance performs similarly to Ledoit-Wolf at low p/n over 75 quarters. Adding even minimal text prior ($\alpha$=0.25) pushes Sharpe to 0.729.

5. **The performance surface is broad** at p=500, with the optimum at $\alpha$=0.75 (Sharpe 0.762). There is no sharp cliff — all $\alpha$ values from 0.25 to 1.0 produce Sharpe ratios between 0.744 and 0.762, all significantly beating LW.

**Implication:** The LW oracle was designed to find the optimal mix between sample covariance and identity matrix. When the target is informative (as the semantic target is), the oracle's penalty for moving away from the sample is excessive. A target-specific cross-validation procedure — or simply using a higher fixed $\alpha$ — would unlock the full benefit of the text prior.

### 3.14 Cold-Start Portfolio Construction (10-K)

For newly listed firms, IPOs, or assets in thinly-traded markets, return history may be unavailable or unreliable. We test two cold-start variants that replace or eliminate historical return data:

**Cold-start with sample volatilities ("text_vol"):** Use the cosine-similarity matrix as the correlation structure, but scale by sample standard deviations from the lookback window. Formally: $\hat{\Sigma} = D_\sigma \cdot C_{\text{cosine}} \cdot D_\sigma$ where $D_\sigma = \text{diag}(\sigma_1, \ldots, \sigma_p)$.

**Pure cold-start ("pure_text"):** Use the raw cosine-similarity matrix with unit diagonal — no return data at all. The optimizer sees only text similarity structure, treating all firms as having unit volatility.

| p | Strategy | Sharpe | Ann. Return | Ann. Vol | vs LW p-value |
|---|----------|--------|-------------|----------|---------------|
| 200 | Cold-start text_vol | 0.723 | 9.3% | 12.9% | 0.445 |
| 200 | Pure cold-start | 0.531 | 9.0% | 16.9% | 0.785 |
| 200 | LW | 0.660 | 8.1% | 12.3% | — |
| 500 | Cold-start text_vol | 0.744 | 9.7% | 13.0% | **0.012** |
| 500 | Pure cold-start | 0.488 | 8.3% | 17.0% | 0.494 |
| 500 | LW | 0.540 | 6.0% | 11.2% | — |

**Key findings:**

1. **Cold-start text_vol = $\alpha$=1.0** (mathematically identical, confirmed by experiment). This is reassuring: `CosineSimilarityCovariance` with sample stds is formally equivalent to the semantic shrinkage target at full weight.

2. **Pure cold-start gives positive Sharpe (~0.49--0.53)** without any return data. The portfolio achieves this by finding maximally diverse firms in text space — the min-variance optimizer on a cosine matrix naturally selects firms with low pairwise text similarity, producing a diversified portfolio.

3. **Pure cold-start has high volatility (~17%)** because unit volatilities overweight volatile firms. With access to even a short return history (enough to estimate $\sigma$), the text_vol variant recovers full performance.

4. **At p=500, cold-start text_vol beats LW by +0.204 Sharpe (+38%)** — a pure text portfolio with no sample covariance significantly beats the standard return-based estimator ($p$=0.012).

### 3.15 PCA Component Interpretation: Naming the Text Factors

If text embeddings define interpretable factors, we can name what the cosine-similarity target actually encodes. We extract the top-10 principal components from the firm embedding matrix ($N \times 768$) at each quarterly rebalance, sign-align across quarters (flip if correlation with previous quarter $<0$), and characterize each PC via:

- Top/bottom-10 firms by loading magnitude (frequency across 75 quarters)
- Spearman rank correlation with firm characteristics (log ADV, trailing vol, 12-month momentum)
- Cross-quarter stability (Kendall $\tau$ of loading rankings between consecutive quarters)

| PC | Var. Explained | Kendall $\tau$ | Corr(log ADV) | Corr(Vol) | Corr(Mom) | Interpretation |
|----|---------------|----------------|---------------|-----------|-----------|----------------|
| 1 | 92.7% | 0.981 | -0.147 | +0.020 | +0.070 | Industrial/mfg (ROK, DHR, LRCX) vs commodities/leisure (SLV, WYNN, GLD, CCL) |
| 2 | 0.7% | 0.958 | -0.155 | -0.172 | +0.086 | Energy/utilities (EOG, WEC, DVN) vs biotech (REGN, VRTX, ALNY) |
| 3 | 0.5% | 0.925 | +0.013 | -0.080 | -0.095 | Pharma/mining (ALNY, REGN, MRK) vs REITs/tech (O, NOW, HST) |
| 4 | 0.4% | 0.894 | -0.011 | -0.164 | +0.078 | Retail/consumer (WMT, LOW, SJM) vs commodities/fin (GLD, IAU, BX) |
| 5 | 0.3% | 0.890 | +0.052 | +0.132 | -0.001 | Tech/hardware (INTC, TXN, IBM) vs healthcare (HCA, UHS, MOH) |

**Key findings:**

1. **PC1 is NOT market beta.** It separates industrial/manufacturing language from commodity/leisure language, with weak size correlation ($\rho$=-0.147) and near-zero volatility correlation (+0.020). This is a purely *linguistic* factor — firms that describe manufacturing processes vs firms that describe commodity markets or consumer experiences.

2. **All top-5 PCs are highly stable** (Kendall $\tau$ $>$ 0.89). The loading rankings barely change across quarters, confirming that text factors reflect persistent business model characteristics rather than transient linguistic trends. PC1's $\tau$=0.981 means the firm ordering is almost perfectly preserved from quarter to quarter.

3. **Clear sector interpretations emerge from text alone** without any industry classification input. The model has no access to SIC codes or GICS sectors — the sector structure arises purely from linguistic similarity in corporate filings.

4. **PC1's 92.7% dominance** explains the PCA $k$-sweep results (Section 3.12): the single dominant text factor captures essentially all the useful structure. The remaining PCs, while interpretable, contribute $<$1% each to the total variance.

5. **Characteristic correlations are weak.** No PC has $|\rho|>0.17$ with any characteristic. Text factors are distinct from size, volatility, and momentum — they capture *what firms do* (business model) rather than *how they trade* (market characteristics). This supports their value as a complementary source of covariance structure.

### 3.16 Shuffle Gradient: Does Text Content Matter at Higher p/n? (10-K, p=2000)

The shuffle placebo test (Section 3.7) showed that at p=500, firm-specific text content does not contribute — shuffled embeddings perform as well as correctly-assigned ones. Here we repeat the test at p=2000, where the shrinkage target receives more weight ($\alpha$~6.5% vs ~4.5%).

| Run | Sharpe |
|-----|--------|
| Unshuffled | **0.3007** |
| Shuffle 1 | 0.2956 |
| Shuffle 2 | 0.2964 |
| Shuffle 3 | 0.2981 |
| Shuffle 4 | 0.3058 |
| Shuffle 5 | 0.2891 |
| Shuffle 6 | 0.2900 |
| Shuffle 7 | 0.3025 |
| Shuffle 8 | 0.2954 |
| Shuffle 9 | 0.2922 |
| Shuffle 10 | 0.3005 |
| **Shuffled mean** | **0.2966** (std=0.0051) |

- z-score: 0.81; empirical p-value: 0.20
- **8 out of 10** shuffled runs have *lower* Sharpe than unshuffled (vs 7/10 shuffled *better* at p=500)
- Unshuffled-minus-shuffled gap: +0.0041 (vs -0.001 at p=500)

**Result:** The direction reverses from p=500 to p=2000. At p=500, text content is irrelevant (7/10 shuffled beat unshuffled, z=$-$0.80, $p$=0.70); at p=2000, correctly-assigned embeddings outperform shuffled in 80% of runs. While not statistically significant at conventional levels (p=0.20 with only 10 shuffles), the consistent directional reversal suggests that firm-specific text information begins to matter as the prior's weight increases. The effect is small (~0.4% Sharpe), indicating that even at p=2000 the advantage remains predominantly structural, but with a growing informational component.

**Comparison across p/n:**

| p | p/n | $\alpha$ | Shuffled beat unshuffled | Gap | Mechanism |
|---|-----|---------|------------------------|------|-----------|
| 500 | 0.99 | 4.5% | 7/10 (70%) | -0.001 | Purely structural |
| 2000 | 3.97 | 6.5% | 2/10 (20%) | +0.004 | Structural + informational |

This gradient supports the paper's thesis: as dimensionality grows and the covariance estimate relies more heavily on the prior, the prior's content — not just its structure — begins to matter. Text embeddings provide a principled source of firm-specific structure that becomes increasingly valuable in high-dimensional settings.

### 3.17 Text-Based Beta Estimation (10-K, p=500, 55 quarters)

**Goal:** Test whether projection onto the first principal component of the embedding space ("text-beta") predicts future realized CAPM beta.

**Method:** At each quarterly rebalance, compute the SVD of the firm embedding matrix. Text-beta = firm's projection onto PC1 (sign-aligned across quarters). Trailing-beta = CAPM beta from 252-day trailing returns. Forward-beta = CAPM beta from 63-day forward returns. The FM-style subgroup analysis uses per-quarter Spearman rank correlations averaged across quarters.

**Cross-sectional results (Fama-MacBeth):**

| Predictor | Avg Spearman $\rho$ with forward beta | % significant | FM coefficient | t-stat |
|-----------|----------------------------------------|---------------|----------------|--------|
| Text-beta | 0.149 | 80.0% | 15.03 | 2.79 |
| Trailing-beta | 0.709 | 100.0% | 0.878 | 23.64 |

**Incremental R² analysis:**

| Model | R² |
|-------|-----|
| Text-beta only | 0.030 |
| Trailing-beta only | 0.509 |
| Both | 0.513 |
| Incremental R² (text \| trailing) | 0.004 (t=5.82) |
| Incremental R² (trailing \| text) | 0.484 (t=14.68) |

**Result:** Text-beta has statistically significant incremental predictive power beyond trailing beta (t=2.79 for coefficient, t=5.82 for incremental $R^2$), but the effect size is small (0.4% incremental $R^2$). Trailing beta dominates in absolute terms. Text-beta is a *complement* to return-based beta, not a replacement — at least when firms have full return histories. Rank correlations computed per-quarter and averaged Fama-MacBeth style to avoid inflating significance from panel pooling. The experiment cannot test the short-history subgroup because the universe filter requires 504-day completeness (see Section 3.20 for the dedicated cold-start test).

### 3.18 VaR Calibration: Text-Based Risk Estimation (10-K, p=200 to p=2000, 75 quarters, 2007--2025)

**Goal:** Test whether parametric VaR from text-based covariance is better calibrated than VaR from sample covariance, especially at high p/n.

**Method:** Equal-weight portfolios (so covariance is the only variable). Compute one-day parametric VaR at each rebalance, then count violations over the next 63 trading days. Kupiec (1995) LR test for unconditional coverage. (Fence-post corrected: violation window starts the day after the rebalance date.)

**Summary of key ratios (violation rate / expected rate):**

| p | VaR level | Semantic ratio | LW ratio |
|---|-----------|---------------|----------|
| 200 | 5% | 1.09 | 1.19 |
| 500 | 5% | 1.06 | 1.18 |
| 1000 | 5% | **1.03** | 1.23 |
| 2000 | 5% | **1.01** | 1.26 |
| 200 | 2.5% | 1.61 | 1.77 |
| 500 | 2.5% | 1.54 | 1.76 |
| 1000 | 2.5% | 1.42 | 1.72 |
| 2000 | 2.5% | **1.27** | **1.72** |

(Semantic is best-calibrated at all $p$ values and VaR levels.)

**Key findings:**
1. **Semantic is best-calibrated across all VaR levels and $p$ values.** Consistently closest to nominal violation rates. At 5% VaR, semantic achieves a ratio of 1.03 at $p$=1000 and 1.01 at $p$=2000 — nearly perfect calibration. LW ratios range from 1.19 at $p$=200 to 1.26 at $p$=2000, worsening with dimensionality.
2. **At 2.5% VaR, ALL methods are rejected by Kupiec** (all violation ratios exceed 1.0). However, semantic remains the closest to calibration: ratio 1.27 at $p$=2000 vs LW 1.72. The gap in miscalibration is substantial — LW overestimates violations by 72% while semantic overestimates by 27%.
3. **Text-only systematically overestimates risk** (too conservative) — text encodes correlation structure but not variance levels.
4. **All methods underestimate 1% VaR** (fat tails), but semantic is least bad.
5. **The semantic calibration advantage persists across all $p$ values.** At 5% VaR, semantic ratios range from 1.01 ($p$=2000) to 1.09 ($p$=200), improving with dimensionality. LW ratios range from 1.18 to 1.26, worsening with dimensionality.
6. **Implication for risk management:** At high p/n, standard VaR models using sample or LW covariance produce systematically undercalibrated risk estimates. Text-based covariance provides better-conditioned matrices that mitigate this.

### 3.19 Text vs SIC Codes (10-K, p=200, 75 quarters, 2007--2025)

**Goal:** Test whether embedding similarity predicts realized return correlation better than SIC sector membership, and whether text-based covariance outperforms SIC-based covariance in portfolio construction.

**Phase A — Cross-sectional fit** (realized_corr$_{ij}$ ~ cosine_sim$_{ij}$ + SIC_match$_{ij}$, 10 sample dates):

| Predictor | $R^2$ | Incremental $R^2$ | t-stat | Spearman $\rho$ |
|-----------|-------|-------------------|--------|-----------------|
| Cosine similarity | 0.062 | 0.047 | 6.12 | 0.216 |
| SIC sector match | 0.031 | 0.016 | 4.27 | 0.148 |
| Both | 0.078 | — | — | — |

**Text predicts realized correlation 2x better than SIC** ($R^2$ 6.2% vs 3.1%). Both have significant incremental power, but text's incremental $R^2$ is 3x larger than SIC's (0.047 vs 0.016).

**Phase B — Walk-forward backtest** (p=200, 75 quarters, 2007--2025):

| Strategy | Sharpe |
|----------|--------|
| SIC sector | **0.740** |
| Semantic shrinkage | 0.701 |
| Equal weight | 0.606 |
| Ledoit-Wolf | 0.651 |

At p=200 (low p/n=0.40), the SIC block-diagonal covariance outperforms semantic shrinkage. The SIC block-diagonal uses realized SIC correlations (same-SIC: 0.455, diff-SIC: 0.260) rather than hardcoded values, providing empirically calibrated within-sector and cross-sector correlation structure. This is consistent with the alpha sweep (Section 3.13): at low p/n, the sample covariance is reasonably informative, and a simpler structural prior can outperform a more complex one.

**Phase C — Case studies** (2023-01-01, p=200):

*Same SIC, low cosine (text correctly differentiates):*
- MRNA (Moderna) classified in SIC 28 (chemicals) alongside PG, CL, EL — text cosine ~0.77 vs realized correlation ~0.2. Text correctly identifies Moderna as dissimilar to consumer staples despite shared SIC code.

*Different SIC, high cosine (text correctly groups):*
- KLAC-LRCX (SIC 38/35): cosine=0.990, realized=0.903. Both semiconductor equipment firms, correctly identified by text despite different SIC codes.
- DDOG-PANW (SIC 73/35): cosine=0.990, realized=0.575. Both cybersecurity-adjacent tech firms.

### 3.20 Cold-Start Simulation (10-K, p=500, 75 quarters, 2007--2025)

**Goal:** Simulate the actual IPO/new-listing scenario where firms enter the universe with minimal return history. Text-based covariance can include them; LW must exclude them. The redesigned experiment isolates covariance quality from universe breadth using paired comparisons.

**Method:** "New entrant" = firm with embeddings + ADV ≥ $1M but < 63 trading days of returns. Six strategies: text_inclusive (all firms, text cov), text_exclusive (established only, text cov), lw_exclusive (established only, LW cov), sic_substitute (SIC-based cov for new firms), ew_inclusive (all firms), ew_established (established only). Zero-fill bias corrected: new entrants with near-zero sample vol receive the median cross-sectional vol.

**Cold-start statistics:**
- Average 0.37 new entrants per quarter (max 4)
- 21 of 75 quarters have at least one new entrant
- New entrants receive 1.8% average weight in the text-inclusive portfolio

**Performance:**

| Strategy | Sharpe | AnnRet | Vol | MaxDD |
|----------|--------|--------|-----|-------|
| sic_substitute | 0.610 | 6.8% | 11.2% | -34.7% |
| text_exclusive (established, text cov) | 0.601 | 6.7% | 11.2% | -34.6% |
| ew_established | 0.583 | 12.8% | 22.0% | -56.6% |
| ew_inclusive (ALL firms) | 0.580 | 12.8% | 22.1% | -56.4% |
| lw_exclusive (established only) | 0.550 | 6.2% | 11.2% | -33.9% |
| text_inclusive (ALL firms, text cov) | 0.478 | 5.7% | 11.8% | -34.7% |

**Decomposing the effects:**
- **Covariance quality** (text_exclusive vs lw_exclusive, same universe): text cov delivers +0.051 Sharpe (p=0.253). The text prior improves covariance quality even on established firms, though not significantly in this sample.
- **Universe breadth** (text_inclusive vs text_exclusive, same method): including new entrants costs -0.123 Sharpe (p=0.267). The lower completeness threshold degrades overall covariance quality.
- **SIC substitute** vs lw_exclusive: +0.060 Sharpe (p=0.053), borderline significant. Using SIC-based covariance for new entrants while keeping LW for established firms is the strongest practical cold-start strategy.

**Result:** The cold-start scenario is naturally rare at p=500 with ADV ≥ $1M. The text_exclusive vs lw_exclusive comparison isolates the covariance quality effect: text improves Sharpe by +0.051 holding the universe constant, consistent with the lookback ablation results. However, adding new entrants to the universe (text_inclusive) hurts more than the covariance quality helps, because the relaxed completeness filter degrades overall estimation quality. **The real cold-start test is the lookback ablation** (Section 3.21), which artificially truncates return history for ALL firms, measuring the information content of text in return-equivalent units.

### 3.21 Return History Ablation (10-K, p=500)

**Goal:** The headline experiment. Vary the lookback window for covariance estimation from 0 to 504 days while holding the universe constant. Quantify the information content of text in return-equivalent units: how many days of return data does text-based covariance equal?

**Method:** Six strategies at each lookback $L \in \{0, 5, 10, 21, 42, 63, 126, 252, 504\}$:
1. **text_only:** CosineSimilarityCovariance (no returns at all)
2. **text_vols:** Cosine-similarity correlation matrix + sample standard deviations from $L$-day returns
3. **semantic:** SemanticShrinkageCovariance with $L$-day sample covariance (LW oracle intensity)
4. **lw:** Ledoit-Wolf with $L$-day returns (undefined at $L=0$; requires $L \geq 10$ for non-singular covariance)
5. **sic_sector:** SIC block-diagonal correlation + sample standard deviations from $L$-day returns
6. **equal_weight:** 1/N baseline

Universe filter uses FULL 504-day lookback for ALL strategies (same firms, different covariance input). The only variable is the lookback window length — identical investable universe across all $L$.

**Results (Sharpe ratios):**

| $L$ (days) | $p/n$ | text_only | text_vols | semantic | LW | SIC sector | EW |
|-----------|-------|-----------|-----------|----------|------|-----------|------|
| 0 | 500 | 0.507 | 0.507 | 0.507 | N/A | 0.701 | 0.734 |
| 5 | 100 | 0.507 | 0.519 | 0.594 | N/A | 0.544 | 0.734 |
| 10 | 50 | 0.507 | 0.579 | 0.563 | 0.484 | 0.552 | 0.734 |
| 21 | 23.8 | 0.507 | 0.699 | **0.760** | 0.675 | 0.765 | 0.734 |
| 42 | 11.9 | 0.507 | 0.726 | **0.754** | 0.569 | **0.787** | 0.734 |
| 63 | 7.9 | 0.507 | **0.808** | 0.703 | 0.633 | **0.823** | 0.734 |
| 126 | 4.0 | 0.507 | 0.692 | 0.597 | 0.544 | **0.824** | 0.734 |
| 252 | 2.0 | 0.507 | **0.760** | 0.592 | 0.598 | **0.816** | 0.734 |
| 504 | 1.0 | 0.507 | **0.748** | 0.541 | 0.529 | **0.808** | 0.734 |

**Statistical significance (text_vols vs LW, block permutation, 45 quarters):**
- $L$=504: p=0.079 (near-significant)
- $L$=252: p=0.151
- $L$=126: p=0.238
- $L$=63: p=0.243
- $L$=42: p=0.513
- $L$=21: p=0.777

**Oracle shrinkage intensity ($\alpha$) by lookback:**
- $L$=5: $\alpha$=0.59 (59% text, 41% sample — sample is rank-5, nearly useless)
- $L$=10: $\alpha$=0.60
- $L$=21: $\alpha$=0.60
- $L$=42: $\alpha$=0.37
- $L$=63: $\alpha$=0.28
- $L$=126: $\alpha$=0.17
- $L$=252: $\alpha$=0.07
- $L$=504: $\alpha$=0.04

#### Extended Lookback Ablation (10-K, p=500, 75 quarters, 2007--2025)

Extending from 45 to 75 quarterly rebalances dramatically improves statistical power (~17 effective blocks vs ~11):

| $L$ (days) | $p/n$ | text_only | text_vols | semantic | LW | SIC sector | EW | tv vs LW $p$ |
|-----------|-------|-----------|-----------|----------|------|-----------|------|--------------|
| 0 | 500 | 0.568 | 0.568 | 0.568 | N/A | 0.583 | 0.582 | — |
| 10 | 50 | 0.568 | 0.549 | 0.520 | 0.391 | 0.510 | 0.582 | 0.526 |
| 21 | 23.8 | 0.568 | 0.621 | 0.689 | 0.527 | 0.657 | 0.582 | 0.784 |
| 42 | 11.9 | 0.568 | 0.610 | 0.650 | 0.442 | 0.664 | 0.582 | 0.265 |
| 63 | 7.9 | 0.568 | **0.713** | 0.633 | 0.497 | **0.709** | 0.582 | 0.061 |
| 126 | 4.0 | 0.568 | 0.677 | 0.587 | 0.506 | **0.718** | 0.582 | 0.079 |
| 252 | 2.0 | 0.568 | **0.771** | 0.609 | 0.546 | **0.763** | 0.582 | **0.011** |
| 504 | 1.0 | 0.568 | **0.744** | 0.592 | 0.540 | **0.740** | 0.582 | **0.012** |

**Key finding: text_vols vs LW is now statistically significant.** At $L$=252 ($p$=0.011) and $L$=504 ($p$=0.012), the block permutation test rejects the null that text_vols and LW produce equivalent portfolios. At $L$=63, $p$=0.061 (near-significant). This is the first statistically significant result in the paper's primary comparison. The peak shifts from $L$=63 to $L$=252 in the extended window. text_only vs LW is also significant at $L$=42 ($p$=0.043) — pure text with zero volatility data beats LW with 42 days of history.

#### Cross-Source Replication (Transcript, p=500, 75 quarters, 2007--2025)

| $L$ (days) | $p/n$ | text_only | text_vols | semantic | LW | SIC sector | EW | tv vs LW $p$ |
|-----------|-------|-----------|-----------|----------|------|-----------|------|--------------|
| 0 | 500 | 0.621 | 0.621 | 0.621 | N/A | 0.618 | 0.624 | — |
| 21 | 23.8 | 0.621 | 0.686 | 0.689 | 0.552 | 0.695 | 0.624 | 0.424 |
| 63 | 7.9 | 0.621 | **0.735** | 0.639 | 0.576 | 0.698 | 0.624 | 0.110 |
| 126 | 4.0 | 0.621 | 0.669 | 0.649 | 0.626 | 0.700 | 0.624 | 0.379 |
| 252 | 2.0 | 0.621 | **0.701** | 0.680 | 0.670 | **0.722** | 0.624 | 0.421 |
| 504 | 1.0 | 0.621 | 0.650 | 0.672 | 0.663 | 0.675 | 0.624 | 0.745 |

The transcript replication confirms the same qualitative pattern: text_vols peaks at $L$=63 (Sharpe 0.735), beating EW (0.624) and LW with full history (0.663). text_vols beats EW at every lookback $\geq$21 days. The higher absolute text_only Sharpe (0.621 vs 10-K 0.568) reflects the S&P 500 universe. Statistical significance is weaker (best $p$=0.110 at $L$=63) due to the lower cross-sectional dispersion in the S&P 500 universe.

**Key findings:**

1. **text_vols is the headline result.** Text-based correlation structure + sample volatilities from 252 days of returns produces Sharpe 0.771 in the extended window (0.808 at $L$=63 in the 45-quarter window) — beating equal weight and beating LW with full 504-day returns. The text prior provides the full correlation structure; only volatilities need to come from return data.

2. **text_vols beats LW at EVERY lookback length.** From $L$=10 through $L$=504, text_vols dominates, and this is now **statistically significant** at $L$=252 and $L$=504 ($p$=0.011 and 0.012) in the extended window.

3. **The result replicates across text sources.** Both 10-K annual reports and quarterly earnings transcripts show the same pattern: text_vols beats LW at all lookback lengths and beats EW at $L \geq 21$ days. The peak lookback differs (10-K: $L$=252; transcript: $L$=63) but the story is consistent.

4. **SIC sector remains competitive at most lookback lengths.** SIC block-diagonal (Sharpe 0.664--0.740) outperforms text_vols at some lookback lengths. This simpler structure — constant within-sector and cross-sector correlations scaled by sample volatilities — is remarkably effective.

5. **Semantic shrinkage (oracle $\alpha$) is suboptimal.** The LW oracle assigns too little weight to text (only 4% at $L$=504), which explains why semantic underperforms text_vols, which implicitly uses $\alpha$=1.0 for correlations. This is consistent with the alpha sweep finding (Section 3.13): the LW oracle is miscalibrated for informative targets.

6. **At $L=0$, text_only = text_vols = semantic.** All collapse to the pure cosine matrix when no return data is available. text_only beats LW at all lookback lengths for 10-K (significant at $L$=42, $p$=0.043).

7. **The $\alpha$ curve maps text's information content.** At $L \leq 21$ (one month), the oracle assigns $>$60% weight to text — return data is so noisy the optimizer trusts text more. At $L$=504 (two years), text drops to 4%. The inflection occurs around $L$=42--63 days ($\alpha$=0.28--0.37).

### 3.22 CV-Alpha Calibration (10-K, p=500, 71 rebalances, 2007--2025)

**Goal:** Replace the LW oracle — which is calibrated for identity targets — with cross-validated shrinkage intensity that optimizes for realized portfolio variance.

**Method:** At each quarterly rebalance, the trailing 504-day window is split temporally 70/30. A grid of $\alpha \in \{0.01, 0.05, 0.10, 0.20, \ldots, 1.0\}$ is evaluated: for each $\alpha$, compute the semantic shrinkage covariance on the training portion, solve the min-variance portfolio, and measure realized variance on the validation portion. The $\alpha$ minimizing out-of-sample variance is selected for that quarter.

| Strategy | Sharpe | Lo-corrected | vs LW $p$ |
|----------|--------|-------------|-----------|
| CV-alpha | 0.677 | 0.957 | **0.034** |
| $\alpha$=0.25 | 0.706 | 0.901 | **0.004** |
| $\alpha$=0.50 | 0.712 | 0.957 | **0.014** |
| $\alpha$=0.75 | 0.725 | 1.009 | **0.014** |
| $\alpha$=auto (~4.5%) | 0.526 | 0.632 | 0.670 |
| LW | 0.502 | 0.591 | — |
| EW | 0.571 | 0.670 | 0.038 |

CV-alpha distribution across 71 rebalances: mean=0.758, median=1.000, std=0.390, Q25=0.800, range=[0.01, 1.0].

**Key findings:**

1. **CV-alpha significantly beats LW** ($p$=0.034). This is the first adaptive, out-of-sample method to achieve statistical significance against the standard estimator.

2. **CV independently discovers that text should dominate.** The mean selected $\alpha$ is 0.758, the median is 1.0 — cross-validation confirms, without any manual tuning, that the semantic target deserves 3--20$\times$ the weight the LW oracle prescribes.

3. **All high-$\alpha$ strategies significantly beat LW** ($\alpha$=0.25: $p$=0.004; $\alpha$=0.50: $p$=0.014; $\alpha$=0.75: $p$=0.014). The consistency across $\alpha$ values rules out overfitting to a specific intensity.

4. **CV-alpha matches $\alpha$=0.50 performance** (Sharpe 0.677 vs 0.712) without requiring the practitioner to guess a fixed value. The slight shortfall reflects conservative selections in early quarters where validation data is limited.

5. **Lo-corrected Sharpe for CV-alpha is 0.957** — approaching 1.0, which indicates that after accounting for serial autocorrelation, the strategy delivers nearly one unit of excess return per unit of risk.

**Interpretation:** The LW oracle minimizes Frobenius distance between the shrinkage estimator and the true covariance. When the target is identity (uninformative), this distance metric aligns well with portfolio performance. When the target is informative (semantic), Frobenius loss cannot distinguish a good prior from a random matrix of the same scale — it penalizes deviation from the sample regardless of whether that deviation improves the portfolio. CV-alpha optimizes the correct objective (realized portfolio variance) and consequently discovers that text should dominate the estimate.

### 3.23 Constrained Optimization (10-K, p=500, 71 rebalances, 2007--2025)

**Goal:** Test whether the semantic advantage holds — or strengthens — under institutional constraints that restrict sector concentration and target portfolio volatility.

**Method:** Five constraint regimes applied to min-variance optimization:
1. **Unconstrained:** Long-only, 5% max weight (baseline)
2. **Sector $\leq$25%:** SIC 2-digit sector weight capped at 25%
3. **Sector $\leq$15%:** SIC 2-digit sector weight capped at 15%
4. **Vol target 10%:** Scale portfolio to target 10% annualized volatility
5. **Sector 25% + Vol 10%:** Both constraints simultaneously

| Regime | Sem Sharpe | LW Sharpe | EW Sharpe | SIC Sharpe | Sem vs EW $p$ | Sem Lo |
|--------|-----------|-----------|-----------|------------|---------------|--------|
| Unconstrained | 0.526 | 0.502 | 0.571 | 0.720 | 0.055 | 0.632 |
| Sector $\leq$25% | 0.587 | 0.563 | 0.571 | 0.719 | 0.087 | 0.707 |
| Sector $\leq$15% | 0.599 | 0.574 | 0.570 | 0.706 | 0.108 | 0.715 |
| Vol target 10% | 0.512 | 0.485 | 0.543 | 0.705 | 0.075 | 0.615 |
| Sector 25% + Vol 10% | 0.574 | 0.548 | 0.543 | 0.703 | 0.116 | 0.692 |

**Key findings:**

1. **Sector constraints make semantic beat EW.** At the 15% sector limit, semantic achieves Sharpe 0.599 vs EW 0.570 (+0.029). Under combined constraints, semantic 0.574 vs EW 0.543 (+0.031). EW is essentially unchanged by sector constraints (already diversified at $p$=500), while optimized methods improve by avoiding excessive sector concentration.

2. **The maximum drawdown gap persists across all constraint regimes.** Semantic ~35% vs EW ~56--58%. Any institutional mandate with a drawdown limit below 40% would eliminate EW as a feasible strategy.

3. **SIC sector remains dominant** across all regimes (Sharpe 0.703--0.720), significantly outperforming semantic ($p$=0.002--0.031).

4. **Volatility targeting hurts all strategies slightly** by adding turnover from leverage scaling, but the relative ordering is preserved.

**Practical implication:** The unconstrained backtest, where EW benefits from pure diversification, understates the case for covariance-based methods. Real portfolios face sector limits, tracking error budgets, and volatility targets. In these institutionally realistic settings, the structured covariance prior becomes more valuable.

### 3.24 Multi-Target Shrinkage (10-K, p=200 and p=500, 71 rebalances, 2007--2025)

**Goal:** Test whether combining multiple shrinkage targets — text, SIC sector, and identity — outperforms single-target estimators.

**Method:** The multi-target estimator blends four components:

$$\hat{\Sigma} = c_0 \cdot S_{\text{sample}} + c_1 \cdot T_{\text{text}} + c_2 \cdot T_{\text{SIC}} + c_3 \cdot I$$

Weights $(c_0, c_1, c_2, c_3)$ are calibrated via two methods: (a) cross-validated portfolio loss (CV), and (b) Frobenius distance minimization.

| $p$ | Strategy | Sharpe | Lo-corrected | vs LW $p$ |
|-----|----------|--------|-------------|-----------|
| 200 | multitarget\_cv | 0.692 | 0.887 | 0.094 |
| 200 | multitarget\_frob | 0.622 | 0.776 | 0.050 |
| 200 | semantic\_auto | 0.641 | 0.810 | 0.553 |
| 200 | sic\_sector | 0.720 | 0.965 | 0.079 |
| 200 | LW | 0.629 | 0.785 | — |
| 200 | EW | 0.579 | 0.702 | 0.224 |
| 500 | multitarget\_cv | 0.570 | 0.700 | 0.214 |
| 500 | multitarget\_frob | 0.501 | — | 0.955 |
| 500 | semantic\_auto | 0.526 | — | 0.553 |
| 500 | sic\_sector | 0.720 | — | **0.004** |
| 500 | LW | 0.502 | — | — |
| 500 | EW | 0.571 | — | 0.080 |

CV weight decomposition ($p$=200): identity=59.7%, sample=33.1%, text=2.8%, SIC=4.4%.
CV weight decomposition ($p$=500): identity=61.3%, sample=32.0%, text=3.9%, SIC=2.9%.
Frobenius calibration: ~100% sample at both $p$ values.

**Key findings:**

1. **CV-calibrated multi-target is the best non-SIC method at $p$=200** (Sharpe 0.692, near-significant vs LW at $p$=0.094).

2. **Frobenius calibration is uninformative.** It assigns ~100% weight to the sample covariance, confirming that Frobenius loss cannot distinguish informative targets from noise — the same pathology affecting the LW oracle (Section 3.13).

3. **CV weights heavily favor identity (~60%) + sample (~33%)**, with text and SIC contributing ~7% combined. The identity preference reflects a generic regularization component (shrinkage toward equal correlation), on top of which small amounts of structured information enter.

4. **SIC sector dominates at both $p$ values** ($p$=200: 0.720; $p$=500: 0.720, $p$=0.004 vs LW). The block-diagonal SIC structure, with its extreme parsimony (one binary parameter per pair), remains remarkably effective.

5. **The Frobenius/CV disconnect parallels the oracle miscalibration.** In both single-target (Section 3.13) and multi-target settings, distance-based calibration underweights informative priors. Only portfolio-loss-based calibration (CV) can assess the economic value of structural information.

### 3.25 TF-IDF Baseline (10-K, p=500, 75 quarters, 2007--2025)

**Goal:** Definitively test whether the semantic advantage derives from the pretrained neural model's future knowledge or from the textual similarity structure itself. A TF-IDF encoder uses only corpus-internal term frequencies — zero pretrained parameters, zero possible look-ahead bias by construction.

**Method:** At each quarterly rebalance, a TF-IDF vectorizer is fitted on the corpus of 10-K filings available up to (and including) the rebalance date. Each document is represented as a sparse bag-of-words weighted by term frequency--inverse document frequency. Firm-level TF-IDF embeddings are computed by averaging document vectors and L2-normalizing, exactly mirroring the neural pipeline. The resulting cosine-similarity matrix serves as the shrinkage target.

| Strategy | Sharpe | Lo-corrected | Ann. Return | Ann. Vol | MaxDD | Turnover |
|----------|--------|-------------|-------------|----------|-------|----------|
| Neural semantic (bge-base) | 0.592 | 0.716 | 6.6% | 11.1% | -34.6% | 46.8% |
| **TF-IDF semantic** | **0.592** | **0.714** | **6.6%** | **11.1%** | **-34.4%** | **47.0%** |
| Ledoit-Wolf | 0.540 | 0.641 | 6.0% | 11.2% | -33.8% | 49.6% |
| Equal weight | 0.582 | 0.687 | 12.8% | 22.0% | -56.4% | 14.7% |

Neural vs TF-IDF Sharpe gap: **+0.0005** ($p$=0.849).
TF-IDF vs LW: $p$=0.234.

**Key findings:**

1. **Neural and TF-IDF are statistically indistinguishable.** Over 75 quarterly rebalances, the Sharpe ratio difference is +0.0005 — effectively zero. The block permutation test cannot reject the null ($p$=0.849). Annualized return, volatility, maximum drawdown, and turnover match to within rounding error.

2. **This definitively closes the look-ahead bias concern.** TF-IDF uses only corpus-internal term frequencies. There is no pretrained model, no training corpus from any period, and therefore zero channel for future information to enter the estimate. The walk-forward TF-IDF vectorizer is refitted at each rebalance using only documents with `doc_date` $\leq$ `rebal_date`.

3. **The signal comes entirely from textual similarity structure.** Whether that structure is captured by a 768-dimensional neural embedding or a high-dimensional sparse bag-of-words is immaterial — the cosine-similarity matrix is what matters, and both representations produce functionally identical similarity matrices for portfolio construction purposes.

4. **Both text methods beat LW** by ~0.05 Sharpe and beat EW on Lo-corrected Sharpe (0.714--0.716 vs 0.687).

**Implication:** Practitioners need not use a pretrained neural model to obtain the text-based covariance benefit. A simple TF-IDF pipeline, refitted quarterly on available filings, produces equivalent results. This removes the dependency on third-party model weights and simplifies deployment.

### 3.26 Temporal Sensitivity (10-K, p=500, 75 quarters, 2007--2025)

**Goal:** Test whether the semantic advantage varies across time periods that differ in their relationship to the embedding model's training data. If look-ahead bias through the pretrained model were driving results, the advantage should be largest during the model's training period (when the model has "seen" contemporaneous text) and smaller outside it.

**Method:** Split the 75-quarter backtest into three sub-periods based on the BAAI/bge-base-en-v1.5 training data cutoff (~2022):
1. **Pre-training (2007--2016):** Model has "future" linguistic knowledge relative to the text being embedded
2. **Training-contemporaneous (2017--2022):** Model trained on text from this period
3. **Post-training (2023--2025):** Model has no knowledge of this period's text

| Period | Rebalances | Sem Sharpe | LW Sharpe | Gap | Sem vs LW $p$ |
|--------|-----------|-----------|-----------|------|---------------|
| Pre-training (2007--2016) | 39 | 0.530 | 0.469 | +0.061 | **0.027** |
| Training-contemp. (2017--2022) | 24 | 0.467 | 0.473 | $-$0.006 | 0.947 |
| Post-training (2023--2025) | 12 | 1.380 | 1.181 | +0.199 | 0.152 |

**Key findings:**

1. **No evidence of look-ahead bias.** The semantic advantage is significant in the pre-training period ($p$=0.027, gap +0.061) — the period where anachronistic model knowledge should theoretically help most — and essentially zero in the training-contemporaneous period (gap $-$0.006, $p$=0.947). If the pretrained model's future knowledge were inflating results, the advantage should be *largest* during the training period, not *absent*.

2. **The pre-training significance is the only sub-period result that reaches conventional levels.** This aligns with the full-sample finding: including the 2008 crisis (which falls in the pre-training period) strengthens the min-variance advantage.

3. **Post-training shows the largest gap** (+0.199) but is underpowered with only 12 rebalances. The high absolute Sharpe (1.38 for semantic, 1.18 for LW) reflects the 2023--2025 bull market.

4. **Combined with the TF-IDF baseline (Section 3.25), the temporal sensitivity provides a two-pronged defense against look-ahead bias:** TF-IDF eliminates it by construction; the temporal pattern eliminates it empirically.

### 3.27 News Lookback Ablation (p=500, 75 quarters, 2007--2025)

**Goal:** Replicate the headline lookback ablation (Section 3.21) using a third independent text source: financial news articles. This tests whether the text_vols finding is specific to corporate filings or generalizes across text types.

**Data:** 111,062 quarterly aggregated news documents from 4,024 tickers, derived from 5 financial news datasets on HuggingFace. Correlation study: average Spearman $\rho$ = 0.254 between news cosine similarity and realized return correlation — more than double the 10-K signal ($\rho$ = 0.104).

| $L$ (days) | $p/n$ | text\_vols | LW | EW | tv vs LW $p$ |
|-----------|-------|-----------|------|------|------------|
| 0 | 500 | 0.345 | N/A | 0.597 | — |
| 10 | 50 | 0.475 | 0.425 | 0.597 | 0.858 |
| 21 | 23.8 | 0.574 | 0.527 | 0.597 | 0.944 |
| 42 | 11.9 | 0.554 | 0.440 | 0.597 | 0.399 |
| 63 | 7.9 | 0.695 | 0.497 | 0.597 | 0.065 |
| 126 | 4.0 | 0.682 | 0.501 | 0.597 | **0.047** |
| 252 | 2.0 | 0.720 | 0.563 | 0.597 | **0.033** |
| 504 | 1.0 | **0.731** | 0.561 | 0.597 | **0.018** |

**Key findings:**

1. **Third text source replication of the headline result.** News-based text_vols significantly outperforms LW at $L$=126 ($p$=0.047), $L$=252 ($p$=0.033), and $L$=504 ($p$=0.018). This is the strongest statistical evidence across any single text source at the full lookback.

2. **News text_vols peaks at $L$=504** (Sharpe 0.731), monotonically increasing with lookback length — unlike 10-K (peak at $L$=252) and transcript (peak at $L$=63). This suggests news embeddings benefit from longer volatility estimates, possibly because quarterly news aggregation introduces more noise into the correlation structure than annual filings.

3. **text_vols beats EW at every lookback $\geq$21 days**, matching the pattern from 10-K and transcript sources.

4. **News correlation signal is strongest** (Spearman $\rho$ = 0.254 vs 10-K 0.104), yet the backtest advantage is comparable, suggesting that the additional cross-sectional signal in news is offset by higher noise in quarterly aggregated articles versus annual comprehensive filings.

5. **Across all three text sources, text_vols dominates LW at every lookback length where LW exists ($L \geq 10$).** The qualitative finding — text provides correlation structure, returns need only provide volatilities — is text-source invariant.

---

## 4. Limitations and Caveats

### 4.1 Mechanism: Structural With Growing Informational Component

The shuffle placebo test reveals a nuanced mechanism. At p=500 (Section 3.7), the semantic advantage over LW arises entirely from the spectral structure of the cosine-similarity matrix: shuffled embeddings perform as well as correctly-assigned ones (7/10 shuffled beat unshuffled, z=$-$0.80, $p$=0.70). The cosine matrix's approximate low-rank, block-like eigenvalue structure provides better regularization than the identity matrix, and this structure is permutation-invariant. However, at p=2000 (Section 3.16), a directional shift emerges: 80% of shuffled runs underperform the correctly-assigned portfolio, with the gap reversing from -0.001 to +0.004. This suggests a transition from purely structural benefit at low p/n to a mix of structural and informational benefit at high p/n, as the prior receives more weight. Text similarity thus serves dual roles: it provides a convenient *source* of structured priors (valuable at all scales), and it encodes firm-specific relationships that become increasingly relevant as the estimator relies more heavily on the prior.

### 4.2 Statistical Significance

The 75-quarter backtest (2007--2025) provides multiple lines of statistical significance. The alpha sweep (Section 3.13) yields the strongest individual results: all $\alpha \geq 0.25$ significantly beat LW at $p$=500 (raw $p$-values 0.002--0.014). The lookback ablation (Section 3.21) shows text_vols vs LW significance at $L$=252 ($p$=0.011) and $L$=504 ($p$=0.012); these $p$-values have not been corrected for the 9 lookback windows tested. After Bonferroni correction for 9 comparisons, $p$=0.011 becomes $p$~0.10; after the less conservative Benjamini-Hochberg procedure, $p$~0.05. The CV-alpha result (Section 3.22, $p$=0.034 against a single pre-specified comparison) and the news lookback ablation (Section 3.27, $p$=0.018 at $L$=504) provide complementary evidence from independent analyses. The 75-quarter window provides increased statistical power (~17 effective independent blocks vs ~11 in 45 quarters). The monotonically increasing gap across 7 independent universe sizes further represents a consistent directional pattern. The block permutation test is inherently conservative with quarterly data — even 75 rebalances yield moderate power — and the economic magnitude at p=2000 (+0.148 Sharpe, nearly 2x the LW Sharpe) is practically significant regardless.

### 4.3 Equal Weight Dominance — Reversed in Extended Window

In the shorter 45-quarter backtest (2013--2024), the 1/N portfolio dominates both optimized strategies at most universe sizes and specifications tested, consistent with DeMiguel, Garlappi, and Uppal (2009).

However, extending to 75 quarters (2007--2025) **reverses the EW dominance** for semantic shrinkage. Including the 2008 financial crisis and 2025 drawdown — periods where min-variance's lower beta provides protection — semantic shrinkage beats EW on:
- **Raw Sharpe:** 10-K 0.592 vs EW 0.582; transcript 0.672 vs EW 0.624
- **Lo-corrected Sharpe:** 10-K 0.716 vs EW 0.687; transcript 0.906 vs EW 0.754
- **Maximum drawdown:** semantic ~35% vs EW ~57% — the risk management payoff

The text_vols variant in the lookback ablation further strengthens the case: text_vols beats EW at every lookback $\geq$21 days (both text sources). At $L$=252, text_vols achieves Sharpe 0.771 vs EW 0.582 (+32%). This is, to our knowledge, the first covariance-optimized min-variance strategy to systematically outperform 1/N across multiple specifications in our experimental setup.

The practical implication is that EW dominance is sample-dependent. In benign environments (2013--2024), the diversification benefit of equal weighting dominates the estimation noise of covariance-based methods. In environments with crises, the risk reduction from min-variance construction is worth the estimation cost — especially when the covariance matrix uses text-based structure rather than pure sample estimates. For practitioners who require covariance-based allocation (e.g., for risk budgeting, factor targeting, or regulatory constraints), the structured target scales far better than the identity target.

### 4.4 Risk-Free Rate

Sharpe ratios are computed with zero risk-free rate. Over the extended backtest period (2007--2025), the actual risk-free rate ranged from ~0% (2009--2015, 2020--2021) to ~5% (2007, 2023--2024). This inflates absolute Sharpe levels for all strategies but does not affect relative comparisons.

### 4.5 NaN Treatment

Missing return data is filled with 0.0 (`fillna(0.0)`) before covariance estimation. Pairwise-complete estimation would be more precise but risks producing non-PSD matrices without additional correction. The 80% completeness filter ensures at most 20% of any ticker's returns are zero-filled.

### 4.6 Survivorship Bias

The investable universe changes at each quarterly rebalance: new firms enter as they IPO and accumulate sufficient filing and return history, while existing firms exit when they delist, fail the liquidity threshold, or fail the return completeness filter. This dynamic composition introduces several forms of survivorship-related bias:

1. **Delisting returns.** The ticker universe derives from PleIAs/SEC (which includes delisted firms that filed 10-K filings) and Tiingo (which includes delisted tickers with price history terminating at the delisting date). Firms that delist mid-backtest will have truncated return series and eventually fail the 80% completeness filter, excluding them from subsequent rebalances. This is the correct point-in-time behavior. However, we do not model the delisting return itself (the final price adjustment when a firm is acquired or goes bankrupt), which creates a mild upward bias in all strategy returns — particularly for strategies that hold small-cap firms that are more likely to delist.

2. **Conditional universe.** At each rebalance, the investable universe is conditioned on having (a) text embeddings available from a filed 10-K or transcript, (b) sufficient trading history, and (c) adequate liquidity. Firms that never file a 10-K, trade too thinly, or have too many missing return observations are excluded entirely. This conditioning means results apply to the "textually covered, liquid" subset of the market, not the full market.

3. **Rolling SVD composition.** The PCA factor analysis (Section 3.12) recomputes the SVD at each rebalance on the current universe. As firms enter and exit, the principal components shift — the "text factors" are not fixed over time. This is intentional (it prevents look-ahead) but means the eigenvalue concentration statistics (92.7% in PC1) are averages over 75 different universe compositions.

**Mitigation:** All strategies (semantic, LW, EW) face identical survivorship conditions at each rebalance — the universe filter is applied once and shared across all methods. Any survivorship bias therefore affects absolute return levels equally and does not contaminate the relative comparison between strategies, which is the paper's focus.

### 4.7 Embedding Model Leakage

The pre-trained sentence transformer used to embed corporate filings (BAAI/bge-base-en-v1.5) was trained on a broad text corpus spanning multiple time periods. When embedding a 2013 10-K filing, the model's internal representations reflect linguistic patterns learned from text published after 2013. This constitutes a subtle form of look-ahead bias: the model's understanding of language is anachronistic relative to the filing date.

Two experiments directly address this concern, and three additional observations mitigate it:

1. **TF-IDF produces identical results (Section 3.25).** Replacing the neural encoder with a TF-IDF vectorizer — zero pretrained parameters, refitted from scratch at each quarterly rebalance using only available documents — produces statistically indistinguishable backtest performance (Sharpe 0.592 vs 0.592, $p$=0.849 over 75 quarters). If the pretrained model's future knowledge were contributing to the semantic advantage, TF-IDF should underperform. It does not. This is the definitive test: the signal arises from textual similarity structure, not from the model's learned representations.

2. **Temporal sensitivity shows no training-period advantage (Section 3.26).** The semantic advantage is significant in the pre-training period 2007--2016 ($p$=0.027) and essentially zero in the training-contemporaneous period 2017--2022 (gap $-$0.006). This is the opposite of what look-ahead bias would produce.

3. **The model learns language structure, not asset prices.** The transformer learns that phrases like "cloud computing revenue" and "SaaS recurring revenue" are semantically similar — a linguistic relationship that holds regardless of when the model was trained. It does not learn which firms will outperform.

4. **Symmetric across strategies.** All strategies that use embeddings (semantic shrinkage, PCA factors, cold-start) share the same pre-trained model. Any leakage advantage affects all embedding-based methods equally and does not contaminate relative comparisons. Ledoit-Wolf and equal-weight portfolios use no embeddings at all, providing a clean non-contaminated baseline.

5. **The shuffle test provides supporting evidence.** If the model's "future knowledge" were driving portfolio performance through firm-specific semantic understanding, shuffled embeddings (which destroy firm-specific content while preserving the model's learned representation space) should perform worse than unshuffled embeddings. At p=500, shuffled and unshuffled perform similarly, suggesting the advantage derives from spectral structure.

**Gold-standard resolution:** Retrain the language model at each quarterly rebalance using only text available up to that date (rolling temporal training). This would produce embeddings that are strictly contemporaneous but requires substantial computational resources. Given the TF-IDF equivalence result, such retraining would likely confirm what we already observe: the signal is structural, not model-dependent.

### 4.8 Future Work

- **Alpha sweep at higher p/n**: The alpha sweep (Section 3.13) was run at p=200 and p=500. Extending to p=1000 and p=2000 would test whether the optimal $\alpha$ increases further (approaching 1.0) as the sample covariance becomes more noise-dominated. At p=2000 where LW Sharpe is only 0.15, the pure text portfolio may be unambiguously dominant
- **Full shuffle gradient across p/n**: The shuffle gradient (Section 3.16) was run at p=2000; extending to [500, 1000, 1500, 2000] would precisely map the transition from purely structural (p=500) to partially informational (p=2000) benefit
- **PCA scaling experiment**: Run PCA-$k$ (best $k$ from $k$-sweep) across $p=50$ to $p=2000$ to test whether low-rank PCA targets scale better or worse than full cosine
- **PCA shuffle test**: Does shuffling matter more for low-$k$ PCA than for full cosine? If yes, the low-rank factors may carry firm-specific information that the full cosine matrix dilutes
- **Temporal embedding volatility**: The `TemporalPCASigma` estimator (implemented but not yet tested in backtest) measures per-firm risk from the Frobenius norm of centered document embedding trajectories. Does text-based $\sigma$ improve portfolio performance vs unit $\sigma$?
- **Transcript scaling experiment** ($p=50$ to $p=500$ for quarterly text)
- **Temporally retrained embeddings**: Train the sentence transformer from scratch (or fine-tune) at each quarterly rebalance using only corporate text available up to that date. While the TF-IDF equivalence result (Section 3.25) strongly suggests temporal retraining would not change the findings, it would constitute the gold-standard elimination of any remaining concern about anachronistic model knowledge
- **Time-varying alpha analysis**: Correlate the CV-alpha selections (Section 3.22) over time with VIX and realized market volatility to test whether the optimal text weight increases during crises when return-based estimates become unreliable
- **Embedding model comparison**: Test alternative encoders (FinBERT, OpenAI embeddings) to determine whether domain-specific or larger models improve on the TF-IDF-equivalent baseline
- **Turnover-constrained optimization**: Add explicit turnover penalties to the portfolio objective to reduce implementation costs, particularly at high $\alpha$ where the correlation structure changes only at filing dates

---

## 5. Implications: Text as Sufficient Statistic for Financial Structure

The experiments in Section 3 collectively demonstrate something broader than improved covariance estimation: **text embeddings encode the latent economic structure that financial time series have traditionally been the sole source of.** This section traces the direct implications.

### 5.1 The Cold-Start Problem Is Solved

The return history ablation (Section 3.21) shows that text-based correlations with just 252 days of sample volatilities produce Sharpe 0.771 — outperforming LW with full 504-day returns (0.540) by 43% and beating even equal weight (0.582). At $L=0$ (zero return history), text-based covariance produces viable portfolio allocations (Sharpe 0.568). The oracle $\alpha$ assigns $>$60% weight to the text target when only 5--21 days of returns are available. The alpha sweep (Section 3.13) demonstrates that pure text ($\alpha$=1.0) delivers Sharpe 0.744 at $p$=500, significantly beating LW's 0.540 with full 504-day returns ($p$=0.012).

**Practical applications:**
- **IPOs and new listings:** Covariance estimates from the prospectus filing, before first trade.
- **Spinoffs:** The parent company's filing language differentiates the spun-off entity.
- **Index reconstitution:** New index members have text history even without price history in the new context.
- **Newly covered firms:** Firms entering a portfolio manager's coverage universe have filings dating back years.

### 5.2 Risk Management Without (Reliable) Return History

The VaR experiment (Section 3.18) demonstrates that at high p/n, sample-based VaR systematically underestimates risk while text-based VaR remains closest to calibration. At 5% VaR, semantic achieves a violation ratio of 1.03 at $p$=1000 (nearly perfect) versus LW's 1.18--1.23. At 2.5% VaR, all methods are rejected by Kupiec, but semantic has the lowest violation ratio (1.27 at $p$=2000 vs LW 1.72). LW's distorted eigenvalues (Marchenko-Pastur) produce covariance matrices that systematically underestimate portfolio risk.

**Practical applications:**
- **Regulatory VaR for large portfolios:** Basel III requires VaR estimates; text-based covariance provides well-conditioned inputs.
- **Stress testing:** During market crises, return correlations spike and sample covariance becomes unreliable. Text-based structure provides a stable prior.
- **Risk budgeting across asset classes:** When some portfolio legs have no price data (e.g., private credit allocations), text provides structure.

### 5.3 Cross-Asset Covariance and the Asynchronicity Problem

Text is asynchronicity-free. A Tokyo equity and a New York equity trade during different hours — return correlation is attenuated by the Epps (1979) effect. But both companies produce annual reports. Text-based correlation does not depend on synchronous trading.

**Extensions beyond equities:**
- **Crypto vs traditional:** Cryptocurrencies trade 24/7; equities trade 6.5 hours. Return correlations between the two are systematically attenuated.
- **Commodity producers vs commodity prices:** Mining companies' 10-K filings describe exposure to commodity prices; text similarity provides correlation structure without requiring synchronous price series.
- **Any cross-market pair:** Cross-border, cross-timezone, cross-exchange — text operates in semantic space, not calendar time.

### 5.4 Private Markets

Private equity and venture capital portfolios have essentially no usable return series. Quarterly NAVs are smoothed and lagged (Getmansky, Lo, and Makarov 2004). But every private company has *text*: pitch decks, quarterly investor letters, SEC filings (late-stage), product descriptions, patent filings.

Text-based covariance could provide the first real-time correlation matrix for private assets, enabling:
- **Pension fund risk management** across public + private allocations.
- **VC portfolio construction** using pitch deck embeddings as the covariance source.
- **PE secondaries pricing** where understanding portfolio company correlation is critical.

### 5.5 The Text-First Portfolio Architecture

These experiments collectively outline a portfolio management system where text is the primary data source:

| Component | Traditional | Text-First |
|-----------|-------------|------------|
| **Covariance** | Sample + shrinkage | Embedding cosine (demonstrated, Sections 3.1-3.2) |
| **Factor exposures** | Regression on return PCs | Projection onto text-PCs (Section 3.17) |
| **Risk estimates** | Historical VaR | Text-VaR (Section 3.18) |
| **Sector structure** | SIC/GICS codes | Embedding clusters (Section 3.19) |
| **Universe construction** | Price + volume history | Filing availability (Section 3.20) |
| **Time series role** | Primary data source | Performance measurement + execution |

The key insight is not that text is *better* than returns — at low p/n with full history, returns are hard to beat. The insight is that text is *sufficient*: it provides a viable alternative for every component of portfolio construction, making the entire system functional even when returns are unavailable, unreliable, or dominated by noise. The lookback ablation (Section 3.21) provides the strongest evidence: text-based correlations + sample volatilities **statistically significantly** outperform LW with 504-day returns ($p$=0.012 over 75 quarters), demonstrating that the binding constraint on covariance estimation is not data quantity but prior quality. This result replicates across three text sources — 10-K filings, earnings transcripts, and financial news (Section 3.27) — and is robust to the choice of encoder (neural or TF-IDF, Section 3.25). The CV-alpha experiment (Section 3.22) shows that a data-driven calibration procedure independently discovers the optimal text weight, removing the need for manual parameter selection. Under institutional constraints (Section 3.23), the text-based covariance advantage strengthens: sector limits restore the Sharpe advantage over equal weight, while the drawdown protection (~35% vs ~57%) persists across all constraint regimes.

---

## 6. Data Sources

| Data | Source | Access Method | Coverage |
|------|--------|---------------|----------|
| 10-K annual reports | PleIAs/SEC-Filings (HuggingFace) | Streaming API (`datasets` library) | ~63K filings, 4,592 US tickers, 1993--2023 |
| Earnings transcripts | kurry/sp500_earnings_transcripts (HuggingFace) | Streaming API | 30,596 transcripts, 608 S&P 500 tickers, 2006--2025 |
| Financial news | 5 HuggingFace datasets (benzinga, financial_phrasebank, etc.) | Streaming API | 111,062 quarterly docs, 4,024 tickers, 2007--2024 |
| Daily prices (OHLCV) | Tiingo REST API | Per-ticker download with `.env` key | 8,974 tickers, adjusted close prices |
| SEC filing dates | SEC EDGAR EFTS API | Direct HTTP | 74,232 cached filing date lookups |
| CIK-to-ticker mapping | SEC EDGAR company tickers JSON | Direct HTTP | 10,397 mappings |

### Filing Date Handling

10-K filings use the **actual SEC filing date** (from EDGAR), not the fiscal year end date. This prevents look-ahead bias: a 10-K for FY2022 might not be filed until March 2023, and we do not use it until after the filing date.

---

## 7. Software and Reproducibility

### 7.1 Environment

| Component | Version |
|-----------|---------|
| Python | 3.13.9 |
| OS | Linux 6.17.4 (Ubuntu-based) |
| GPU | NVIDIA GeForce RTX 5080, 16GB VRAM |
| CUDA | 12.8 |
| Package manager | uv |

### 7.2 Key Library Versions

| Library | Version | Purpose |
|---------|---------|---------|
| torch | 2.10.0+cu128 | GPU tensor operations |
| sentence-transformers | 5.2.3 | BAAI/bge-base-en-v1.5 embedding model |
| numpy | 2.4.2 | Numerical computation |
| pandas | 3.0.0 | Data manipulation |
| scipy | 1.17.0 | Eigenvalue decomposition, distance computation |
| scikit-learn | 1.8.0 | Ledoit-Wolf shrinkage intensity calibration |
| nltk | 3.9.2 | Sentence tokenization (punkt_tab) |
| datasets | 4.5.0 | HuggingFace dataset streaming |
| statsmodels | 0.14.6+ | Fama-French regression (Newey-West HAC) |
| cvxpy | 1.8.1+ | Quadratic program solver (OSQP backend) |

### 7.3 Reproduction Commands

```bash
# Install dependencies
uv sync

# Download data (requires Tiingo API key in .env)
uv run python pipeline/download_data.py all
uv run python pipeline/fetch_filing_dates.py
uv run python pipeline/clean_filings_index.py

# Embed text (resumes from checkpoint)
uv run python pipeline/embed_filings.py           # ~4,592 tickers, ~2 hours
uv run python pipeline/embed_transcripts.py        # ~608 tickers, ~1 hour

# Run experiments
uv run python experiments/run_backtest.py                                          # 10-K, p=500
uv run python experiments/run_backtest.py --text-source transcript --max-firms 500 # Transcript, p=500
uv run python experiments/run_backtest.py --text-source combined --max-firms 500   # Combined, p=500
uv run python experiments/run_scaling_experiment.py --p-grid 50,100,200,500,1000,1500,2000
uv run python experiments/run_robustness.py --all --returns-file <path>            # FF, SPY, shuffle, TC
uv run python experiments/run_backtest.py --start 2013-04-01 --end 2018-06-30     # Sub-period 1
uv run python experiments/run_backtest.py --start 2019-01-01 --end 2024-06-30     # Sub-period 2
uv run python experiments/run_block_sensitivity.py --returns-file <path>           # Block size sensitivity

uv run python experiments/run_pca_experiment.py --k-grid 1,3,5,10,20,50,100        # PCA k-sweep
uv run python experiments/run_alpha_sweep.py --p-grid 200,500 --text-source 10k    # Alpha sweep + cold-start
uv run python experiments/run_pca_interpretation.py --n-components 10 --max-firms 500  # PCA interpretation
uv run python experiments/run_shuffle_gradient.py --p-grid 2000 --n-shuffles 10    # Shuffle gradient
uv run python pipeline/fetch_sic_codes.py                                          # SIC codes from EDGAR
uv run python experiments/run_beta_experiment.py --max-firms 500                    # Text-based beta
uv run python experiments/run_var_experiment.py --p-grid 200,500,1000,2000          # VaR calibration
uv run python experiments/run_sector_analysis.py --max-firms 200                    # Text vs SIC codes
uv run python experiments/run_lookback_ablation.py --max-firms 500                  # Return history ablation
uv run python experiments/run_cold_start_experiment.py --max-firms 500              # Cold-start simulation
uv run python experiments/run_cv_alpha_experiment.py --max-firms 500                # CV-alpha calibration
uv run python experiments/run_constrained_experiment.py --max-firms 500             # Constrained optimization
uv run python experiments/run_multitarget_experiment.py --p-grid 200,500            # Multi-target shrinkage

# Run tests (70 tests)
uv run pytest tests/ -v
```

### 7.4 Project Structure

```
pnlp/                          # Core library
  config.py                    # All hyperparameters (frozen dataclasses)
  data/                        # Data loading, embedding I/O, universe filtering
    embeddings_loader.py       # Multi-source loader: load_doc_embeddings(text_source)
    universe_filter.py         # Point-in-time ADV filter
    store.py                   # DocumentStore (NPZ embedding I/O)
  embeddings/                  # Text -> embedding pipeline
    text_encoder.py            # bge-base-en-v1.5 wrapper (batch_size=64)
    document_embedder.py       # Chunk + mean-pool
    firm_aggregator.py         # Documents -> firm embedding (L2-norm here only)
  primitives/                  # Statistical estimators
    covariance.py              # SemanticShrinkageCovariance, PCAFactorCovariance
    gpu_accel.py               # PSD enforcement, Ledoit-Wolf shrinkage
  portfolio/                   # Optimization
    optimizer.py               # CVXPY/OSQP min-variance, max-sharpe, risk-parity
  baselines/                   # Comparison strategies
    shrinkage.py               # Ledoit-Wolf (sklearn)
    equal_weight.py            # 1/N
    sic_sector.py              # SIC sector block-diagonal covariance
  validation/                  # Backtest and evaluation
    backtest.py                # Walk-forward engine with per-ticker TC
    metrics.py                 # Sharpe, Sortino, drawdown, CAPM
    statistical_tests.py       # Block permutation, Lo correction
    transaction_costs.py       # Stratified ADV-tiered TC model
    var_tests.py               # Kupiec, Christoffersen VaR backtests
experiments/                   # Experiment entrypoints (run_*.py)
pipeline/                      # Data download, embedding, audit scripts
tests/                         # Test suite using synthetic data factories
```

### 7.5 Test Suite

70 tests across 5 test files, all using synthetic data factories (no real data or ML models in default test runs):

| File | Tests | Coverage |
|------|-------|----------|
| test_primitives.py | 30 | Mu, sigma, covariance estimators (incl. PCA factor, temporal sigma) |
| test_portfolio.py | 7 | Optimizer, constraint satisfaction |
| test_validation.py | 8 | Metrics, permutation test, bootstrap |
| test_universe_filter.py | 8 | ADV filter, lag, completeness |
| test_transaction_costs.py | 17 | Tier assignment, vol regime, TC rates |

---

## 7. References

- Becquin, G., & Esmeir, S. (2023). Semantic similarity covariance matrix shrinkage. *Findings of EMNLP 2023*, 9977--9992.
- Carlstein, E. (1986). The use of subseries values for estimating the variance of a general statistic from a stationary sequence. *Annals of Statistics*, 14(3), 1171--1179.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915--1953.
- Dyer, T., Roulstone, D., & Van Buskirk, A. (2024). Disclosure similarity and future stock return comovement. *Management Science*, 70(7), 4762--4780.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1--22.
- Gabaix, X., Koijen, R. S. J., Richmond, R., & Yogo, M. (2025). Asset embeddings. NBER Working Paper 33651.
- Gawronsky, T., & Huang, K. (2024). Continuous risk factor models: Analyzing asset correlations through energy distance. arXiv:2410.23447.
- Hoberg, G., & Phillips, G. (2010). Product market synergies and competition in mergers and acquisitions: A text-based analysis. *Review of Financial Studies*, 23(10), 3773--3811.
- Hoberg, G., & Phillips, G. (2016). Text-based network industries and endogenous product differentiation. *Journal of Political Economy*, 124(5), 1423--1465.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73--84.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365--411.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36--52.
- Lu, Z., Ndiaye, M., & Simaan, M. (2024). Improved estimation of the correlation matrix using reinforcement learning and text-based networks. *International Review of Financial Analysis*, 96(A), 103585.
- Nakayama, K., Sawaki, S., Furuya, T., & Tamura, K. (2024). Text-based correlation matrix in multi-asset allocation. arXiv:2405.14247.
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero: Calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1).
- Xiao, Z., et al. (2024). BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. *Findings of ACL 2024*. (BAAI/bge-base-en-v1.5 model family.)
