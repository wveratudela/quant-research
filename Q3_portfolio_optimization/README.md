# Q3 — Portfolio Optimisation

A multi-asset portfolio optimisation framework built from scratch, implementing Markowitz Mean-Variance Optimisation and the Black-Litterman model on the Magnificent 7 stocks.

---

## Objective

Given a portfolio of 7 large-cap US tech stocks, find the optimal allocation weights that maximise risk-adjusted returns. Move beyond equal-weight allocation using two frameworks: classical Markowitz optimisation (data-driven) and Black-Litterman (view-driven). Evaluate all portfolios against SPY as a benchmark.

---

## Strategy Logic

**Equal Weight Baseline**
- 14.3% allocation to each of the 7 assets
- No optimisation — serves as the benchmark to beat

**Markowitz Mean-Variance Optimisation**
- Compute annualised returns, volatility, and covariance matrix from 5 years of daily data
- Trace the efficient frontier by minimising portfolio variance for 100 target return levels
- Identify three key portfolios: minimum variance (MVP), maximum Sharpe, and target-return optimised (same return as equal weight, minimum volatility)
- Monte Carlo simulation: 10,000 Dirichlet random portfolios to visualise the feasible region
- Capital Market Line drawn from risk-free rate (4%) through the maximum Sharpe portfolio

**Black-Litterman Model**
- Compute market cap weights for all 7 assets
- Derive equilibrium returns: `π = λ × Σ × w_market` with risk aversion λ = 2.5
- Encode personal view: AAPL, GOOGL, MSFT, NVDA outperform AMZN, META, TSLA by 5-10%
- Compute posterior returns `μ_BL` blending equilibrium and views via Bayes' theorem
- Rerun optimisation on posterior returns
- Sensitivity test: high vs low confidence in views

---

## Assets

**Magnificent 7:** AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA

Selected as a representative large-cap tech portfolio. All assets share positive correlations (0.34–0.64), confirming limited diversification within this universe — a key finding that motivates Q4's mixed asset class approach.

---

## Files

| File | Description |
|------|-------------|
| `Q3_notebook.ipynb` | Main research notebook with analysis and results |
| `Q3_functions.py` | Modular functions: `fetch_portfolio_data()`, `market_cap()`, `returns_volatility()`, `frontier_optimizer()`, `max_sharpe()`, `black_litterman()`, `plot_frontier()`, `plot_performance()` |

---

## Key Results

### Portfolio Comparison (5 years, $10,000 starting capital)

| Portfolio | Final Value | Return | Volatility | Sharpe |
|-----------|-------------|--------|------------|--------|
| Equal Weight | ~$40,000 | ~300% | 29.9% | 1.09 |
| Target Return Optimised | ~$45,000 | ~350% | 26.6% | — |
| Minimum Variance | ~$21,000 | ~110% | 24.2% | — |
| Max Sharpe (100% NVDA) | ~$153,000 | ~1,430% | 51.8% | 1.83 |
| Black-Litterman | ~$49,000 | ~390% | 24.2% | 0.48 |
| SPY (benchmark) | ~$17,000 | ~70% | — | 0.46 |

### Optimised Weights — Key Portfolios

| Asset | Equal | Target Return | Min Variance | Black-Litterman |
|-------|-------|---------------|--------------|-----------------|
| AAPL | 14.3% | 34.2% | 36.8% | 18.8% |
| AMZN | 14.3% | 0.0% | 0.0% | 11.4% |
| GOOGL | 14.3% | 26.6% | 14.4% | 17.9% |
| META | 14.3% | 0.0% | 0.0% | 8.1% |
| MSFT | 14.3% | 21.9% | 48.9% | 15.1% |
| NVDA | 14.3% | 17.29% | 0.0% | 21.4% |
| TSLA | 14.3% | 0.0% | 0.0% | 7.4% |

---

## Key Findings

**Diversification through low correlation beats individual asset quality.** META appeared in nearly every optimised portfolio despite being a subjective underperformer — its low correlation with other assets (0.34–0.44) contributes genuine variance reduction. An asset's portfolio value is not determined by its standalone performance.

**The efficiency principle eliminates high-volatility, low-return assets.** TSLA and AMZN were consistently zeroed out. Both sit inside the efficient frontier individually — meaning a combination of other assets can achieve the same return with lower risk.

**Raw Markowitz is dangerously overfit to historical data.** The unconstrained Sharpe maximisation allocated 100% to NVDA — a direct consequence of NVDA's exceptional 5-year AI-driven run. This illustrates estimation error: small changes in expected returns produce wildly different optimal weights. This is the most famous weakness of mean-variance optimisation.

**Black-Litterman produces more robust, diversified portfolios.** By anchoring to market equilibrium returns rather than raw history, BL reduces NVDA's implied return from 99% to 27% — a far more conservative and defensible estimate. With high confidence in the outperformer view, the portfolio concentrates in AAPL and GOOGL rather than NVDA, reflecting both the view and risk efficiency.

**The portfolio value comparison is illustrative, not predictive.** Weights are optimised on the same data used to evaluate performance. A proper out-of-sample evaluation would require rolling window optimisation and forward testing.

---

## Limitations

1. **Estimation error** — optimal weights are highly sensitive to expected return inputs. Small changes in historical returns produce dramatically different portfolios. The 100% NVDA result from raw Markowitz illustrates this failure mode clearly. Black-Litterman partially addresses this but does not eliminate it.

2. **Static weights** — the optimised portfolio is calculated once and held fixed. Correlations and volatilities change over time. A portfolio optimal today may be inefficient in 6 months. A rolling or dynamic rebalancing framework is needed for real deployment.

3. **In-sample optimisation** — weights are derived from and evaluated on the same historical period. The portfolio value comparison is illustrative of framework behaviour, not a prediction of future performance.

---

## Next Steps

- Implement rolling window optimisation — recalculate weights quarterly using only past data
- Add transaction costs — frequent rebalancing erodes returns meaningfully
- Extend to mixed asset classes — stocks, crypto, and ETFs (Q4)
- Explore Lyapunov stability-based dynamic rebalancing as a novel contribution (Q6)

---

## Dependencies

```
pandas, numpy, yfinance, matplotlib, seaborn, scipy
```
