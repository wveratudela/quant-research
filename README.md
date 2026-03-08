# Quant Research

A growing collection of quantitative finance research projects built from scratch in Python.

I am a PhD researcher in mechanical engineering, repositioning my skills into quantitative finance. This repository documents that journey — starting from first principles and progressively building more sophisticated tools and strategies. Each project is self-contained, fully documented, and includes a critical analysis of results, not just the wins.

This is a work in progress. Projects will be added regularly as the research develops.

---

## Background & Motivation

My background is in numerical modelling, optimisation, and scientific computing (Python, C++, MATLAB). Quantitative finance is a natural extension of those skills into a domain I find genuinely compelling — markets are complex systems with noisy signals, and the challenge of extracting meaning from that noise is exactly the kind of problem I enjoy.

The goal of this repo is not to find a magic strategy. It is to build rigorous, honest research habits: test on real data, measure the right things, understand *why* something works, and be equally curious about the failures.

---

## Repository Structure

```
quant-research/
│
├── README.md                        ← You are here
│
├── Q1_momentum_backtester/
│   ├── Q1_notebook.ipynb
│   ├── Q1_functions.py
│   └── README.md
│
├── Q2_pairs_trading/
│   ├── Q2_notebook.ipynb
│   ├── Q2_functions.py
│   └── README.md
│
├── Q3_portfolio_optimization/
│   ├── Q3_notebook.ipynb
│   ├── Q3_functions.py
│   └── README.md
│
├── Q4_mixed_asset_portfolio/
│   ├── Q4_notebook.ipynb
│   ├── Q4_functions.py
│   └── README.md
│
├── Q5_ml_signal_generation/         ← Coming soon
│   └── ...
│
└── utils/
    └── common.py                    ← Shared utilities (data fetching, metrics)
```

---

## Project Roadmap

| # | Project | Status | Key Concept |
|---|---------|--------|-------------|
| Q1 | Momentum Backtester | ✅ Complete | MA crossover, signal generation, performance metrics |
| Q2 | Mean Reversion & Pairs Trading | ✅ Complete | Cointegration, z-score signals, spread trading |
| Q3 | Multi-Asset Portfolio Optimisation | ✅ Complete | Markowitz, Sharpe maximisation, Magnificent 7 vs SPY |
| Q4 | Mixed Asset Class Portfolio | ✅ Complete | Stocks + crypto + ETFs, rebalancing strategies |
| Q5 | ML Signal Generation | 🔄 In progress | Feature engineering, classification, signal prediction |
| Q6 | Lyapunov stability | 📋 Planned | Novel contribution |


---

## Key Findings Summary

### Q1 — Momentum Backtester (AAPL, 10 years)

A MA20/MA50 crossover strategy tested on AAPL over 10 years, compared against buy-and-hold AAPL and the S&P 500 (SPY).

| Metric | Buy/Sell | Buy & Hold AAPL | Buy & Hold SPY |
|--------|----------|-----------------|----------------|
| Total Return | 327% | 951% | 246% |
| Sharpe Ratio | 0.53 | 0.68 | 0.47 |
| Max Drawdown | -28.9% | -38.7% | -34.1% |
| CAGR | ~15.7% | ~26.5% | ~13.3% |

**Main takeaway:** The Buy/Sell strategy significantly underperforms buy-and-hold AAPL in bull markets but offers meaningful drawdown protection during crisis periods (e.g. 2020). It functions as a bear market defence mechanism rather than a return maximiser. Survivorship bias, transaction costs, and lookahead bias are acknowledged limitations.

---

### Q2 — Pairs Trading (Visa & Mastercard, 10 years)

A market-neutral pairs trading strategy on V/MA, tested over 10 years, compared against buy-and-hold V, MA, and SPY.

| Metric | Pairs V/MA | Buy & Hold V | Buy & Hold MA | Buy & Hold SPY |
|--------|------------|--------------|---------------|----------------|
| Total Return | 58% | 345% | 492% | 242% |
| Sharpe Ratio | 0.13 | 0.45 | 0.52 | 0.46 |
| Max Drawdown | -2.1% | -36.4% | -41.0% | -34.1% |
| CAGR | 4.7% | 16.1% | 19.5% | 13.1% |

**Main takeaway:** The pairs strategy delivers modest absolute returns but with a near-zero maximum drawdown of -2.1% — roughly 17x better capital preservation than buy-and-hold alternatives. The 2020 crash that wiped 30-40% from all other strategies left the pairs portfolio virtually untouched. The strategy is not a return maximiser — it is a capital preservation tool, most valuable to risk-averse investors or as a market-neutral complement to a directional portfolio. In a leveraged institutional context, the stable low-volatility returns become commercially significant.

---

### Q3 — Portfolio Optimisation (Magnificent 7, 5 years)

Markowitz mean-variance optimisation and Black-Litterman model applied to the Magnificent 7, compared against equal-weight allocation and SPY.

| Portfolio | Final Value ($10k start) | Volatility | Sharpe |
|-----------|--------------------------|------------|--------|
| Equal Weight | ~$40,000 | 29.9% | 1.07 |
| Target Return Optimised | ~$45,000 | 26.6% | — |
| Minimum Variance | ~$21,000 | 23.7% | — |
| Max Sharpe (100% NVDA) | ~$153,000 | 51.8% | 1.80 |
| Black-Litterman | ~$49,000 | 26.6% | - |
| SPY benchmark | ~$17,000 | — | - |

**Main takeaway:** Markowitz optimisation successfully reduces portfolio volatility by 11.2% while maintaining equal-weight returns — but the unconstrained Sharpe maximisation degenerates to 100% NVDA, exposing the critical weakness of mean-variance optimisation: estimation error. Raw historical returns are a poor proxy for future expectations. Black-Litterman addresses this by anchoring to market equilibrium returns and blending in investor views with explicit confidence levels, producing more diversified and robust portfolios. The most counterintuitive finding: META appeared in nearly every optimised portfolio despite subjective underperformance, because its low correlation with other assets provides a genuine diversification benefit that the optimizer refuses to discard.

---

### Q4 — Mixed Asset Class Portfolio (Stocks + Crypto + ETFs, ~8 years)

Multi-asset portfolio optimisation across 10 assets spanning equities, cryptocurrencies, gold, bonds, and international ETFs. Extends Q3's Markowitz and Black-Litterman framework to demonstrate the value of true cross-asset diversification.

| Portfolio | Final Value ($10k) | Volatility | Key Weights |
|-----------|-------------------|------------|-------------|
| Equal Weight | ~$65,000 | 26.1% | 10% each |
| Target Return Optimised | ~$95,000 | 14.4% | GLD 67%, NVDA 18% |
| Minimum Variance | ~$20,000 | 7.0% | TLT 42%, GLD 27%, SPY 25% |
| Maximum Sharpe | ~$130,000 | — | GLD 60%, NVDA 29% |
| Black-Litterman | ~$145,000 | 14.4% | NVDA 36%, AAPL 33%, BTC 11% |

**Main takeaway:** Cross-asset diversification is fundamentally more powerful than sector diversification. Adding gold (GLD) and long-term Treasuries (TLT) — both with near-zero or negative correlations with equities — dropped the minimum variance portfolio volatility from 0.24 (Q3, Mag 7 only) to 0.07. The source of diversification matters more than the number of assets. Black-Litterman with investor views outperformed all methods (~$145k), confirming that quantitative optimisation combined with informed conviction produces better outcomes than either approach alone. Key lesson: bonds protect against recession-driven crashes but fail catastrophically during inflation — TLT lost value in 2022 while equities also fell, demonstrating that no asset is an unconditional safe haven.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Jupyter Notebook | Research environment |
| pandas & numpy | Data manipulation and numerical computing |
| yfinance | Market data retrieval |
| matplotlib & seaborn | Visualisation |
| statsmodels | Statistical tests (ADF, cointegration) |
| scikit-learn | Regression (hedge ratio), ML (Q5, coming soon) |

---

## Disclaimer

This repository is for educational and research purposes only. Nothing here constitutes financial advice. All strategies are tested on historical data, and past performance does not guarantee future results.
