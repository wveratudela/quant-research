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
├── Q2_mean_reversion/               ← Coming soon
│   └── ...
│
├── Q3_portfolio_optimisation/       ← Coming soon
│   └── ...
│
├── Q4_mixed_asset_portfolio/        ← Coming soon
│   └── ...
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
| Q2 | Mean Reversion & Pairs Trading | 🔄 In progress | Cointegration, z-score signals, spread trading |
| Q3 | Multi-Asset Portfolio Optimisation | 📋 Planned | Markowitz, Sharpe maximisation, Magnificent 7 vs SPY |
| Q4 | Mixed Asset Class Portfolio | 📋 Planned | Stocks + crypto + ETFs, rebalancing strategies |
| Q5 | ML Signal Generation | 📋 Planned | Feature engineering, classification, signal prediction |

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

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Jupyter Notebook | Research environment |
| pandas & numpy | Data manipulation and numerical computing |
| yfinance | Market data retrieval |
| matplotlib & seaborn | Visualisation |
| scikit-learn | ML (Q5, coming soon) |

---

## Disclaimer

This repository is for educational and research purposes only. Nothing here constitutes financial advice. All strategies are tested on historical data, and past performance does not guarantee future results.
