# Quant Research

Signal generation, portfolio construction, and risk measurement.
Research-first, honest about failures.

**github.com/wveratudela/quant-research** · *Dr. Walter Vera-Tudela*

---

## Overview

Five projects covering the core empirical toolkit of systematic finance: trend-following, statistical arbitrage, portfolio optimisation, tail risk measurement, and ML-based signal generation. Each project is self-contained, implemented from first principles in Python, and includes a critical analysis of results — including where the more complex approach failed to justify itself.

These are not demonstrations that a strategy works. They are honest stress tests of whether it does, and why.

---

## Repository Structure
```
quant-research/
│
├── README.md                              ← You are here
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
├── Q3_portfolio_optimisation/
│   ├── Q3_notebook.ipynb
│   ├── Q3_functions.py
│   └── README.md
│
├── Q4_tail_risk_stress_testing/
│   ├── Q4_notebook.ipynb
│   ├── Q4_functions.py
│   └── README.md
│
├── Q5_ml_signal_generation/
│   ├── Q5_notebook.ipynb
│   ├── Q5_functions.py
│   └── README.md
│
└── utils/
    └── common.py                          ← Shared utilities (data fetching, metrics)
```

---

## Project Roadmap

| # | Project | Status | Key Concept |
|---|---------|--------|-------------|
| Q1 | Momentum Backtester | ✅ Complete | MA crossover, signal generation, performance metrics |
| Q2 | Mean Reversion & Pairs Trading | ✅ Complete | Cointegration, z-score signals, spread trading |
| Q3 | Multi-Asset Portfolio Optimisation | ✅ Complete | Markowitz, Black-Litterman, cross-asset diversification |
| Q4 | Tail Risk & Stress Testing | ✅ Complete | Monte Carlo VaR, historical crisis replay, tail decomposition |
| Q5 | ML Signal Generation | ✅ Complete | Feature engineering, XGBoost classification, walk-forward validation |

---

## Projects

### Q1 — Momentum Backtester
`Python · yfinance · pandas · matplotlib`

MA20/MA50 Golden/Death Cross on AAPL over 10 years with tranched entry, benchmarked against buy-and-hold AAPL and SPY.

| Metric | MA Crossover | Buy & Hold AAPL | Buy & Hold SPY |
|---|---|---|---|
| Total Return | 327% | 951% | 246% |
| Sharpe Ratio | 0.53 | 0.68 | 0.47 |
| Max Drawdown | **−28.9%** | −38.7% | −34.1% |
| CAGR | 15.7% | 26.5% | 13.3% |

**Finding:** The crossover strategy underperforms buy-and-hold AAPL in sustained bull markets but delivers meaningful drawdown protection in crisis regimes (2020: strategy down 12% vs benchmark down 38%). This is the baseline all subsequent projects are measured against.

---

### Q2 — Mean Reversion & Pairs Trading
`Python · statsmodels · scipy · matplotlib`

Statistical arbitrage on Visa/Mastercard using ADF + Engle-Granger cointegration (p = 0.0006), OLS hedge ratio β ≈ 0.55 (R² ≈ 0.99), and z-score signals (±2σ entry, z = 0 exit). Dollar-neutral construction.

| Metric | Pairs V/MA | Buy & Hold V | Buy & Hold MA | Buy & Hold SPY |
|---|---|---|---|---|
| Total Return | 58% | 345% | 492% | 242% |
| Sharpe Ratio | 0.13 | 0.45 | 0.52 | 0.46 |
| Max Drawdown | **−2.1%** | −36.4% | −41.0% | −34.1% |
| CAGR | 4.7% | 16.1% | 19.5% | 13.1% |

**Finding:** The strategy delivers modest absolute returns but near-zero drawdown across a decade including the 2020 crash — roughly 17× better capital preservation than directional alternatives. This is not a return maximiser; it is a capital preservation tool. In a leveraged institutional context, stable low-volatility returns become commercially significant.

---

### Q3 — Multi-Asset Portfolio Optimisation
`Python · cvxpy · numpy · scipy`

Markowitz mean-variance optimisation and Black-Litterman across 10 assets spanning equities, crypto, gold, bonds, and international ETFs (~8 years). Extends Q1/Q2 to a full multi-asset universe and tests the limits of both frameworks.

| Portfolio | Final Value ($10k) | Volatility |
|---|---|---|
| Equal Weight | ~$65,000 | 26.1% |
| Target Return Optimised | ~$95,000 | 14.4% |
| Minimum Variance | ~$20,000 | **7.0%** |
| Maximum Sharpe | ~$130,000 | — |
| Black-Litterman | ~$145,000 | 26.1% |

**Finding:** Cross-asset diversification drops minimum variance portfolio volatility from 0.24 (Mag 7 only, single sector) to 0.07 — a 3× reduction driven by near-zero-correlation assets (GLD, TLT). Source of diversification matters more than number of assets. Black-Litterman proved structurally misaligned with mixed-asset universes: its market-cap equilibrium prior systematically suppresses GLD and TLT, neutralising most of the diversification benefit. A proper implementation requires asset-class-specific priors.

---

### Q4 — Tail Risk & Stress Testing
`Python · numpy · scipy · matplotlib`

Monte Carlo VaR, historical crisis replay (COVID-19 + 2022 rate shock), and per-asset tail loss decomposition on the same 10-asset universe as Q3.

| Scenario | MC VaR Predicted | Actual Loss |
|---|---|---|
| COVID-19 (Mar 2020) | 7.8% | **25.5%** |
| 2022 Rate Shock | 7.8% | **37.4%** |

**TLT contribution to tail loss:** 0.06% in COVID → 60% in 2022.

**Finding:** MC VaR systematically underestimates tail risk in structural break scenarios — it models volatility, not regime change. The same asset (TLT) provided near-perfect crisis protection in 2020 and became the dominant loss driver in 2022. There is no unconditional safe haven; asset behaviour is regime-dependent, not asset-class-dependent.

---

### Q5 — ML Signal Generation
`Python · scikit-learn · XGBoost · pandas`

Walk-forward supervised classification on AAPL (10 years) testing whether ML complexity improves on a simple baseline. Three feature sets: price-only, price + volume, price + volume + cross-asset. Models: Logistic Regression, Random Forest, XGBoost.

| Strategy | Final Value ($10k) |
|---|---|
| Price only (XGBoost) | ~$22,000 |
| Price + Volume (XGBoost) | ~$30,000 |
| Price + Volume + Cross-asset | ~$25,000 |
| **MA Crossover baseline (Q1)** | **~$41,000** |
| Buy & Hold | ~$110,000 |

**Finding:** Volume-confirmed features outperform price-only by 36%, with Volume Ratio as the single most predictive feature above all price inputs. However, no model exceeded 57.7% naive baseline accuracy, and all underperformed the simple MA crossover in backtested returns. Cross-asset features diluted rather than improved signal quality. For trending assets in sustained bull markets, momentum rules remain more effective than supervised classification.

---

## Research Positioning

> *"I design quantitative trading and portfolio systems that remain stable under market regime transitions."*

Q1–Q2 establish signal generation and statistical arbitrage foundations. Q3–Q4 develop multi-asset optimisation with explicit tail risk and regime analysis. Q5 bridges classical signals and ML and documents the limits of the latter.

The engineering-to-finance extension of this work — applying control theory and dynamical systems methods to portfolio construction — is developed in the companion **M repo**.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| pandas & numpy | Data manipulation and numerical computing |
| yfinance | Market data retrieval |
| statsmodels | Cointegration, ADF testing |
| scikit-learn & XGBoost | ML models, walk-forward validation |
| cvxpy & scipy | Portfolio optimisation (SLSQP, convex) |
| matplotlib & seaborn | Visualisation |

All projects implemented from first principles with strict walk-forward validation. No lookahead bias.

---

## Related Repositories

| Repo | Focus |
|---|---|
| [F — Financial Engineering](https://github.com/wveratudela/financial-engineering) | Derivatives pricing, Greeks, yield curve, volatility surface |

---

*Dr. Walter Vera-Tudela · [github.com/wveratudela](https://github.com/wveratudela)*

---

## Disclaimer

This repository is for educational and research purposes only. Nothing here constitutes financial advice. All strategies are tested on historical data, and past performance does not guarantee future results.
