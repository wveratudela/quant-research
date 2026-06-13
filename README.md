# Quant Research

A growing collection of quantitative finance research projects built from scratch in Python.

I am a PhD researcher in mechanical engineering, repositioning my skills into quantitative finance. This repository documents that journey — starting from first principles and progressively building more sophisticated tools and strategies. Each project is self-contained, fully documented, and includes a critical analysis of results, not just the wins.

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
    └── common.py                    ← Shared utilities (data fetching, metrics)
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

## Key Findings Summary

### Q1 — Momentum Backtester (AAPL, 10 years)

A MA20/MA50 crossover strategy tested on AAPL over 10 years, compared against buy-and-hold AAPL and the S&P 500.

| Metric | Buy/Sell | Buy & Hold AAPL | Buy & Hold SPY |
|--------|----------|-----------------|----------------|
| Total Return | 327% | 951% | 246% |
| Sharpe Ratio | 0.53 | 0.68 | 0.47 |
| Max Drawdown | -28.9% | -38.7% | -34.1% |
| CAGR | ~15.7% | ~26.5% | ~13.3% |

**Main takeaway:** The MA crossover strategy significantly underperforms buy-and-hold in sustained bull markets but provides meaningful drawdown protection during crises (e.g. 2020). It functions as a bear market defence mechanism rather than a return maximiser. Survivorship bias, transaction costs, and lookahead bias are acknowledged limitations.

---

### Q2 — Pairs Trading (Visa & Mastercard, 10 years)

A market-neutral pairs trading strategy on V/MA, tested over 10 years.

| Metric | Pairs V/MA | Buy & Hold V | Buy & Hold MA | Buy & Hold SPY |
|--------|------------|--------------|---------------|----------------|
| Total Return | 58% | 345% | 492% | 242% |
| Sharpe Ratio | 0.13 | 0.45 | 0.52 | 0.46 |
| Max Drawdown | -2.1% | -36.4% | -41.0% | -34.1% |
| CAGR | 4.7% | 16.1% | 19.5% | 13.1% |

**Main takeaway:** The pairs strategy delivers modest absolute returns but with a near-zero maximum drawdown of -2.1% — roughly 17× better capital preservation than buy-and-hold alternatives. The 2020 crash that wiped 30–40% from all directional strategies left the pairs portfolio virtually untouched. The strategy is not a return maximiser — it is a capital preservation tool, and in a leveraged institutional context, the stable low-volatility returns become commercially significant.

---

### Q3 — Multi-Asset Portfolio Optimisation (Stocks + Crypto + ETFs, ~8 years)

Markowitz mean-variance optimisation and Black-Litterman model applied to a 10-asset universe spanning equities, crypto, gold, bonds, and international ETFs.

| Portfolio | Final Value ($10k) | Volatility | Key Weights |
|-----------|-------------------|------------|-------------|
| Equal Weight | ~$65,000 | 26.1% | 10% each |
| Target Return Optimised | ~$95,000 | 14.4% | GLD 67%, NVDA 18% |
| Minimum Variance | ~$20,000 | 7.0% | TLT 42%, GLD 27%, SPY 25% |
| Maximum Sharpe | ~$130,000 | — | GLD 60%, NVDA 29% |
| Black-Litterman | ~$145,000 | 26.1% | NVDA 35%, AAPL 30%, BTC 9% |

**Main takeaway:** Cross-asset diversification is fundamentally more powerful than sector diversification. Adding gold and long-term Treasuries — both with near-zero or negative correlations to equities — dropped minimum variance portfolio volatility from 0.24 (equities only) to 0.07. The source of diversification matters more than the number of assets. Black-Litterman as standardly implemented proved structurally misaligned with mixed-asset universes: its market-cap equilibrium prior systematically suppresses GLD and TLT, and a proper implementation requires asset-class-specific priors.

> **Note on single-sector optimisation:** Running the same Markowitz framework on a correlated sector (the Magnificent 7) produces a degenerate result — unconstrained Sharpe maximisation collapses to 100% NVDA, exposing the core weakness of mean-variance optimisation: estimation error amplifies any asset with a historically high return. Within a single correlated sector, the optimizer eliminates every asset that cannot independently justify its volatility with sufficient return. Cross-asset diversification is what makes optimisation meaningful.

---

### Q4 — Tail Risk & Stress Testing (~8 years, two crises)

A portfolio stress testing framework applying Monte Carlo simulation, historical crisis replay, and tail risk decomposition to the same 10-asset universe as Q3. Min-variance weights from Q3 are tested against equal-weight allocation across two structurally different crises.

| Scenario | VaR 95% | ES 95% | Character |
|---|---|---|---|
| MC Normal | 7.8% | 11.0% | Baseline forward-looking |
| MC Student-t (ν=4) | 10.7% | 17.7% | Fat tails, marginal improvement |
| COVID-19 (equal weights) | 25.5% | 26.5% | Sharp crash, fast recovery |
| 2022 Rate Shock (equal weights) | 37.4% | 38.4% | Sustained grind, no recovery |

**Min-variance weights across crises:**

| Crisis | VaR 95% | Reduction vs Equal | Primary loss driver |
|--------|---------|-------------------|---------------------|
| COVID-19 | 8.9% | −65% | SPY (63%) |
| 2022 Rate Shock | 22.2% | −41% | **TLT (60%)** |

**Main takeaway:** Monte Carlo VaR massively underestimates real crisis severity — normal-distribution MC predicted 7.8% VaR; COVID delivered 25.5%, the 2022 rate shock delivered 37.4%. Even Student-t fat tails barely moved the needle. Historical scenario replay is the only honest tail risk measure. The deeper finding is regime dependence: TLT contributed 0.06% of tail losses during COVID (functioning exactly as a safe haven), then became the single largest loss driver in 2022 with a 60% contribution — a 1000× flip in risk profile driven by the Fed's rate hiking cycle. Diversification assumptions must be regime-conditioned. There is no unconditional safe haven.

---

### Q5 — ML Signal Generation (AAPL, 10 years)

Walk-forward ML signal generation testing the hypothesis that volume-confirmed features outperform price-only features. Three XGBoost models compared against a MA crossover baseline and buy-and-hold.

| Strategy | Final Value ($10k) | Notes |
|----------|-------------------|-------|
| Price only (XGBoost) | ~$22,000 | Below MA crossover |
| Price + Volume (XGBoost) | ~$30,000 | +36% vs price only |
| Price + Volume + Cross-asset | ~$25,000 | Cross-asset hurt performance |
| MA Crossover (Q1 baseline) | ~$41,000 | Simple beats complex |
| Buy & Hold | ~$110,000 | Bull market dominates |

**Main takeaway:** The volume hypothesis is confirmed — volume-confirmed features improve ML signal quality by 36% over price-only features, with Volume Ratio ranking as the single most important predictor. However, no ML model exceeded the 57.7% naive baseline accuracy, and all underperformed the simple MA crossover in backtested returns. For trending assets in sustained bull markets, momentum-following rules remain more effective than supervised classification. Cross-asset features diluted rather than improved signal quality — local volume information outperforms global macro context for single-asset prediction.

---

## Portfolio Positioning

*"I build quantitative research tools that go beyond surface-level backtesting — measuring what breaks, under what conditions, and why."*

Q1–Q2 establish signal generation and statistical arbitrage foundations. Q3 develops multi-asset portfolio optimisation and exposes its structural limitations. Q4 stress-tests those portfolios against real crises and shows where the assumptions break down. Q5 bridges classical signals and machine learning, with honest accounting of where complexity fails to beat simplicity.

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
| scikit-learn | ML models, preprocessing, metrics |
| xgboost | Gradient boosting classifier |
| scipy | Portfolio optimisation (SLSQP) |
| cvxpy | Convex optimisation |

---

## Disclaimer

This repository is for educational and research purposes only. Nothing here constitutes financial advice. All strategies are tested on historical data, and past performance does not guarantee future results.