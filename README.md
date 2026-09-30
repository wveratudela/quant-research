# Quant Research

Signal generation, statistical arbitrage, portfolio construction, tail-risk analysis, and machine-learning signal generation.

**Research-first. Transparent assumptions. Negative results kept.**

**github.com/wveratudela/quant-research** · *Dr. Walter Vera-Tudela*

---

## Overview

This repository contains five empirical research projects covering a compact toolkit for systematic finance:

1. trend-following and backtesting;
2. pairs trading and mean reversion;
3. mixed-asset portfolio optimisation;
4. tail-risk measurement and stress testing;
5. machine-learning signal generation.

Each project contains a research notebook and a supporting Python module. Shared data utilities live in `utils/common.py`; repository-level documentation is centralized in this README.

The goal is not to demonstrate that every strategy works. The notebooks are designed to make assumptions, execution rules, validation choices, and failure modes explicit. Where a more complex method does not justify itself, that result is retained rather than tuned away.

### Research principles

- **Chronology matters.** Trading signals are evaluated using information available at the decision time, with delayed execution where appropriate.
- **Benchmarks matter.** Active strategies are compared with simple alternatives such as buy-and-hold or deterministic rules.
- **Risk matters.** Return is considered alongside volatility, drawdown, VaR, Expected Shortfall, and tail attribution.
- **In-sample and out-of-sample results are distinguished.** Descriptive historical optimisation is not presented as predictive performance.
- **Negative results are useful results.** Complexity is not treated as evidence of edge.

---

## Repository Structure

```text
quant-research/
│
├── README.md
│
├── Q1_momentum_backtester/
│   ├── Q1_notebook.ipynb
│   └── Q1_functions.py
│
├── Q2_pairs_trading/
│   ├── Q2_notebook.ipynb
│   └── Q2_functions.py
│
├── Q3_portfolio_optimisation/
│   ├── Q3_notebook.ipynb
│   └── Q3_functions.py
│
├── Q4_tail_risk_stress_testing/
│   ├── Q4_notebook.ipynb
│   └── Q4_functions.py
│
├── Q5_ml_signal_generation/
│   ├── Q5_notebook.ipynb
│   └── Q5_functions.py
│
└── utils/
    └── common.py
```

---

## Project Roadmap

| # | Project | Status | Core question |
|---|---|---|---|
| Q1 | Momentum Backtester | ✅ Complete | Can a simple MA20/MA50 long/cash rule improve the return–drawdown profile of buy-and-hold? |
| Q2 | Pairs Trading | ✅ Complete | Does formation-period cointegration translate into useful out-of-sample mean-reversion trades? |
| Q3 | Mixed-Asset Portfolio Optimisation | ✅ Complete | How do expected returns, covariance, constraints, and priors shape portfolio allocations? |
| Q4 | Tail Risk & Stress Testing | ✅ Complete | How do optimized portfolios behave under simulated and historical tail events, and which assets drive the losses? |
| Q5 | ML Signal Generation | ✅ Complete | Can standard ML classifiers and richer features beat simple directional and trading baselines? |

---

## Projects

### Q1 — Momentum Backtester

[`Q1_notebook.ipynb`](Q1_momentum_backtester/Q1_notebook.ipynb) · [`Q1_functions.py`](Q1_momentum_backtester/Q1_functions.py)

`Python · pandas · yfinance · matplotlib`

A transparent long/cash backtester based on MA20/MA50 Golden/Death Cross signals. The strategy is evaluated on AAPL and SPY from 2016–2025, with pre-sample indicator warm-up and next-session execution.

| Metric | AAPL MA | AAPL Buy & Hold | SPY MA | SPY Buy & Hold |
|---|---:|---:|---:|---:|
| Total Return | 415.8% | 1049.7% | 164.9% | 303.2% |
| CAGR | 17.9% | 27.7% | 10.2% | 15.0% |
| Sharpe | 0.71 | 0.85 | 0.56 | 0.65 |
| Max Drawdown | **−23.9%** | −38.5% | **−27.2%** | −33.7% |

**Finding:** the crossover materially reduces drawdown but does not beat buy-and-hold on return or Sharpe over this sample. Its role is better interpreted as exposure management than return enhancement.

---

### Q2 — Pairs Trading

[`Q2_notebook.ipynb`](Q2_pairs_trading/Q2_notebook.ipynb) · [`Q2_functions.py`](Q2_pairs_trading/Q2_functions.py)

`Python · pandas · statsmodels · scipy · matplotlib`

A cointegration-based KO/PEP pairs strategy with a fixed 2011–2015 formation period and 2016–2025 out-of-sample trading period. The framework uses ADF and Engle-Granger diagnostics, a static OLS hedge ratio, rolling spread z-scores, next-session execution, hedge-ratio-aware sizing, and daily marked-to-market accounting.

Formation-period cointegration is statistically supported at the 5% level (`p ≈ 0.043`), but the trading result is weak.

| Metric | Pairs | KO Buy & Hold | PEP Buy & Hold | SPY Buy & Hold |
|---|---:|---:|---:|---:|
| Total Return | 18.6% | 126.3% | 97.3% | 303.2% |
| CAGR | 1.7% | 8.5% | 7.0% | 15.0% |
| Sharpe | −0.28 | 0.33 | 0.25 | 0.65 |
| Max Drawdown | **−23.8%** | −37.0% | −30.3% | −33.7% |

**Finding:** formation-period cointegration alone is not sufficient to produce attractive out-of-sample returns. The strategy reduces drawdown relative to the directional benchmarks, but the return is too low to compensate for the capital tied up in the trade.

---

### Q3 — Mixed-Asset Portfolio Optimisation

[`Q3_notebook.ipynb`](Q3_portfolio_optimisation/Q3_notebook.ipynb) · [`Q3_functions.py`](Q3_portfolio_optimisation/Q3_functions.py)

`Python · NumPy · pandas · SciPy · matplotlib`

A mixed-asset portfolio study across ten instruments spanning equities, crypto, gold, broad-market and international equity ETFs, and long-duration US Treasuries.

The notebook constructs the efficient frontier and compares equal weight, target-return minimum variance, global minimum variance, maximum Sharpe, and Black-Litterman allocations.

| Portfolio | Expected Return | Volatility | Sharpe |
|---|---:|---:|---:|
| Equal Weight | 29.0% | 23.7% | 1.05 |
| Target-Return Min Variance | 29.0% | **18.4%** | **1.36** |
| Global Min Variance | 8.7% | **10.0%** | 0.47 |
| Maximum Sharpe | 29.2% | 18.6% | **1.36** |

At the equal-weight expected return, minimum-variance optimisation reduces estimated volatility by approximately **22.5%**.

Black-Litterman is implemented separately using an equal-weight reference prior and an explicit relative view, illustrating how changing the expected-return model can materially alter the resulting allocation.

**Finding:** diversification is driven by covariance structure, not merely by the number of assets. The optimizer can materially reshape risk, but the resulting allocations remain conditional on estimated returns, covariance, constraints, and prior assumptions. Historical wealth paths in this notebook are descriptive and in-sample, not predictive backtests.

---

### Q4 — Tail Risk & Stress Testing

[`Q4_notebook.ipynb`](Q4_tail_risk_stress_testing/Q4_notebook.ipynb) · [`Q4_functions.py`](Q4_tail_risk_stress_testing/Q4_functions.py)

`Python · NumPy · pandas · SciPy · matplotlib`

A stress-testing framework applied to the **Global Minimum-Variance** and **Maximum-Sharpe** portfolios inherited from Q3.

The notebook combines:

- correlated Student-t Monte Carlo simulation;
- VaR and Expected Shortfall;
- historical COVID-19 crisis replay;
- and per-asset tail-loss attribution.

| Scenario | Portfolio | VaR 95% | ES 95% | Worst Loss |
|---|---|---:|---:|---:|
| Student-t Monte Carlo | Global Min Variance | 7.6% | 11.4% | 25.9% |
| Student-t Monte Carlo | Max Sharpe | 3.5% | 10.5% | **34.9%** |
| COVID-19 Replay | Global Min Variance | **6.3%** | **8.6%** | **10.8%** |
| COVID-19 Replay | Max Sharpe | 11.5% | 12.0% | 12.7% |

**Finding:** optimisation and stress testing answer different questions. The Maximum-Sharpe portfolio has a much higher simulated return distribution but more severe extreme losses, while the Global Minimum-Variance allocation is more defensive during the historical COVID stress window. Tail attribution also shows that portfolio-level VaR/ES alone cannot explain *why* a portfolio loses money.

---

### Q5 — ML Signal Generation

[`Q5_notebook.ipynb`](Q5_ml_signal_generation/Q5_notebook.ipynb) · [`Q5_functions.py`](Q5_ml_signal_generation/Q5_functions.py)

`Python · pandas · scikit-learn · XGBoost · matplotlib`

A weekly directional-classification experiment on AAPL with SPY as broad-market context.

The notebook uses a two-stage development process:

1. **Model selection:** Logistic Regression vs Random Forest vs XGBoost using the same price-and-volume feature set.
2. **Feature-set selection:** price only vs price + volume vs price + volume + SPY context using the selected model.

Selection is performed on 2016–2022 data. The locked specification — **XGBoost + price and volume features** — is then evaluated on an untouched 2023–2025 period using expanding-window walk-forward prediction.

| Final Evaluation | Result |
|---|---:|
| Always-up baseline accuracy | 54.5% |
| XGBoost accuracy | 49.4% |
| XGBoost precision | 53.0% |
| XGBoost recall | 62.4% |

The final trading comparison uses the same 2023–2025 window and next-session execution for the ML signal and the MA20/MA50 rule.

| Metric | MA20/MA50 | XGBoost ML | Buy & Hold |
|---|---:|---:|---:|
| Total Return | 58.6% | 24.5% | **121.6%** |
| CAGR | 16.7% | 7.6% | **30.5%** |
| Max Drawdown | **−23.5%** | −31.4% | −33.4% |
| Annualized Volatility | **17.6%** | 19.4% | 25.6% |

**Finding:** volume improves XGBoost relative to price-only features during development, but the improvement does not survive as a robust forecasting or trading edge. The locked ML model underperforms both the naïve directional baseline and the simpler MA20/MA50 rule in the final evaluation period.

---

## Research Positioning

The repository progresses from simple signals to portfolio construction and risk analysis:

- **Q1** establishes explicit signal generation, delayed execution, and benchmark-aware backtesting.
- **Q2** moves from directional exposure to relative-value trading and statistical relationship testing.
- **Q3** shifts the problem from security selection to allocation across heterogeneous asset classes.
- **Q4** tests those allocations under simulated and historical tail conditions and decomposes the source of losses.
- **Q5** introduces supervised machine learning while preserving chronological validation and documents where added complexity fails to generate an edge.

Together, the projects emphasize a common principle:

> **A quantitative model is useful only to the extent that its assumptions, validation, risk, and failure modes are understood.**

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| pandas & NumPy | Data manipulation and numerical computing |
| yfinance | Historical market data |
| statsmodels | Unit-root and cointegration testing |
| scikit-learn | Preprocessing, classification, evaluation |
| XGBoost | Nonlinear supervised classification |
| SciPy | Optimisation and statistical methods |
| matplotlib & seaborn | Visualisation |

---

## Related Repositories

| Repository | Focus |
|---|---|
| [Financial Engineering](https://github.com/wveratudela/financial-engineering) | Derivatives pricing, Greeks, yield curves, volatility, and other financial-engineering methods |
| [Mathematical Methods](https://github.com/wveratudela/mathematical-methods) | Control theory, filtering, optimisation, and mathematical methods applied to portfolio problems |

---

## Disclaimer

This repository is for educational and research purposes only. Nothing here constitutes financial advice. All strategies and portfolio analyses are based on historical or simulated data, and past performance does not guarantee future results.

---

*Dr. Walter Vera-Tudela · [github.com/wveratudela](https://github.com/wveratudela)*
