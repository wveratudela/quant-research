# Q4t — Tail Risk & Stress Testing Framework

A portfolio stress testing framework applying Monte Carlo simulation, historical crisis replay, and tail risk decomposition to a mixed-asset universe.

---

## Core Finding

**The min-variance optimizer correctly identified crypto as the dominant tail risk during COVID (ETH + BTC = 46% of tail losses), but catastrophically mispriced TLT — the asset it most heavily allocated to (42%) — which became the single largest loss driver during the 2022 rate shock, contributing 60% of tail losses.**

Diversification is regime-dependent. The same asset that protects you in one crisis can destroy you in another.

---

## Objective

Build a stress testing engine to answer three questions:

1. **Do Monte Carlo simulations (normal vs fat-tailed) accurately predict crisis losses?**
2. **How does the same portfolio behave across different crisis types?**
3. **Which assets actually cause losses during tail events?**

Standard VaR models use historical covariance matrices and assume normal distributions. This framework tests whether those assumptions hold during real market shocks.

---

## Universe

| Asset | Ticker | Class | Notes |
|-------|--------|-------|-------|
| Apple | AAPL | Stock | Tech equity |
| Nvidia | NVDA | Stock | High-vol growth |
| Chevron | CVX | Stock | Energy / commodity proxy |
| Taiwan Semiconductor | TSM | Stock | International tech |
| Bitcoin | BTC-USD | Crypto | 24/7 trading, high vol |
| Ethereum | ETH-USD | Crypto | Correlated with BTC (ρ = 0.78) |
| SPDR Gold ETF | GLD | Commodity ETF | Low correlation hedge |
| S&P 500 ETF | SPY | Equity ETF | Market proxy |
| 20+ Year Treasury ETF | TLT | Bond ETF | Rate-sensitive safe haven |
| Developed Markets ETF | VEA | International ETF | Non-US equities |

---

## Methodology

### 1. Monte Carlo Simulation

Generate 10,000 synthetic one-year forward paths using Cholesky-decomposed correlated returns:

$$R = \mu \cdot \Delta t + L \cdot Z \cdot \sqrt{\Delta t}$$

where $\Sigma = LL^\top$ (Cholesky factorization) and $Z \sim \mathcal{N}(0,I)$ or $Z \sim t_\nu$ for fat tails.

Each simulation compounds 252 daily returns into a terminal portfolio value. VaR and CVaR are computed from the distribution of terminal outcomes.

### 2. Historical Crisis Replay

Extract actual daily returns during defined crisis windows (COVID-19, 2022 Rate Shock) and compute cumulative portfolio losses day by day. Each trading day represents a scenario: "what if I had to liquidate today?"

Key fix: crypto trades 7 days/week, equities trade 5. Forward-fill equity prices through weekends to avoid calendar misalignment artifacts.

### 3. Tail Risk Contribution

For the worst X% of days (by portfolio cumulative return), decompose losses into per-asset contributions:

$$C_i = w_i \times (r_i - 1)$$

Positive = asset hurt the portfolio. Negative = asset helped. This answers: "what killed me?"

---

## Crisis Windows Tested

| Event | Window | Character |
|-------|--------|-----------|
| COVID-19 Recession | 2020-02-01 → 2020-04-30 | Sharp crash, fast V-shaped recovery |
| 2022 Rate Shock | 2022-01-03 → 2022-12-31 | Sustained grind, no recovery within window |

Earlier crises (2008, dot-com, 1987) excluded due to missing crypto data — BTC begins 2017, ETH even later.

---

## Results

### VaR Comparison Across Scenarios ($120,500 starting capital)

| Scenario | VaR 95% | ES 95% | Character |
|---|---|---|---|
| **MC Normal** | 7.8% ($9,447) | 11.0% ($13,230) | Baseline forward-looking |
| **MC Student-t (ν=4)** | 10.7% ($12,865) | 17.7% ($21,331) | Fatter tails, marginal improvement |
| **COVID-19 (equal weights)** | 25.5% ($30,729) | 26.5% ($31,935) | Sharp crash, thin tail beyond VaR |
| **2022 Rate Shock (equal weights)** | 37.4% ($45,096) | 38.4% ($46,227) | Sustained grind, 47% worse than COVID |

**Key finding:** Monte Carlo VaR massively underestimates real crisis severity. The normal-distribution MC predicted 7.8% VaR; COVID delivered 25.5%, 2022 delivered 37.4%. Even fat-tailed Student-t barely moved the needle (10.7%). Historical simulation is the only honest tail risk measure.

---

### Portfolio Weight Comparison: Equal vs Min-Variance

Tested two weight sets across both crises:

**Equal weights:** 10% per asset  
**Min-variance weights (from Q4 optimizer):** TLT 42%, SPY 25%, GLD 27%, VEA 4%, CVX 3%, all others 0%

#### COVID-19 Results

| Weights | VaR 95% | ES 95% | Top loss driver |
|---|---|---|---|
| Equal | 25.5% | 26.5% | ETH (26%) + BTC (20%) + CVX (15%) |
| Min-Var | **8.9%** | 10.6% | SPY (63%) only |

Min-variance cut COVID losses by **65%**. Heavy TLT/GLD allocation worked exactly as designed — these assets contributed 0.06% and 0.24% respectively to tail losses.

#### 2022 Rate Shock Results

| Weights | VaR 95% | ES 95% | Top loss driver |
|---|---|---|---|
| Equal | 37.4% | 38.4% | ETH (21%) + BTC (17%) + NVDA (17%) |
| Min-Var | **22.2%** | 22.8% | **TLT (60%)** |

Min-variance only cut rate shock losses by **41%** (vs 65% in COVID). The asset that saved the portfolio in 2020 became the killer in 2022. TLT went from 0.06% contribution in COVID to 60% in the rate shock — a 1000x flip in risk profile.

---

### Tail Risk Contributions: The "What Kills You" View

**COVID-19 (min-var weights, worst 75% of days):**

| Asset | Contribution |
|-------|--------------|
| SPY | 63% |
| CVX | 17% |
| VEA | 11% |
| **TLT** | **0.06%** ← safe haven |
| **GLD** | **0.24%** ← safe haven |

**2022 Rate Shock (min-var weights, worst 75% of days):**

| Asset | Contribution |
|-------|--------------|
| **TLT** | **60%** ← single largest killer |
| SPY | 23% |
| GLD | 12% |
| VEA | 6% |
| CVX | -1.4% ← actually helped (energy rally) |

The same optimizer, the same "safe" allocation, opposite outcomes. **There is no unconditional safe haven.**

---

## Key Findings

**1. Monte Carlo simulation systematically underprices tail risk**

Normal-distribution MC predicted 7.8% VaR. Reality delivered 25–37%. Even Student-t fat tails barely helped. The problem is structural: MC uses the normal-regime covariance matrix, which doesn't know that correlations converge in crises. Diversification disappears exactly when you need it.

**2. Crisis character matters more than crisis severity**

COVID was a sharp, fear-driven crash where TLT and GLD worked as designed. The 2022 rate shock was an inflation-driven grind where bonds and equities fell simultaneously. VaR numbers alone miss this — you need per-asset loss decomposition to understand *why* the portfolio failed.

**3. Min-variance optimization is regime-specific**

The optimizer correctly allocated to TLT and GLD because they had negative correlations with equities *in the historical data*. It could not foresee a regime shift where those correlations break down. The lesson: backward-looking covariance is a poor guide to forward tail risk.

**4. Crypto was the predicted tail risk, TLT was the surprise**

The optimizer zeroed out crypto in the min-variance portfolio — recognizing it as high-vol and correlated with equities. That call was vindicated in both crises. The catastrophic miss was TLT, which looked safe historically but collapsed when the Fed hiked rates aggressively.

**5. CVX provided genuine diversification in 2022**

Energy stocks rallied post-Ukraine invasion while everything else fell. CVX had a *negative* contribution to tail losses in 2022 (−1.4%), meaning it offset losses rather than causing them. This is what real diversification looks like — an asset that zigs when others zag.

---

## Files

| File | Description |
|------|-------------|
| `Q4t_notebook.ipynb` | Full stress testing analysis with visualizations |
| `Q4t_functions.py` | Core simulation and risk calculation functions |
| `README.md` | This document |

---

## Limitations

### 1. Calendar Misalignment (crypto 24/7 vs equities 5-day)

Bitcoin and Ethereum trade weekends and holidays; stocks don't. Forward-filling equity prices through non-trading days preserves all data but slightly understates equity volatility relative to crypto. A more rigorous approach would align all assets to weekly returns before computing covariance, at the cost of reduced sample size.

### 2. Missing Asset Problem for Pre-2017 Crises

BTC begins 2017, ETH begins late 2017. The Great Recession (2007–2009), dot-com crash (2000–2002), and earlier crises cannot be replayed with the full universe. Proxying crypto with high-vol tech stocks introduces modelling assumptions that belong in a caveat, not silently assumed.

### 3. Static Weights with No Rebalancing

All scenarios assume buy-and-hold with fixed percentage weights. In reality, as NVDA rallies 400%, it becomes a much larger fraction of the portfolio, concentrating risk. Rebalancing back to target weights incurs transaction costs and tax drag — both ignored here.

### 4. Single Historical Path per Crisis

Each crisis happened once. COVID VaR is computed from ~60 trading days in Feb–Apr 2020. That's not a large sample for a 95th percentile estimate. Monte Carlo simulations have 10,000 paths; historical replay has one. The statistical uncertainty around historical VaR is larger than the numbers suggest.

### 5. Normal-Regime Covariance for MC

The Monte Carlo engine uses the full 8-year historical covariance matrix, which blends calm and crisis periods. A more sophisticated approach would estimate separate calm-regime and stressed-regime covariance matrices, then test how much correlation converges during actual tail events.

---

## Next Steps

- **Stress-adjusted covariance matrix:** Estimate crisis-regime correlations separately (e.g., rolling correlation during VIX > 30 periods) and compare MC VaR using stressed vs normal covariance
- **Expand crisis library:** Add 2008 GFC using equity-only subset to test bond behavior during the last major recession
- **Dynamic rebalancing simulation:** Test how frequently rebalancing back to min-var weights affects realized tail losses (transaction costs vs risk reduction tradeoff)
- **Regime detection framework:** Build a simple regime classifier (VIX thresholds + MA crossovers) to test whether switching between aggressive/defensive allocations improves risk-adjusted returns
- **Copula-based tail dependence:** Model joint extremes using Student-t copulas rather than multivariate normal to capture the empirical fact that assets crash together more often than normal distributions predict

---

## Dependencies

```
pandas, numpy, yfinance, matplotlib, seaborn, scipy
```

---

## Disclaimer

This framework is for educational and research purposes only. Past crisis performance does not predict future outcomes. No asset is an unconditional safe haven. All risk metrics (VaR, CVaR, tail contributions) are backward-looking estimates with substantial statistical uncertainty. Do not use this for live trading without independent validation, out-of-sample testing, and professional risk oversight.