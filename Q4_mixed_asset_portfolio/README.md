# Q4 — Mixed Asset Class Portfolio Optimisation

A multi-asset portfolio optimisation framework extending Q3's Markowitz and Black-Litterman approach to a diversified universe spanning stocks, cryptocurrencies, and ETFs/bonds.

---

## Objective

Demonstrate that true diversification requires asset class diversity, not just asset quantity. By combining equities, crypto, gold, bonds, and international ETFs, test whether cross-asset diversification materially improves the efficient frontier compared to the single-sector Mag 7 portfolio in Q3.

---

## Key Difference from Q3

Q3 optimised across 7 large-cap US tech stocks — all positively correlated (0.34–0.64), all driven by the same macro factors. Q4 introduces genuinely uncorrelated asset classes, particularly gold (GLD) and long-term Treasuries (TLT), which have near-zero or negative correlations with equities. This shifts the efficient frontier dramatically leftward — the minimum variance portfolio drops from 0.24 volatility (Q3) to 0.07 (Q4).

---

## Universe

| Asset | Ticker | Class |
|-------|--------|-------|
| Apple | AAPL | Stock |
| Nvidia | NVDA | Stock |
| Chevron | CVX | Stock |
| Taiwan Semiconductor | TSM | Stock |
| Bitcoin | BTC-USD | Crypto |
| Ethereum | ETH-USD | Crypto |
| SPDR Gold ETF | GLD | Commodity ETF |
| S&P 500 ETF | SPY | Equity ETF |
| 20+ Year Treasury ETF | TLT | Bond ETF |
| Developed Markets ETF | VEA | International ETF |

---

## Data Handling

Stocks and ETFs trade Monday–Friday. Crypto trades 24/7. Data is combined using an outer join with forward filling on non-trading days, then aligned to the common start date (2017-11-09, when ETH data begins). This approach preserves crypto weekend data at the cost of slightly understating stock volatility relative to crypto — acknowledged as a limitation.

---

## Strategy Logic

Identical pipeline to Q3:
- Efficient frontier via constrained mean-variance optimisation (scipy SLSQP)
- Monte Carlo simulation: 10,000 Dirichlet random portfolios
- Four key portfolios: equal weight, target-return optimised, minimum variance, maximum Sharpe
- Black-Litterman: market equilibrium returns blended with investor views
- View encoded: AAPL, NVDA, BTC, outperform ETH 5%

---

## Files

| File | Description |
|------|-------------|
| `Q4_notebook.ipynb` | Main research notebook |
| `Q4_functions.py` | Shared functions with Q3, extended for multi-asset class handling |

---

## Results ($10,000 starting capital, ~8 years)

| Portfolio | Final Value | Volatility | Notes |
|-----------|-------------|------------|-------|
| Equal Weight | ~$65,000 | 26.1% | Baseline |
| Target Return Optimised | ~$95,000 | 14.4% | 44.9% vol reduction vs equal weight |
| Minimum Variance | ~$20,000 | 7.0% | TLT 41.5%, GLD 26.5%, SPY 24.8% |
| Maximum Sharpe | ~$130,000 | — | GLD 60%, NVDA 29% |
| Black-Litterman | ~$145,000 | 26.1% | NVDA 35%, AAPL 30%, BTC 9% |
| SPY benchmark | ~$25,000 | — | Reference |

### Optimised Weights by Method

| Asset | Equal | Optimised | Min Var | Max Sharpe | BL |
|-------|-------|-----------|---------|------------|----|
| AAPL | 10% | 8% | 0% | 0% | 30% |
| CVX | 10% | 0% | 3% | 0% | 3% |
| NVDA | 10% | 18% | 0% | 29% | 35% |
| TSM | 10% | 1% | 0% | 0% | 14% |
| BTC-USD | 10% | 5% | 0% | 7% | 9% |
| ETH-USD | 10% | 1% | 0% | 4% | 0% |
| GLD | 10% | 67% | 27% | 60% | 1% |
| SPY | 10% | 0% | 25% | 0% | 6% |
| TLT | 10% | 0% | 42% | 0% | 1% |
| VEA | 10% | 0% | 4% | 0% | 1% |

---

## Key Findings

**Cross-asset diversification is fundamentally more powerful than sector diversification.** MVP volatility dropped from 0.24 (Q3, Mag 7) to 0.07 (Q4, mixed assets) — a 3.4x improvement. This was not achieved by better stock selection but by introducing TLT and GLD, which carry near-zero or negative correlations with equities. The source of diversification matters more than the number of assets.

**Gold is the optimizer's preferred diversifier in this universe.** GLD dominated the target-return optimised (67%) and maximum Sharpe (60%) portfolios due to its near-zero correlation with all equity assets (0.05–0.11). Combined with NVDA's high return, the max Sharpe portfolio essentially holds NVDA for return and GLD as ballast — a combination that raw historical data strongly favours.

**TLT dominates minimum variance but fails during inflation.** At 41.5% weight in the MVP, long-term Treasuries provide the strongest crisis hedge in this universe — but only against recession-driven crashes. The 2022 rate hiking cycle crushed TLT while equities also fell simultaneously, demonstrating that bonds are not unconditional safe havens. They protect against fear, not inflation.

**The optimizer is ruthless about redundancy.** CVX and VEA — both relatively stable assets — were consistently zeroed out because their high correlation with SPY made them redundant. Stability alone does not earn portfolio weight; *uncorrelated* stability does. This is the efficiency principle applied across asset classes.

**Black-Litterman is structurally misaligned with mixed asset class universes.** After correcting a bug in the original implementation, BL achieved only ~0.5% volatility reduction — not the 44.9% previously reported. The root cause is architectural: BL's equilibrium prior derives expected returns from market cap weights, which systematically suppresses GLD and TLT. Gold and Treasury bonds have relatively small market caps compared to NVDA or AAPL, so the equilibrium framework assigns them near-zero or negative expected returns regardless of investor views. The corrected BL weights (NVDA 36%, AAPL 32%, BTC 11%) correctly reflect the investor view but sacrifice the variance reduction that GLD and TLT provide. A proper mixed asset BL implementation requires asset-class-specific priors — a known limitation of standard BL applied outside equity-only universes.

**The portfolio value comparison is illustrative, not predictive.** Weights are derived from and evaluated on the same historical period. A proper evaluation requires rolling window optimisation with out-of-sample testing.

---

## Limitations

1. **Calendar misalignment** — crypto trades 24/7 while stocks trade 5 days per week. Forward filling stock prices on weekends understates their true volatility relative to crypto and artificially reduces stock-crypto covariance. Weekly returns aggregation would produce more honest estimates.

2. **In-sample optimisation** — weights are derived from and evaluated on the same historical period. All final portfolio values reflect historical performance, not future prediction. A proper evaluation requires rolling window optimisation with out-of-sample testing.

3. **Static weights with no rebalancing** — optimal weights are calculated once and held fixed. As prices move, actual weights drift from optimal. A portfolio starting at 36% NVDA becomes significantly overweight NVDA as it rallies, fundamentally changing the risk profile. Rebalancing is required in practice but incurs transaction costs that erode returns.

---

## Next Steps

- Implement weekly returns to resolve calendar misalignment between crypto and equities
- Add tracking error constraint — optimize weights while penalizing large deviations from current holdings, enabling portfolio transition without excessive turnover
- Rolling window optimisation — recalculate weights quarterly using only past data for honest out-of-sample evaluation
- Extend to rebalancing simulation — measure the actual cost of maintaining target weights over time

---

## Dependencies

```
pandas, numpy, yfinance, matplotlib, seaborn, scipy
```
