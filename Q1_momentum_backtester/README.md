# Q1 — Momentum Backtester

A backtesting framework built from scratch to evaluate a momentum-based trading strategy using moving average crossovers on US equities.

---

## Objective

Test whether a simple MA20/MA50 crossover strategy can outperform buying and holding the same asset, and benchmark both against the S&P 500 index (SPY). Evaluate performance not just on returns, but on risk-adjusted metrics and drawdown behaviour across different market regimes.

---

## Strategy Logic

**Signal Generation**
- Calculate 20-day and 50-day rolling averages of the closing price
- **Golden Cross** — MA20 crosses above MA50 → enter long position
- **Death Cross** — MA20 crosses below MA50 → exit to cash
- No short selling

**Execution**
- Two modes: all-in on signal, or tranched entry (50% on Golden Cross, remaining 50% if price holds above both MAs after 10 days)
- Whole shares only (no fractional trading)
- No transaction costs (acknowledged limitation)

---

## Files

| File | Description |
|------|-------------|
| `Q1_notebook.ipynb` | Main research notebook with analysis and results |
| `Q1_functions.py` | Modular functions: `fetch_data()`, `add_signals()`, `run_backtest()`, `compute_metrics()`, `plot_performance()` |

---

## Results (AAPL, 10 years)

| Metric | Buy/Sell | Buy/Sell (Tranched) | Buy & Hold AAPL | Buy & Hold SPY |
|--------|----------|---------------------|-----------------|----------------|
| Total Return | 327% | 310% | 951% | 246% |
| Sharpe Ratio | 0.53 | — | 0.68 | 0.47 |
| Max Drawdown | -28.9% | -22.7% | -38.7% | — |
| Win Rate | — | — | — | — |

**The strategy outperforms SPY on raw returns but significantly underperforms buy-and-hold AAPL.** The tranched version sacrifices ~17% total return in exchange for a meaningfully lower max drawdown (-22.7% vs -28.9%).

---

## When Does It Work?

The strategy performs best during **high volatility and crisis periods** (e.g. 2020 COVID crash). Death Cross signals reduce exposure before the worst drawdowns. In sustained bull markets (e.g. 2021–2024), the strategy underperforms because it sits in cash during rallies — missing gains while waiting for a new Golden Cross.

**Character of the strategy:** bear market defence mechanism, not a bull market return maximiser.

---

## Limitations

1. **Survivorship bias** — tested only on companies that survived and thrived. Testing on Kodak produces devastating results ($10k → $94). Failed companies like Enron and Lehman Brothers are no longer listed and therefore invisible to the backtest.
2. **Transaction costs** — all trades assumed frictionless. Brokerage fees, bid-ask spread, and market impact are not modelled.
3. **Lookahead bias** — not present in this implementation, but a critical risk in any backtester worth auditing.

---

## Next Steps

- Add transaction costs to the engine
- Test across a broader universe of tickers including losers
- Explore what asset characteristics (volatility, trend strength) predict strategy success

---

## Dependencies

```
pandas, numpy, yfinance, matplotlib, seaborn
```