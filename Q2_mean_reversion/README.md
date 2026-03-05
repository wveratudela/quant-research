# Q2 — Pairs Trading

A market-neutral pairs trading framework built from scratch to exploit the cointegration relationship between two financially linked assets.

---

## Objective

Identify a statistically cointegrated pair of stocks, model the spread between them, and trade mean reversion of that spread. Evaluate performance not on absolute returns, but on risk-adjusted metrics and capital preservation — the true value proposition of market-neutral strategies.

---

## Strategy Logic

**Pair Selection**
- Test each asset individually for non-stationarity (ADF test, p > 0.05 required)
- Test the pair for cointegration (Engle-Granger test, p < 0.05 required)
- Calculate hedge ratio β via OLS regression: `Spread = Asset_1 - β × Asset_2`
- Validate with R² and out-of-sample stability check

**Signal Generation**
- Calculate rolling z-score of the spread: `z = (Spread - μ_252d) / σ_60d`
- Mixed windows: 252-day mean (stable equilibrium anchor), 60-day std (responsive to current volatility)
- State machine signal: enter on |z| > 2, exit when z reverts to 0

**Execution**
- z > +2 → short Asset 1, long Asset 2 (spread will narrow)
- z < -2 → long Asset 1, short Asset 2 (spread will widen back)
- Dollar-neutral on entry: equal capital allocated to each leg
- Short proceeds fund the long leg — net cash outflow approximately zero

**P&L Tracking**
- Realized P&L (cash) — locked in on trade close, used for metrics
- Mark-to-market Total (cash + open position value) — used for equity curve
- The distinction between realized and unrealized P&L is explicitly acknowledged

---

## Pair Tested

**Visa (V) and Mastercard (MA)** — selected after systematic testing of multiple candidate pairs. KO/PEP, XOM/CVX, and MSFT/AAPL all failed the cointegration test over the 10-year window. V/MA passed with p = 0.0006, reflecting their near-identical business models and shared exposure to consumer spending and payment network economics.

---

## Files

| File | Description |
|------|-------------|
| `Q2_notebook.ipynb` | Main research notebook with analysis and results |
| `Q2_functions.py` | Modular functions: `fetch_data()`, `check_data()`, `check_signals()`, `run_pairs_trading()`, `compute_metrics()`, `plot_performance()` |

---

## Results (V/MA, 10 years)

| Metric | Pairs V/MA | Buy & Hold V | Buy & Hold MA | Buy & Hold SPY |
|--------|------------|--------------|---------------|----------------|
| Total Return | 58% | 345% | 492% | 242% |
| Sharpe Ratio | 0.13 | 0.45 | 0.52 | 0.46 |
| Max Drawdown | -2.1% | -36.4% | -41.0% | -34.1% |
| CAGR | 4.7% | 16.1% | 19.5% | 13.1% |

---

## When Does It Work?

The strategy is explicitly **market-neutral** — it does not capture bull market returns. Its value is most visible during crisis periods. The 2020 COVID crash that reduced buy-and-hold portfolios by 30-40% left the pairs strategy virtually untouched, as losses on one leg were offset by gains on the other.

The strategy performs best in **volatile, mean-reverting markets** where the spread oscillates predictably. It underperforms when the cointegration relationship drifts or breaks, generating false signals on a widening spread that never reverts.

**Character of the strategy:** capital preservation tool, not a return maximiser. In an institutional context, the near-zero drawdown and market-neutral returns become commercially significant when leveraged.

---

## Limitations

1. **Zero borrow rate** — shorting shares incurs an annualized borrow fee (typically 0.5–3% for liquid large-caps) and margin requirements. Neither is modelled here, meaningfully overstating real returns.
2. **Cointegration instability** — the cointegration relationship was validated on historical data but can break permanently due to regulatory changes, competitive shifts, or macroeconomic regime changes. The strategy has no mechanism to detect structural breaks and will continue generating signals on a broken relationship indefinitely.
3. **Realized vs unrealized P&L** — metrics are calculated on realized cash (closed trades) rather than mark-to-market total value. This is a deliberate methodological choice that more honestly reflects delivered performance, but understates intraday volatility. A full implementation would report both.

---

## Next Steps

- Add borrow rate and transaction costs
- Implement rolling cointegration testing to detect relationship breakdown
- Test across a broader universe of candidate pairs
- Explore dynamic hedge ratio (Kalman filter) instead of static OLS β

---

## Dependencies

```
pandas, numpy, yfinance, matplotlib, seaborn, statsmodels, scikit-learn
```
