# Q5 — ML Signal Generation with Volume-Confirmed Features

A machine learning framework for generating trading signals from price, volume, and cross-asset features, validated using walk-forward methodology and benchmarked against the MA crossover strategy from Q1.

---

## Objective

Test a falsifiable hypothesis: **volume-confirmed signals outperform price-only signals** in predicting 5-day forward returns. Rather than predicting stock prices — a well-known failure mode of naive ML applications in finance — this framework identifies recurring patterns in price and volume data that have historically preceded positive or negative returns, using those patterns to generate better-timed entry and exit signals than simple moving average crossovers.

---

## Hypothesis

*"Volume carries predictive information beyond price alone. A model trained on volume-confirmed features will generate more accurate trading signals than an identical model trained on price features only."*

This is testable, falsifiable, and directly comparable — the same model architecture, the same walk-forward validation, the same backtesting engine. The only variable is the feature set.

---

## Feature Set (15 total)

### Price-Based (8)
| Feature | Description |
|---------|-------------|
| mom_5 | 5-day log momentum |
| mom_20 | 20-day log momentum |
| mom_60 | 60-day log momentum |
| MA_ratio | MA20/MA50 ratio — golden cross regime indicator |
| Stoch_14 | Stochastic Oscillator — price position within 14-day high/low range |
| BBP | Bollinger Band Position — price within 20-day volatility band |
| real_vola_20 | 20-day realized volatility (rolling std of daily returns) |
| positive_days | Fraction of up days in last 20 trading days |

### Volume-Based (5)
| Feature | Description |
|---------|-------------|
| VR | Volume ratio — today / 20-day average |
| OBV_normalized | On-Balance Volume, z-score normalized over 20-day window |
| pv_divergence | Log price change / log volume change over 5 days (clipped ±10) |
| volu_trend | 5-day MA of volume / 20-day MA of volume |
| high_volu_flag | Binary — 1 if volume > 2x 20-day average |

### Cross-Asset (4)
| Feature | Description |
|---------|-------------|
| spy_ret_5 | SPY 5-day log return — broad market context |
| btc_ret_5 | BTC 5-day log return — risk appetite indicator |
| gld_ret_5 | GLD 5-day log return — risk-off signal |
| rolling_corr | 20-day rolling correlation between AAPL and SPY |

---

## Methodology

**Target variable:** Binary classification — will the 5-day forward return be positive (1) or negative (0)?

**Sampling:** Non-overlapping 5-day windows to prevent label autocorrelation. 10 years of daily data → ~719 independent samples.

**Validation:** Walk-forward expanding window — train on all data up to point T, predict T+1. Never uses future data in training. Minimum 50 samples before first prediction.

**Class imbalance:** `class_weight='balanced'` applied to all models. Naive baseline (always predict "up") = 57.7%.

**Models compared:**
- Logistic Regression — linear baseline
- Random Forest Classifier — non-linear ensemble
- XGBoost Classifier — gradient boosting, strongest performer

**Feature sets tested:**
1. Price only (8 features)
2. Price + Volume (13 features)
3. Price + Volume + Cross-asset (17 features)

---

## Files

| File | Description |
|------|-------------|
| `Q5_notebook.ipynb` | Main research notebook |
| `Q5_functions.py` | `engineer_features()`, `model_training()`, `compute_metrics()`, `signal_backtest()` |

---

## Results

### Model Performance (XGBoost, walk-forward)

| Feature Set | Accuracy | Precision | Recall | Final Value ($10k) |
|-------------|----------|-----------|--------|-------------------|
| Naive baseline (always up) | 57.7% | — | — | — |
| Price only | 51.7% | 57.5% | 62.7% | ~$22,000 |
| Price + Volume | 54.1% | 59.0% | 67.4% | ~$30,000 |
| Price + Volume + Cross-asset | ~53% | — | — | ~$25,000 |
| MA Crossover (Q1 baseline) | — | — | — | ~$41,000 |
| Buy & Hold | — | — | — | ~$110,000 |

### Model Comparison (Price + Volume features)

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression | 47.8% | 55.2% | 50.5% |
| Random Forest | 50.7% | 55.6% | 71.8% |
| XGBoost | 54.1% | 59.0% | 67.4% |

---

## Key Findings

**The volume hypothesis is confirmed.** Volume-confirmed features improved XGBoost accuracy by 2.4 percentage points and backtested returns by 36% ($30k vs $22k) compared to price-only features. The improvement is consistent across all three metrics — volume adds genuine signal, not noise.

**Volume Ratio (VR) was the single most important feature** in the price+volume model, ranking above all price-based features including momentum. When today's volume exceeds its 20-day average, the price move carries more conviction. The high-volume binary flag (high_volu_flag) added no value beyond the continuous ratio — magnitude matters more than threshold crossing.

**Cross-asset features hurt rather than helped.** Adding BTC, GLD, and SPY returns reduced backtested performance to $25k — below the price+volume result. The local information (how much conviction is behind this price move) is more predictive than global macro context for a single-asset signal. Adding 4 mediocre features diluted the importance of the 3 strongest ones (VR, Stoch_14, vola_20).

**No ML model beat the naive baseline on accuracy.** All three models failed to exceed 57.7% accuracy — the base rate of positive 5-day returns in a bull market. This is not a failure of the methodology — it is an honest finding. For a trending asset (AAPL) over a sustained bull market period, momentum-following rules (MA crossover) remain more effective than supervised classification. The ML approach may show greater relative advantage in mean-reverting or range-bound conditions where momentum signals fail.

**Simple beats complex for trending assets.** The MA crossover from Q1 — two lines of logic — outperformed a carefully engineered ML pipeline on both returns ($41k vs $30k) and interpretability. Complexity is only justified when it demonstrably improves performance. Here, it does not.

---

## Limitations

1. **Survivorship bias** — trained and tested on AAPL, a stock that not only survived but became the most valuable company in history over the test period. Results on a broader universe including losers would almost certainly be weaker.

2. **Single feature set per model** — the same features were used for all three models despite different architectures. Logistic Regression may benefit from different feature engineering (linear transformations, interactions) while tree-based models handle raw features naturally. Model-specific feature selection was not explored.

3. **Fixed prediction window** — 5-day forward return was used throughout. The optimal prediction horizon may differ by model, market regime, or asset. Shorter windows (1-2 days) may suit high-frequency signals; longer windows (20 days) may better match MA crossover timing.

---

## Next Steps

- Remove features rated below the equal importance baseline (high_volu_flag, rolling_corr, volu_trend, gld_ret_5, mom_5) and retrain — fewer high-quality features typically improve generalization
- Test on a broader universe including non-tech assets and historical losers
- Test in mean-reverting or range-bound market conditions where momentum signals fail and ML may have an advantage
- Explore reinforcement learning as an extension — reward profitable trades, penalize losses, allow the model to learn a dynamic trading policy

---

## Dependencies

```
pandas, numpy, yfinance, matplotlib, seaborn, sklearn, xgboost
```