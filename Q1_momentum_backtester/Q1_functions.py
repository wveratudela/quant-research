import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

import sys
sys.path.append('../utils')   # path relative to the notebook
from common import get_ma_windows, compute_vol_thresholds

def add_signals(df, fast=20, slow=50, MA_windows=False, MA_assets=False):
    
    df = df.copy()
    
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Daily_Return'] = df['Close'].pct_change()
    df = df.dropna()

    if MA_windows:
        current_sigma = df['Daily_Return'].std()
        fast, slow = get_ma_windows(current_sigma, {"low":0.010, "medium":0.015, "high":0.025})

    if MA_assets:
        thr = compute_vol_thresholds(df['Log_Returns'])
        fast, slow = get_ma_windows(thr["sigma"].iloc[-1], thr)

    # Calculate 20-day and 50-day rolling averages
    df['MA20'] = df['Close'].rolling(window=fast).mean()
    df['MA50'] = df['Close'].rolling(window=slow).mean()
    
    # Generate signals based on crossovers
    # Signal = 1 (long) when 20-day crosses above 50-day - Crossover UP
    # Signal = 0 (flat/cash) when 20-day crosses below 50-day - Crossover DOWN
    df['Signal'] = (df['MA20'] > df['MA50']).astype(int)
    
    # Identify crossover points
    df['Golden_Cross'] = (df['Signal'] == 1) & (df['Signal'].shift(1) == 0)  # MA20 crosses above MA50
    df['Death_Cross'] = (df['Signal'] == 0) & (df['Signal'].shift(1) == 1)   # MA20 crosses below MA50
    
    return df


def run_backtest(df, starting_capital, initial_long=False):

    df = df.copy()

    df["Cash"] = 0.0
    df["Portfolio_Value"] = 0.0
    df["Shares"] = 0.0

    cash = float(starting_capital)
    shares = 0.0

    pending_entry = initial_long
    pending_exit = False

    for idx, row in df.iterrows():

        price = row["Close"]

        # Execute orders generated from previously available information
        if pending_exit and shares > 0:
            cash += shares * price
            shares = 0.0

        if pending_entry and shares == 0:
            shares = cash / price
            cash = 0.0

        pending_entry = False
        pending_exit = False

        # Generate today's orders
        if row["Golden_Cross"] and shares == 0:
            pending_entry = True

        elif row["Death_Cross"] and shares > 0:
            pending_exit = True

        portfolio_value = shares * price

        df.loc[idx, "Cash"] = cash
        df.loc[idx, "Portfolio_Value"] = portfolio_value
        df.loc[idx, "Shares"] = shares

    df["Total"] = df["Cash"] + df["Portfolio_Value"]

    bh_shares = starting_capital / df["Close"].iloc[0]
    df["Buy_Hold"] = bh_shares * df["Close"]

    return df


def compute_metrics(
    df,
    starting_capital,
    risk_free_rate=0.04,
    trading_days=252,
):

    df = df.copy()

    years = (df.index[-1] - df.index[0]).days / 365.25
    rf_daily = (1 + risk_free_rate) ** (1 / trading_days) - 1

    # ---------------------------------------------------------
    # Moving-average strategy
    # ---------------------------------------------------------

    strategy_returns = df["Total"].pct_change()

    total_return_bs = (
        df["Total"].iloc[-1] / starting_capital - 1
    )

    cagr_bs = (
        df["Total"].iloc[-1] / starting_capital
    ) ** (1 / years) - 1

    excess_returns_bs = strategy_returns - rf_daily

    sharpe_bs = (
        excess_returns_bs.mean()
        / excess_returns_bs.std()
        * np.sqrt(trading_days)
    )

    running_max_bs = df["Total"].cummax()

    df["Drawdown_BS"] = (
        df["Total"] / running_max_bs - 1
    )

    max_drawdown_bs = df["Drawdown_BS"].min()

    calmar_bs = (
        cagr_bs / abs(max_drawdown_bs)
        if max_drawdown_bs != 0
        else np.nan
    )

    # ---------------------------------------------------------
    # Trade-level win rate
    # ---------------------------------------------------------

    previous_shares = df["Shares"].shift(1).fillna(0)

    entry_mask = (
        (previous_shares == 0)
        & (df["Shares"] > 0)
    )

    exit_mask = (
        (previous_shares > 0)
        & (df["Shares"] == 0)
    )

    entry_prices = df.loc[entry_mask, "Close"].to_numpy()
    exit_prices = df.loc[exit_mask, "Close"].to_numpy()

    n_closed_trades = min(
        len(entry_prices),
        len(exit_prices),
    )

    if n_closed_trades > 0:
        trade_returns = (
            exit_prices[:n_closed_trades]
            / entry_prices[:n_closed_trades]
            - 1
        )

        win_rate_bs = np.mean(trade_returns > 0)

    else:
        win_rate_bs = np.nan

    # ---------------------------------------------------------
    # Buy-and-hold
    # ---------------------------------------------------------

    buy_hold_returns = df["Buy_Hold"].pct_change()

    total_return_bh = (
        df["Buy_Hold"].iloc[-1]
        / starting_capital
        - 1
    )

    cagr_bh = (
        df["Buy_Hold"].iloc[-1]
        / starting_capital
    ) ** (1 / years) - 1

    excess_returns_bh = buy_hold_returns - rf_daily

    sharpe_bh = (
        excess_returns_bh.mean()
        / excess_returns_bh.std()
        * np.sqrt(trading_days)
    )

    running_max_bh = df["Buy_Hold"].cummax()

    df["Drawdown_BH"] = (
        df["Buy_Hold"] / running_max_bh - 1
    )

    max_drawdown_bh = df["Drawdown_BH"].min()

    calmar_bh = (
        cagr_bh / abs(max_drawdown_bh)
        if max_drawdown_bh != 0
        else np.nan
    )

    # ---------------------------------------------------------
    # Calendar-year returns
    # ---------------------------------------------------------

    yearly_bs = (
        (1 + strategy_returns)
        .groupby(df.index.year)
        .prod()
        - 1
    )

    yearly_bh = (
        (1 + buy_hold_returns)
        .groupby(df.index.year)
        .prod()
        - 1
    )

    yearly_df = pd.DataFrame({
        "MA Strategy": yearly_bs,
        "Buy & Hold": yearly_bh,
    })

    yearly_df.index.name = "Year"

    # ---------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------

    comparison_table = pd.DataFrame(
        {
            "MA Strategy": [
                total_return_bs,
                cagr_bs,
                sharpe_bs,
                max_drawdown_bs,
                calmar_bs,
                win_rate_bs,
            ],
            "Buy & Hold": [
                total_return_bh,
                cagr_bh,
                sharpe_bh,
                max_drawdown_bh,
                calmar_bh,
                np.nan,
            ],
        },
        index=[
            "Total Return",
            "CAGR",
            "Sharpe",
            "Max Drawdown",
            "Calmar",
            "Win Rate",
        ],
    )

    return df, comparison_table, yearly_df

    
def plot_performance(dA, dS, yA, yS, monthly=False):

    plt.figure(figsize=(18, 6))
    colors = sns.color_palette("colorblind")

    if monthly:
        dA = dA.resample("ME").last()
        dS = dS.resample("ME").last()

    # ---------------------------------------------------------
    # Equity curve
    # ---------------------------------------------------------

    plt.subplot(1, 3, 1)

    plt.plot(
        dA.index,
        dA["Total"] / dA["Total"].iloc[0],
        label="MA Strategy",
        linewidth=1.5,
        color=colors[0],
    )

    plt.plot(
        dA.index,
        dA["Buy_Hold"] / dA["Buy_Hold"].iloc[0],
        label="Buy & Hold AAPL",
        linewidth=1.0,
        color=colors[1],
    )

    plt.plot(
        dS.index,
        dS["Buy_Hold"] / dS["Buy_Hold"].iloc[0],
        label="Buy & Hold SPY",
        linewidth=1.0,
        color=colors[2],
    )

    plt.title("Growth of Initial Capital")
    plt.xlabel("Date")
    plt.ylabel("Wealth Multiple")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # ---------------------------------------------------------
    # Drawdown
    # ---------------------------------------------------------

    plt.subplot(1, 3, 2)

    plt.plot(
        dA.index,
        dA["Drawdown_BS"],
        label="MA Strategy",
        linewidth=1.5,
        color=colors[0],
    )

    plt.plot(
        dA.index,
        dA["Drawdown_BH"],
        label="Buy & Hold AAPL",
        linewidth=1.0,
        color=colors[1],
    )

    plt.plot(
        dS.index,
        dS["Drawdown_BH"],
        label="Buy & Hold SPY",
        linewidth=1.0,
        color=colors[2],
    )

    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # ---------------------------------------------------------
    # Calendar-year returns
    # ---------------------------------------------------------

    plt.subplot(1, 3, 3)

    years = yA.index
    x = np.arange(len(years))
    width = 0.25

    plt.bar(
        x - width,
        yA["MA Strategy"],
        width,
        label="MA Strategy",
        color=colors[0],
    )

    plt.bar(
        x,
        yA["Buy & Hold"],
        width,
        label="Buy & Hold AAPL",
        color=colors[1],
    )

    plt.bar(
        x + width,
        yS["Buy & Hold"],
        width,
        label="Buy & Hold SPY",
        color=colors[2],
    )

    plt.axhline(
        y=0,
        color="black",
        linewidth=0.8,
        linestyle="--",
    )

    plt.xticks(x, years)
    plt.title("Calendar-Year Returns")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()