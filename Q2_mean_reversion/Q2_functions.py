import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.linear_model import LinearRegression


def check_data(df1, df2):

    result = adfuller(df1['Close'])
    print(f"KO ADF: {result[0]:.4f}, p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print(f"- - - - - WARNING: p-value: {result[1]:.4f} < 5% - - - - -")
    
    result = adfuller(df2['Close'])
    print(f"PEP ADF: {result[0]:.4f}, p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print(f"- - - - - WARNING: p-value: {result[1]:.4f} < 5% - - - - -")
    
    _, p_value, critical_values = coint(df1['Close'], df2['Close'])
    print(f"Cointegration p-value: {p_value:.4f}")
    if p_value > 0.05:
        print(f"- - - - - WARNING: p-value: {p_value:.4f} > 5% - - - - -")
    print(f"Critical values: {critical_values}")
    

def estimate_hedge_ratio(df1, df2):
    """
    Estimate a static OLS hedge ratio using formation-period data.

    Models:
        df1['Close'] = alpha + beta * df2['Close'] + error

    Returns
    -------
    beta : float
        Estimated hedge ratio.
    """

    X = df2["Close"].to_numpy().reshape(-1, 1)
    y = df1["Close"].to_numpy()

    model = LinearRegression().fit(X, y)

    return model.coef_[0]


def build_pairs_signal(
    df1,
    df2,
    beta,
    evaluation_start=None,
    mean_window=252,
    std_window=60,
    entry_z=2.0,
):

    df1 = df1.copy()
    df2 = df2.copy()

    # Construct spread using the hedge ratio
    # estimated from the formation period
    spread = (
        df1["Close"]
        - beta * df2["Close"]
    )

    # Rolling z-score
    rolling_mean = (
        spread
        .rolling(mean_window)
        .mean()
    )

    rolling_std = (
        spread
        .rolling(std_window)
        .std()
    )

    z_score = (
        (spread - rolling_mean)
        / rolling_std
    )

    # Restrict signal generation to the evaluation period,
    # while preserving earlier observations for rolling warm-up
    if evaluation_start is not None:
        z_trade = z_score.loc[
            z_score.index >= evaluation_start
        ].copy()
    else:
        z_trade = z_score.copy()

    # State:
    #  0 = no position
    # +1 = short spread:
    #      short df1, long beta * df2
    # -1 = long spread:
    #      long df1, short beta * df2
    signal = pd.Series(
        0,
        index=z_trade.index,
        dtype=int,
    )

    state = 0

    for i in range(len(z_trade)):

        z = z_trade.iloc[i]

        if pd.isna(z):
            signal.iloc[i] = state
            continue

        if state == 0:

            if z > entry_z:
                state = 1

            elif z < -entry_z:
                state = -1

        elif state == 1 and z < 0:
            state = 0

        elif state == -1 and z > 0:
            state = 0

        signal.iloc[i] = state

    return signal


def run_pairs_trading(
    df1,
    df2,
    signal,
    beta,
    starting_capital,
    gross_leverage=1.0,
):
    """
    Backtest a static-beta pairs strategy.

    Signal convention
    -----------------
     0 : flat
    +1 : short spread  -> short df1, long beta * df2
    -1 : long spread   -> long df1, short beta * df2

    Signals formed at close t are executed at close t+1.

    gross_leverage = 1.0 means gross exposure equals current equity
    whenever a new position is opened.
    """

    df1 = df1.copy()
    df2 = df2.copy()

    df = pd.DataFrame(index=df1.index)
    df["Signal"] = signal

    cash = float(starting_capital)

    q1 = 0.0
    q2 = 0.0
    current_state = 0

    cash_list = []
    q1_list = []
    q2_list = []
    value1_list = []
    value2_list = []
    gross_list = []
    total_list = []

    for i in range(len(df)):

        price1 = df1["Close"].iloc[i]
        price2 = df2["Close"].iloc[i]

        # Signal observed yesterday is executed today
        target_state = (
            int(signal.iloc[i - 1])
            if i > 0
            else 0
        )

        # Rebalance only when the trading state changes
        if target_state != current_state:

            # Close existing positions
            if current_state != 0:
                cash += q1 * price1
                cash += q2 * price2

                q1 = 0.0
                q2 = 0.0

            # Open new spread position
            if target_state != 0:

                current_equity = cash

                gross_budget = (
                    current_equity * gross_leverage
                )

                k = gross_budget / (
                    price1 + abs(beta) * price2
                )

                if target_state == -1:
                    # Long spread:
                    # +V - beta*MA
                    q1 = k
                    q2 = -beta * k

                elif target_state == 1:
                    # Short spread:
                    # -V + beta*MA
                    q1 = -k
                    q2 = beta * k

                # Trading cash flow:
                # buying reduces cash,
                # short selling increases cash
                cash -= q1 * price1
                cash -= q2 * price2

            current_state = target_state

        # Mark positions to today's close
        value1 = q1 * price1
        value2 = q2 * price2

        total = cash + value1 + value2

        gross_exposure = (
            abs(value1) + abs(value2)
        )

        cash_list.append(cash)
        q1_list.append(q1)
        q2_list.append(q2)
        value1_list.append(value1)
        value2_list.append(value2)
        gross_list.append(gross_exposure)
        total_list.append(total)

    df["KO_Shares"] = q1_list
    df["PEP_Shares"] = q2_list

    df["KO_Value"] = value1_list
    df["PEP_Value"] = value2_list

    df["Gross_Exposure"] = gross_list
    df["Cash"] = cash_list
    df["Total"] = total_list

    return df
  
    
def compute_metrics(
    df,
    starting_capital,
    risk_free_rate=0.04,
):

    df = df.copy()

    trading_days = 252

    # Select the relevant equity series
    if "Close" in df.columns:
        df["Buy_Hold"] = (
            starting_capital
            / df["Close"].iloc[0]
        ) * df["Close"]

        equity_col = "Buy_Hold"
        strategy_name = "Buy Hold"

    elif "Total" in df.columns:
        equity_col = "Total"
        strategy_name = "Pairs"

    else:
        raise ValueError(
            "DataFrame must contain either "
            "'Close' or 'Total'."
        )

    equity = df[equity_col]

    # Daily simple returns
    df["Return"] = equity.pct_change()

    # Risk-free rate converted from annual to daily
    rf_daily = (
        (1 + risk_free_rate) ** (1 / trading_days)
        - 1
    )

    excess_returns = df["Return"] - rf_daily

    # Total return
    total_return = (
        equity.iloc[-1] / equity.iloc[0]
        - 1
    )

    # CAGR
    years = (
        (df.index[-1] - df.index[0]).days
        / 365.25
    )

    cagr = (
        (equity.iloc[-1] / equity.iloc[0])
        ** (1 / years)
        - 1
    )

    # Sharpe ratio
    sharpe = (
        excess_returns.mean()
        / excess_returns.std()
        * np.sqrt(trading_days)
    )

    # Drawdown
    df["Running_Max"] = equity.cummax()

    df["Drawdown"] = (
        equity / df["Running_Max"]
        - 1
    )

    max_drawdown = df["Drawdown"].min()

    # Calmar ratio
    calmar = (
        cagr / abs(max_drawdown)
        if max_drawdown != 0
        else np.nan
    )

    # Calendar-year returns
    yearly_returns = (
        (1 + df["Return"])
        .groupby(df.index.year)
        .prod()
        - 1
    )

    yearly_df = pd.DataFrame({
        "Return": yearly_returns,
    })

    # Summary table
    comparison_table = pd.DataFrame(
        {
            strategy_name: [
                total_return,
                cagr,
                sharpe,
                max_drawdown,
                calmar,
            ]
        },
        index=[
            "Total Return",
            "CAGR",
            "Sharpe",
            "Max Drawdown",
            "Calmar",
        ],
    )

    return df, comparison_table, yearly_df

    
def plot_performance(
    dataKO,
    dataPEP,
    dataSPY,
    dataPairs,
    yKO,
    yPEP,
    ySPY,
    yPairs,
):

    plt.figure(figsize=(18, 6))
    colors = sns.color_palette("colorblind")

    # Equity curves
    plt.subplot(1, 3, 1)
    plt.title("Equity Curve")

    plt.plot(
        dataKO.index,
        dataKO["Buy_Hold"] / dataKO["Buy_Hold"].iloc[0],
        color=colors[0],
        label="KO",
        linewidth=1,
    )

    plt.plot(
        dataPEP.index,
        dataPEP["Buy_Hold"] / dataPEP["Buy_Hold"].iloc[0],
        color=colors[1],
        label="PEP",
        linewidth=1,
    )

    plt.plot(
        dataSPY.index,
        dataSPY["Buy_Hold"] / dataSPY["Buy_Hold"].iloc[0],
        color=colors[2],
        label="SPY",
        linewidth=1,
    )

    plt.plot(
        dataPairs.index,
        dataPairs["Total"] / dataPairs["Total"].iloc[0],
        color="k",
        label="Pairs",
        linewidth=1.5,
    )

    plt.axhline(
        1.0,
        color="black",
        linewidth=0.8,
        linestyle="--",
    )

    plt.grid(True, alpha=0.3)
    plt.xlabel("Date")
    plt.ylabel("Wealth Multiple")
    plt.legend()


    # Drawdowns
    plt.subplot(1, 3, 2)
    plt.title("Drawdown")

    plt.plot(
        dataKO.index,
        dataKO["Drawdown"],
        color=colors[0],
        label="KO",
        linewidth=1,
    )

    plt.plot(
        dataPEP.index,
        dataPEP["Drawdown"],
        color=colors[1],
        label="PEP",
        linewidth=1,
    )

    plt.plot(
        dataSPY.index,
        dataSPY["Drawdown"],
        color=colors[2],
        label="SPY",
        linewidth=1,
    )

    plt.plot(
        dataPairs.index,
        dataPairs["Drawdown"],
        color="k",
        label="Pairs",
        linewidth=1.5,
    )

    plt.grid(True, alpha=0.3)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()


    # Calendar-year returns
    plt.subplot(1, 3, 3)
    plt.title("Calendar-Year Returns")

    years = yPairs.index
    x = np.arange(len(years))
    width = 0.2

    plt.bar(
        x - 1.5 * width,
        yKO.loc[years, "Return"],
        width,
        label="KO",
        color=colors[0],
    )

    plt.bar(
        x - 0.5 * width,
        yPEP.loc[years, "Return"],
        width,
        label="PEP",
        color=colors[1],
    )

    plt.bar(
        x + 0.5 * width,
        ySPY.loc[years, "Return"],
        width,
        label="SPY",
        color=colors[2],
    )

    plt.bar(
        x + 1.5 * width,
        yPairs.loc[years, "Return"],
        width,
        label="Pairs",
        color="k",
    )

    plt.axhline(
        0,
        color="black",
        linewidth=0.8,
        linestyle="--",
    )

    plt.xticks(
        x[::2],
        years[::2],
    )

    plt.grid(axis="y", alpha=0.3)
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.legend()

    plt.tight_layout()
    plt.show()