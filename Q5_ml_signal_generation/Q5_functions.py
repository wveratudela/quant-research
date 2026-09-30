import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.dates as mdates


def engineer_features(prices, volumes, target, fast=20, slow=50):

    price = prices[target]
    volume = volumes[target]

    features = pd.DataFrame(index=prices.index)

    # ======================= Price-based features (8) =======================

    # Momentum
    features["mom_5"] = np.log(price / price.shift(5))
    features["mom_20"] = np.log(price / price.shift(20))
    features["mom_60"] = np.log(price / price.shift(60))

    # Moving-average ratio
    ma_fast = price.rolling(fast).mean()
    ma_slow = price.rolling(slow).mean()
    features["MA_ratio"] = ma_fast / ma_slow

    # Position within the 14-day price range
    low_14 = price.rolling(14).min()
    high_14 = price.rolling(14).max()
    features["Stoch_14"] = 100 * (price - low_14) / (high_14 - low_14)

    # Bollinger Band position
    std_20 = price.rolling(20).std()
    upper = ma_fast + 2 * std_20
    lower = ma_fast - 2 * std_20
    features["BBP"] = (price - lower) / (upper - lower)

    # Realized volatility
    returns = price.pct_change(fill_method=None)
    features["real_vola_20"] = returns.rolling(20).std()

    # Fraction of positive trading days
    features["positive_days"] = (returns > 0).rolling(20).mean()

    # ======================= Volume-based features (5) ======================

    volume_avg_20 = volume.rolling(20).mean()

    # Current volume relative to recent average
    features["VR"] = volume / volume_avg_20

    # Normalized On-Balance Volume
    direction = np.sign(returns)
    obv = (volume * direction).cumsum()
    features["OBV_normalized"] = (
        (obv - obv.rolling(20).mean())
        / obv.rolling(20).std()
    )

    # Price-volume divergence
    price_change_5 = np.log(price / price.shift(5))
    volume_change_5 = np.log(
        volume.clip(lower=1) / volume.shift(5).clip(lower=1)
    )

    features["pv_divergence"] = (
        price_change_5 / (volume_change_5 + 1e-10)
    ).clip(-10, 10)

    # Short-term volume trend
    volume_avg_5 = volume.rolling(5).mean()
    features["volu_trend"] = volume_avg_5 / volume_avg_20

    # Unusually high-volume day
    features["high_volu_flag"] = (
        volume > 2 * volume_avg_20
    ).astype(int)

    # ======================= Market-context features (2) ====================

    spy_price = prices["SPY"]

    # Recent broad-market momentum
    features["spy_ret_5"] = np.log(
        spy_price / spy_price.shift(5)
    )

    # Rolling target/SPY correlation
    target_daily = np.log(price / price.shift(1))
    spy_daily = np.log(spy_price / spy_price.shift(1))

    features["rolling_corr"] = (
        target_daily.rolling(20).corr(spy_daily)
    )

    return features.dropna()



def model_training(cols, df, model, min_train=50, prediction_start=None):

    results = []

    for i in range(min_train, len(df)):

        test_date = df.index[i]

        if prediction_start is not None and test_date < pd.Timestamp(prediction_start):
            continue

        train = df.iloc[:i]
        test = df.iloc[i:i+1]

        X_train = train[cols]
        y_train = train["target"]
        X_test = test[cols]
        y_test = test["target"]

        if model == "lr":
            scaler = StandardScaler()
            X_train_model = scaler.fit_transform(X_train)
            X_test_model = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000)

        elif model == "rf":
            X_train_model = X_train
            X_test_model = X_test

            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            )

        elif model == "xgb":
            X_train_model = X_train
            X_test_model = X_test

            clf = XGBClassifier(
                n_estimators=100,
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
            )

        else:
            raise ValueError("model must be 'lr', 'rf', or 'xgb'")

        clf.fit(X_train_model, y_train)
        prediction = clf.predict(X_test_model)[0]

        results.append({
            "date": test_date,
            "actual": y_test.iloc[0],
            "prediction": prediction,
        })

    return pd.DataFrame(results)


def compute_metrics_ml(results_df):

    y_true = results_df["actual"]
    y_pred = results_df["prediction"]

    baseline_accuracy = y_true.mean()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"Always-up baseline accuracy: {baseline_accuracy:.3f}")
    print()
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        cm,
        display_labels=["Down", "Up"],
    ).plot(ax=ax)

    ax.set_title("Final ML Model")
    plt.tight_layout()
    plt.show()

    
    
def signal_backtest(df, signal, starting_capital, initial_long=False):

    df = df.copy()
    signal = signal.reindex(df.index).ffill().fillna(0).astype(int)

    df["Entry"] = (signal == 1) & (signal.shift(1, fill_value=0) == 0)
    df["Exit"] = (signal == 0) & (signal.shift(1, fill_value=0) == 1)

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
        if row["Entry"] and shares == 0:
            pending_entry = True

        elif row["Exit"] and shares > 0:
            pending_exit = True

        portfolio_value = shares * price

        df.loc[idx, "Cash"] = cash
        df.loc[idx, "Portfolio_Value"] = portfolio_value
        df.loc[idx, "Shares"] = shares

    df["Total"] = df["Cash"] + df["Portfolio_Value"]

    bh_shares = starting_capital / df["Close"].iloc[0]
    df["Buy_Hold"] = bh_shares * df["Close"]

    return df


def plot_strategy_comparison(data_q1, data_ml):

    comparison = pd.DataFrame({
        "MA20/MA50": data_q1["Total"],
        "XGBoost ML": data_ml["Total"],
        "Buy & Hold": data_ml["Buy_Hold"],
    })

    growth = comparison / comparison.iloc[0]
    drawdown = comparison / comparison.cummax() - 1

    # Calendar-year returns, including the first partial/full evaluation year
    daily_returns = comparison.pct_change(fill_method=None)
    yearly_returns = (
        (1 + daily_returns)
        .groupby(daily_returns.index.year)
        .prod()
        - 1
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = sns.color_palette("colorblind")

    # Growth of initial capital
    for i, col in enumerate(growth.columns):
        axes[0].plot(growth.index, growth[col], label=col, color=colors[i])

    axes[0].set_title("Growth of Initial Capital")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Wealth Multiple")
    axes[0].set_xlim(comparison.index.min(), comparison.index.max())
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Drawdown
    for i, col in enumerate(drawdown.columns):
        axes[1].plot(drawdown.index, drawdown[col], label=col, color=colors[i])

    axes[1].set_title("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlim(comparison.index.min(), comparison.index.max())
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # Calendar-year returns
    years = yearly_returns.index
    x = np.arange(len(years))
    width = 0.25

    for i, col in enumerate(yearly_returns.columns):
        axes[2].bar(
            x + (i - 1) * width,
            yearly_returns[col],
            width,
            label=col,
            color=colors[i],
        )

    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(years)
    axes[2].set_title("Calendar-Year Returns")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Return")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def strategy_summary(data_q1, data_ml):

    comparison = pd.DataFrame({
        "MA20/MA50": data_q1["Total"],
        "XGBoost ML": data_ml["Total"],
        "Buy & Hold": data_ml["Buy_Hold"],
    })

    returns = comparison.pct_change(fill_method=None)

    years = (
        comparison.index[-1] - comparison.index[0]
    ).days / 365.25

    summary = pd.DataFrame(
        index=[
            "Total Return",
            "CAGR",
            "Max Drawdown",
            "Annualized Volatility",
        ],
        columns=comparison.columns,
        dtype=float,
    )

    summary.loc["Total Return"] = comparison.iloc[-1] / comparison.iloc[0] - 1

    summary.loc["CAGR"] = (
        comparison.iloc[-1] / comparison.iloc[0]
    ) ** (1 / years) - 1

    summary.loc["Max Drawdown"] = (
        comparison / comparison.cummax() - 1
    ).min()

    summary.loc["Annualized Volatility"] = (
        returns.std() * np.sqrt(252)
    )

    return summary