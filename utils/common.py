import pandas as pd
import numpy as np
import yfinance as yf



def fetch_data(ticker, start, end):

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    df.columns = df.columns.get_level_values(0)
    df.columns.name = None

    df = df[["Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"])

    return df


def fetch_portfolio_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    # With auto_adjust=True, 'Close' is already adjusted
    prices = raw['Close']
    volumes = raw['Volume']
    prices = prices.ffill().dropna()
    volumes = volumes.ffill().dropna()
    return prices, volumes


def return_statistics(df, trading_days=252):
    returns = df.pct_change().dropna()

    mu_annual = returns.mean() * trading_days
    sigma_annual = returns.std() * np.sqrt(trading_days)
    cov_annual = returns.cov() * trading_days
    corr_matrix = returns.corr()

    return returns, mu_annual, sigma_annual, cov_annual, corr_matrix