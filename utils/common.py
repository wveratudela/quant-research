import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns


def fetch_data(ticker, start, end):

    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    # With auto_adjust=True, 'Close' is already adjusted
    df.columns = df.columns.get_level_values(0)

    df = df[['Close', 'Volume']]
    df = df.ffill()

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


def market_cap(tickers):

    market_caps = {}
    
    for t in tickers:
        info = yf.Ticker(t).info
        
        mc = info.get("marketCap")
        if mc is None:
            mc = info.get("totalAssets")     # ETFs
        if mc is None:
            supply = info.get("circulatingSupply")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if supply and price:
                mc = supply * price       # crypto fallback
        
        market_caps[t] = mc
    
    market_caps_df = pd.DataFrame.from_dict(market_caps, orient="index", columns=["MarketCap"])
    
    return market_caps_df


def compute_vol_thresholds(returns):
    sigma = returns.rolling(252).std().dropna()
    return {
        "low":    sigma.quantile(0.25),
        "medium": sigma.quantile(0.50),
        "high":   sigma.quantile(0.90),
        "sigma":  sigma
    }


def get_ma_windows(current_sigma, thresholds):
    low    = thresholds["low"]
    medium = thresholds["medium"]
    high   = thresholds["high"]

    if current_sigma > high:
        return 10, 30
    elif current_sigma > medium:
        return 15, 40
    elif current_sigma < low:
        return 50, 200
    else:
        return 20, 50
