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
