import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns


def fetch_data(ticker, start, end):

    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)

    df = df[['Close', 'Volume']]
    df = df.ffill()

    return df


def fetch_portfolio_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, 
                      auto_adjust=False, progress=False)
    prices = raw['Close']
    prices = prices.ffill().dropna()
    
    return prices


def market_cap(tickers):

    market_caps = {}
    
    for t in tickers:
        info = yf.Ticker(t).info
        market_caps[t] = info.get("marketCap", None)
    
    market_caps_df = pd.DataFrame.from_dict(market_caps, orient="index", columns=["MarketCap"])
    market_caps_df = market_caps_df.sort_index()
    
    return market_caps_df
