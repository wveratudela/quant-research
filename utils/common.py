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

