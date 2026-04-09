import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns


def add_signals(df, fast=20, slow=50):
    
    df = df.copy()
    
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Daily_Return'] = df['Close'].pct_change()
    df = df.dropna()

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


def run_backtest(df, starting_capital, tranche=False):

    df = df.copy()
    
    # Create a column for the current day's relationship between MA20 and MA50
    df['Cash'] = 0.0
    df['Portfolio_Value'] = 0.0
    df['Shares'] = 0.0
    
    cash = starting_capital
    shares = 0.0
    shares2 = 0.0
    
    in_gc_window = False
    gc_day_counter = 0
    
    for idx, row in df.iterrows():
        price = row['Close']
        port_val = shares * price
        
        if row['Golden_Cross']:
            if tranche:
                shares = (cash/2) // price
                in_gc_window = True
                gc_day_counter = 0
            else:
                shares = cash // price
            port_val = shares * price
            cash -= port_val
    
        
        if in_gc_window and tranche:
            gc_day_counter += 1
            if gc_day_counter == 10:
                if price > row['MA20'] and price > row['MA50']:
                    shares2 = cash // price
                    cash -= shares2 * price
                    shares += shares2
                    port_val += shares2 * price
                    
                in_gc_window = False
                gc_day_counter = 0
        
        if row['Death_Cross']:
            cash += shares * price
            shares = 0
            shares2 = 0
            port_val = 0
            in_gc_window = False
            gc_day_counter = 0

        
        df.loc[idx, 'Cash'] = cash
        df.loc[idx, 'Portfolio_Value'] = port_val
        df.loc[idx, 'Shares'] = shares
    
    df['Total'] = df['Portfolio_Value'] + df['Cash']
    df['Buy_Hold'] = (starting_capital / df['Close'].iloc[0]) * df['Close']

    return df


def compute_metrics(df,starting_capital):

    df = df.copy()
    
    trading_days = 252
    risk_free_rate = 4/100
    rf_daily = risk_free_rate / trading_days
    years = (df.index[-1] - df.index[0]).days / 365.25

    
    #Buy Sell
    Total_return_BS = (df.iloc[-1]['Total']-starting_capital)/starting_capital
    
    tot_log_returns = np.log(df['Total'] / df['Total'].shift(1))
    excess_return = tot_log_returns - rf_daily
    mu = excess_return.mean()
    sigma = excess_return.std()
    Sharpe_ratio_BS = (mu / sigma) * np.sqrt(trading_days)
    
    df['Running_Max'] = df['Total'].cummax()
    df['Drawdown_BS'] = df['Total'] / df['Running_Max'] - 1
    Maximum_drawdown_BS = df['Drawdown_BS'].min()

    start_val = df['Total'].iloc[0]
    end_val   = df['Total'].iloc[-1]
    CAGR_BS = (end_val / start_val) ** (1 / years) - 1
    
    Calmar_BS = CAGR_BS / Maximum_drawdown_BS

    # Calculate the difference between adjacent elements
    diff = np.diff(df['Signal'])
    # Rising edges (0 to 1 transitions) are where diff == 1
    rising_indices = np.where(diff == 1)[0]
    # Falling edges (1 to 0 transitions) are where diff == -1
    falling_indices = np.where(diff == -1)[0]
    buy = df.iloc[rising_indices+1]['Portfolio_Value'].values
    sell = df.iloc[falling_indices]['Portfolio_Value'].values
    
    if len(sell) == len(buy):
        gains = sell-buy
    else:
        gains = sell-buy[0:-1]
        print('Last buy still holding.')
    
    pos = np.sum(gains > 0)
    neg = np.sum(gains < 0)
    tot = len(gains)
    
    Win_rate_BS = pos/tot

    df['Year'] = df.index.year
    yearly_start = df.groupby('Year')['Total'].first()
    yearly_end   = df.groupby('Year')['Total'].last()
    days = df.groupby('Year').apply(lambda x: (x.index[-1] - x.index[0]).days,
    include_groups=False)
    annualized_BS = (yearly_end / yearly_start) ** (365.25 / days) - 1




    
    #Buy Hold
    Total_return_BH = (df.iloc[-1]['Buy_Hold']-starting_capital)/starting_capital
    
    tot_log_returns = np.log(df['Buy_Hold'] / df['Buy_Hold'].shift(1))
    excess_return = tot_log_returns - rf_daily
    mu = excess_return.mean()
    sigma = excess_return.std()
    Sharpe_ratio_BH = (mu / sigma) * np.sqrt(trading_days)
    
    df['Running_Max'] = df['Buy_Hold'].cummax()
    df['Drawdown_BH'] = df['Buy_Hold'] / df['Running_Max'] - 1
    Maximum_drawdown_BH = df['Drawdown_BH'].min()

    start_val = df['Buy_Hold'].iloc[0]
    end_val   = df['Buy_Hold'].iloc[-1]
    CAGR_BH = (end_val / start_val) ** (1 / years) - 1
    
    Calmar_BH = CAGR_BH / Maximum_drawdown_BH

    if Total_return_BH > 0:
        Win_rate_BH = 1
    else:
        Win_rate_BH = 0

    df['Year'] = df.index.year
    yearly_start = df.groupby('Year')['Buy_Hold'].first()
    yearly_end   = df.groupby('Year')['Buy_Hold'].last()
    days = df.groupby('Year').apply(lambda x: (x.index[-1] - x.index[0]).days,
    include_groups=False)
    annualized_BH = (yearly_end / yearly_start) ** (365.25 / days) - 1

    yearly_df = pd.DataFrame({
    'Start': yearly_start,
    'End': yearly_end,
    'Days': days,
    'Annualized_BS': annualized_BS,
    'Annualized_BH': annualized_BH
    })

    
    
    data = {
        'Buy Sell': [Total_return_BS, Sharpe_ratio_BS, Maximum_drawdown_BS, CAGR_BS, Calmar_BS, Win_rate_BS],
        'Buy Hold': [Total_return_BH, Sharpe_ratio_BH, Maximum_drawdown_BH, CAGR_BH, Calmar_BH, Win_rate_BH]
    }
    
    index = ['Total Return', 'Sharpe', 'Max Drawdown', 'CAGR', 'Calmar', 'Win Rate']
    
    comparison_table = pd.DataFrame(data, index=index)

    return df, comparison_table, yearly_df

    
def plot_performance(dA,dS,yA,yS):

    plt.figure(figsize=(18, 6))
    colors = sns.color_palette("colorblind")
    
    plt.subplot(1, 3, 1)
    plt.title('Equity curve')
    plt.plot(dA.index, dA['Total'], label='Total A', linewidth=1, color=colors[0])
    plt.plot(dA.index, dA['Buy_Hold'], label='HODL A', linewidth=1, color=colors[1])
    plt.plot(dS.index, dS['Buy_Hold'], label='SPY', linewidth=0.5, color=colors[2])
    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.title('Drawdown chart')
    plt.plot(dA.index, dA['Drawdown_BS'], label='Total A', linewidth=1, color=colors[0])
    plt.plot(dA.index, dA['Drawdown_BH'], label='HODL A', linewidth=1, color=colors[1])
    plt.plot(dS.index, dS['Drawdown_BH'], label='SPY', linewidth=0.5, color=colors[2])
    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.title('Annual returns bar chart')
    
    years = yA.index
    x = np.arange(len(years))        # numeric positions for each year
    width = 0.25                     # width of each bar
    
    plt.bar(x - width, yA['Annualized_BS'], width, label='Total A', color=colors[0])
    plt.bar(x, yA['Annualized_BH'], width, label='HODL A', color=colors[1])
    plt.bar(x + width, yS['Annualized_BH'], width, label='SPY', color=colors[2])
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(x[::2], years[::2])             # replace numeric x with year labels, showing every second tick to avoid overlapping
    plt.grid(axis='y', alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Return')
    
    plt.legend()
    
    plt.tight_layout()
    plt.show()