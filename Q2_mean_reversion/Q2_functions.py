import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.linear_model import LinearRegression


def check_data(df1, df2):

    result = adfuller(df1['Close'])
    print(f"V ADF: {result[0]:.4f}, p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print(f"- - - - - WARNING: p-value: {result[1]:.4f} < 5% - - - - -")
    
    result = adfuller(df2['Close'])
    print(f"MA ADF: {result[0]:.4f}, p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print(f"- - - - - WARNING: p-value: {result[1]:.4f} < 5% - - - - -")
    
    score, p_value, critical_values = coint(df1['Close'], df2['Close'])
    print(f"Cointegration p-value: {p_value:.4f}")
    if p_value > 0.05:
        print(f"- - - - - WARNING: p-value: {p_value:.4f} > 5% - - - - -")
    print(f"Critical values: {critical_values}")
    

def check_signals(df1, df2):

    # Regress V on MA
    X = df2['Close'].values.reshape(-1, 1)
    y = df1['Close'].values
    
    model = LinearRegression().fit(X, y)
    beta = model.coef_[0]
    print(f"Hedge ratio β: {beta:.4f}")
    
    r2 = model.score(X, y)
    print(f"R²: {r2:.4f}")
    if r2 < 0.95:
        print(f"- - - - - WARNING: p-value: {p_value:.4f} < 95% - - - - -")

    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    
    spread = df1['Close'] - beta * df2['Close']

    print(f"Spread mean: {spread.mean():.4f}")
    print(f"Spread std: {spread.std():.4f}")


    # Visualize the signals to check for correlation and spread
    plt.figure(figsize=(18, 6))
    colors = sns.color_palette("colorblind")
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color=colors[0], label='Prices')      # Scatter of original data
    plt.plot(x_line, y_line, color='k', linewidth=2, label='Fit',linestyle='--')      # Regression line
    plt.title('MA & V correlation of values')
    plt.xlabel('MA')
    plt.ylabel('V')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(spread, linewidth=1)
    plt.axhline(spread.mean(), color='black', linewidth=0.8, linestyle='--', alpha=0.75, label='Mean')
    plt.axhline(spread.mean()+spread.std(), color='black', linewidth=1, linestyle='--', alpha=0.5, label='+1 std', )
    plt.axhline(spread.mean()-spread.std(), color='black', linewidth=1, linestyle='--', alpha=0.5, label='-1 std')
    plt.title('V - 0.55 × MA Spread')
    plt.xlabel('Year')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

    
    rolling_mean = spread.rolling(window=252).mean()
    rolling_std = spread.rolling(window=60).std()
    z_score = (spread - rolling_mean) / rolling_std
    
    s = z_score
    signal = pd.Series(0, index=s.index)
    
    state = 0
    
    for i in range(1, len(s)):
        if state == 0:
            if s.iloc[i] > 2:
                state = 1
            elif s.iloc[i] < -2:
                state = -1
    
        elif state == 1 and s.iloc[i] < 0:
            state = 0
    
        elif state == -1 and s.iloc[i] > 0:
            state = 0
    
        signal.iloc[i] = state
    
    plt.figure(figsize=(18, 6))
    colors = sns.color_palette("colorblind")

    plt.subplot(1, 2, 1)
    plt.plot(z_score, linewidth=1, label='Z-score')
    plt.axhline(2, color='red', linestyle='--', linewidth=1, label='+2')
    plt.axhline(-2, color='green', linestyle='--', linewidth=1, label='-2')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('Z-score of V/MA Spread (60d std, 252d mean)')
    plt.xlabel('Year')
    plt.ylabel('z-score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(signal)    
    plt.title('Long/Short signal (1 = Long MA/Short V. -1 = Long V/Short MA)')
    plt.xlabel('Year')
    plt.ylabel('Signal')
    plt.grid(True, alpha=0.3)
    plt.show()

    return signal


def run_pairs_trading(df1, df2, signal, starting_capital):

    cash = starting_capital
    port_val = 0
    long_shares = 0
    short_shares = 0
    short_proceeds = 0
    long_leg = 0
    short_leg = 0
    
    index = np.arange(len(signal))   # or use your real index
    df = pd.DataFrame({'Signal': signal}, index=df1.index)
    df.index = pd.to_datetime(df.index)
    
    long_shares_list = []
    short_shares_list = []
    long_leg_list = []
    short_leg_list = []
    port_val_list = []
    cash_list = []
    
    for i in range(len(signal)):
        sig = signal.iloc[i]
        price_1 = df1['Close'].iloc[i]
        price_2 = df2['Close'].iloc[i]
        
        if sig == 1 and signal.iloc[i-1] == 0: # long_spread
            long_shares = (cash/2) // price_2
            short_shares = (cash/2) // price_1
    
            long_leg = long_shares * price_2
            short_proceeds = short_shares * price_1
           
            cash -= long_leg
            cash += short_proceeds
            port_val = long_leg + short_proceeds
    
        elif sig == 0 and signal.iloc[i-1] == 1: # none - reset
            long_leg = long_shares * price_2
            short_leg = short_shares * price_1
    
            cash += long_leg
            cash -= short_leg
            
            long_shares = 0
            short_shares = 0
            short_proceeds = 0
            port_val = 0
    
        elif sig == -1 and signal.iloc[i-1] == 0: # long_spread
            long_shares = (cash/2) // price_1
            short_shares = (cash/2) // price_2
    
            long_leg = long_shares * price_1
            short_proceeds = short_shares * price_2
            
            cash -= long_leg
            cash += short_proceeds
            port_val = long_leg + short_proceeds
    
        elif sig == 0 and signal.iloc[i-1] == -1: # none - reset
            long_leg = long_shares * price_1
            short_leg = short_shares * price_2
    
            cash += long_leg
            cash -= short_leg
            
            long_shares = 0
            short_shares = 0
            short_proceeds = 0
            port_val = 0
    
        if sig == 1 and signal.iloc[i-1] == 1:  # holding: long MA, short V
            long_leg = long_shares * price_2
            short_leg = short_proceeds - short_shares * price_1
            port_val = long_leg + short_leg
    
        elif sig == -1 and signal.iloc[i-1] == -1:  # holding: long V, short MA
            long_leg = long_shares * price_1
            short_leg = short_proceeds - short_shares * price_2
            port_val = long_leg + short_leg
        else:
            port_val = 0
    
        
        long_shares_list.append(long_shares)
        short_shares_list.append(short_shares)
        long_leg_list.append(long_leg)
        short_leg_list.append(short_leg)
        port_val_list.append(port_val)
        cash_list.append(cash)
    
    
    df['Long_Shares'] = long_shares_list
    df['Short_Shares'] = short_shares_list
    df['Long_Leg'] = long_leg_list
    df['Short_Leg'] = short_leg_list
    df['Portfolio_Value'] = port_val_list
    df['Cash'] = cash_list
    df['Total'] = df['Cash'] + df['Portfolio_Value']

    return df
  
    
def compute_metrics(df,starting_capital):

    df = df.copy()

    if 'Close' in df.columns:
        df['Buy_Hold'] = (starting_capital / df['Close'].iloc[0]) * df['Close']
    else:
        df['Buy_Hold'] = df['Cash']
        
    trading_days = 252
    risk_free_rate = 4/100
    rf_daily = risk_free_rate / trading_days
    years = (df.index[-1] - df.index[0]).days / 365.25
    
    #Buy Hold
    Total_return = (df.iloc[-1]['Buy_Hold']-starting_capital)/starting_capital
    
    tot_log_returns = np.log(df['Buy_Hold'] / df['Buy_Hold'].shift(1))
    excess_return = tot_log_returns - rf_daily
    mu = excess_return.mean()
    sigma = excess_return.std()
    Sharpe_ratio = (mu / sigma) * np.sqrt(trading_days)
    
    df['Running_Max'] = df['Buy_Hold'].cummax()
    df['Drawdown'] = df['Buy_Hold'] / df['Running_Max'] - 1
    Maximum_drawdown = df['Drawdown'].min()

    start_val = df['Buy_Hold'].iloc[0]
    end_val   = df['Buy_Hold'].iloc[-1]
    CAGR = (end_val / start_val) ** (1 / years) - 1
    
    Calmar = CAGR / Maximum_drawdown

    df['Year'] = df.index.year
    yearly_start = df.groupby('Year')['Buy_Hold'].first()
    yearly_end   = df.groupby('Year')['Buy_Hold'].last()
    days = df.groupby('Year').apply(lambda x: (x.index[-1] - x.index[0]).days,
    include_groups=False)
    annualized = (yearly_end / yearly_start) ** (365.25 / days) - 1

    yearly_df = pd.DataFrame({
    'Start': yearly_start,
    'End': yearly_end,
    'Days': days,
    'Annualized': annualized
    })

    if 'Close' in df.columns:
        data = {
        'Buy Hold': [Total_return, Sharpe_ratio, Maximum_drawdown, CAGR, Calmar]
        }
    else:
        data = {
        'Pairs': [Total_return, Sharpe_ratio, Maximum_drawdown, CAGR, Calmar]
        }    
    
    
    index = ['Total Return', 'Sharpe', 'Max Drawdown', 'CAGR', 'Calmar']
    
    comparison_table = pd.DataFrame(data, index=index)

    return df, comparison_table, yearly_df

    
def plot_performance(dataV,dataM,dataS,dataP,yV,yM,yS,yP):

    plt.figure(figsize=(18, 6))
    colors = sns.color_palette("colorblind")
    
    plt.subplot(1, 3, 1)
    plt.title('Equity curve')
    plt.plot(dataV.index, dataV['Buy_Hold'], color=colors[0], label='Buy_Hold V', linewidth=1)
    plt.plot(dataM.index, dataM['Buy_Hold'], color=colors[1], label='Buy_Hold MA', linewidth=1)
    plt.plot(dataS.index, dataS['Buy_Hold'], color=colors[2], label='Buy_Hold SPY', linewidth=1)    
    plt.plot(dataS.index, dataP['Total'], color='k', label='Pairs V/MA', linewidth=1.5)    

    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.title('Drawdown chart')
    plt.plot(dataV.index, dataV['Drawdown'], color=colors[0], label='Buy_Hold V', linewidth=1)
    plt.plot(dataM.index, dataM['Drawdown'], color=colors[1], label='Buy_Hold MA', linewidth=1)
    plt.plot(dataS.index, dataS['Drawdown'], color=colors[2], label='Buy_Hold SPY', linewidth=1)    
    plt.plot(dataS.index, dataP['Drawdown'], color='k', label='Pairs V/MA', linewidth=1.5)    

    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.title('Annual returns bar chart')
    
    years = yP.index
    x = np.arange(len(years))
    width = 0.2
    
    plt.bar(x - 1.5*width, yV['Annualized'], width, label='V', color=colors[0])
    plt.bar(x - 0.5*width, yM['Annualized'], width, label='MA', color=colors[1])
    plt.bar(x + 0.5*width, yS['Annualized'], width, label='SPY', color=colors[2])
    plt.bar(x + 1.5*width, yP['Annualized'], width, label='Pairs', color='k')
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(x[::2], years[::2])             # replace numeric x with year labels, showing every second tick to avoid overlapping
    plt.grid(axis='y', alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Return')
    
    plt.legend()
    
    plt.tight_layout()
    plt.show()