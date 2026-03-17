import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



def engineer_features(prices, volumes, TARGET):

    features = pd.DataFrame(index=prices.index)
    
    # ======================= Price-based Features (7) =======================
    
    # 5-day, 20-day, 60-day momentum (log return over window)
    mom_5  = np.log(prices / prices.shift(5))
    mom_20 = np.log(prices / prices.shift(20))
    mom_60 = np.log(prices / prices.shift(60))
    
    # MA20/MA50 ratio (are we in a golden cross regime?)
    MAp_20 = prices.rolling(window=20).mean()
    MAp_50 = prices.rolling(window=50).mean()
    MA_ratio = MAp_20 / MAp_50
    
    # SI-14 (overbought/oversold)
    low14  = prices.rolling(window=14).min()
    high14 = prices.rolling(window=14).max()
    Stoch_14 = 100 * (prices - low14) / (high14 - low14)
    
    # Bollinger Band position (where is the price within its recent range?)
    stdp_20 = prices.rolling(window=20).std()
    upper = MAp_20 + 2 * stdp_20
    lower = MAp_20 - 2 * stdp_20
    BBP = (prices - lower) / (upper - lower)
    
    # Realized volatility — 20-day rolling std of daily returns
    returns = prices.pct_change()
    real_vola_20 = returns.rolling(window=20).std()
    
    # What fraction of last 20 days were positive returns?
    positive_days = (returns > 0).rolling(20).mean()
    
    
    # ======================= Volume-based Features (5) =======================
    
    # Price-volume divergence — 5-day price change / 5-day volume change
    # Volume trend — 5-day MA of volume / 20-day MA of volume
    # High-volume day flag — binary, 1 if volume > 2x average
    
    # Volume ratio — today's volume / 20-day average volume
    VR = volumes / volumes.rolling(20).mean()
    
    # OBV normalized — cumulative volume direction
    daily_direction = np.sign(returns)
    OBV = (volumes * daily_direction).cumsum()
    OBV_normalized = (OBV - OBV.rolling(20).mean()) / OBV.rolling(20).std()
    
    # Price-volume divergence
    price_change_5 = np.log(prices / prices.shift(5))
    vol_change_5   = np.log(volumes.clip(lower=1) / volumes.shift(5).clip(lower=1))
    pv_divergence  = price_change_5 / (vol_change_5 + 1e-10)# small epsilon avoids division by zero
    pv_divergence = pv_divergence.clip(-10, 10) # when volume_change is near zero, the epsilon in the denominator produces an astronomically large ratio. You might want to clip the output:
    
    # Volume trend — 5-day MA / 20-day MA of volume
    MAv_5 = volumes.rolling(window=5).mean()
    MAv_20 = volumes.rolling(window=20).mean()
    volu_trend = MAv_5 / MAv_20
    
    # High volume flag — binary, 1 if volume > 2x 20-day average
    volu_avg_20 = volumes.rolling(20).mean()
    high_volu_flag = (volumes > 2*volu_avg_20).astype(int)
    
    
    # ======================= Cross-asset Features(3) =======================
    
    # BTC/AAPL rolling 20-day correlation (regime indicator)
    # Gold return over past 5 days (risk-off signal)
    # SPY 5-day momentum (broad market context)
    
    # Cross-asset returns (5-day)
    spy_ret_5  = np.log(prices['SPY'] / prices['SPY'].shift(5))
    gld_ret_5  = np.log(prices['GLD'] / prices['GLD'].shift(5))
    btc_ret_5  = np.log(prices['BTC-USD'] / prices['BTC-USD'].shift(5))
    
    # Rolling correlation between TARGET and SPY (20-day)
    aapl_daily = np.log(prices[TARGET] / prices[TARGET].shift(1))
    spy_daily  = np.log(prices['SPY'] / prices['SPY'].shift(1))
    rolling_corr = aapl_daily.rolling(20).corr(spy_daily)
    
    
    features["mom_5"] = mom_5[TARGET]
    features["mom_20"] = mom_20[TARGET]
    features["mom_60"] = mom_60[TARGET]
    features["MA_ratio"] = MA_ratio[TARGET]
    features["Stoch_14"] = Stoch_14[TARGET]
    features["BBP"] = BBP[TARGET]
    features["real_vola_20"] = real_vola_20[TARGET]
    features["positive_days"] = positive_days[TARGET]
    
    features["VR"] = VR[TARGET]
    features["OBV_normalized"] = OBV_normalized[TARGET]
    features["pv_divergence"] = pv_divergence[TARGET]
    features["volu_trend"] = volu_trend[TARGET]
    features["high_volu_flag"] = high_volu_flag[TARGET]
    
    features['spy_ret_5']    = spy_ret_5
    features['gld_ret_5']    = gld_ret_5
    features['btc_ret_5']    = btc_ret_5
    features['rolling_corr'] = rolling_corr    
    
    features = features.dropna()
    features.describe()
    
    return features



def model_training(cols, df_sampled):

    # Walk-forward parameters
    min_train = 50        # minimum samples before first prediction
    results = []  # reset
    
    for i in range(min_train, len(df_sampled)):
        train = df_sampled.iloc[:i]
        test = df_sampled.iloc[i:i+1]
        
        X_train = train[cols]
        y_train = train['target']
        X_test = test[cols]
        y_test = test['target']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Logistic Regression
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        pred_lr = lr.predict(X_test_scaled)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train_scaled, y_train)
        pred_rf = rf.predict(X_test_scaled)
    
        # XGBoost
        xgb = XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric='logloss')
        xgb.fit(X_train_scaled, y_train)
        pred_xgb = xgb.predict(X_test_scaled)
        
        results.append({
            'date': test.index[0],
            'actual': y_test.values[0],
            'pred_lr': pred_lr[0],
            'pred_rf': pred_rf[0],
            'pred_xgb': pred_xgb[0]
        })
    
    results_df = pd.DataFrame(results)  # rebuild after loop

    return results_df


def compute_metrics_ml(df_sampled, results_df):

    print(f"Baseline (always predict up): {df_sampled['target'].mean():.3f}")
    print()

    
    acc_lr = accuracy_score(results_df['actual'], results_df['pred_lr'])
    prec_lr = precision_score(results_df['actual'], results_df['pred_lr'])
    rec_lr = recall_score(results_df['actual'], results_df['pred_lr'])
    
    print(f"Logistic Regression:")
    print(f"  Accuracy:  {acc_lr:.3f}")
    print(f"  Precision: {prec_lr:.3f}")
    print(f"  Recall:    {rec_lr:.3f}")
    print()
    
    
    acc_rf = accuracy_score(results_df['actual'], results_df['pred_rf'])
    prec_rf = precision_score(results_df['actual'], results_df['pred_rf'])
    rec_rf = recall_score(results_df['actual'], results_df['pred_rf'])
    
    print(f"Random Forest:")
    print(f"  Accuracy:  {acc_rf:.3f}")
    print(f"  Precision: {prec_rf:.3f}")
    print(f"  Recall:    {rec_rf:.3f}")
    print()
    
    
    acc_xgb = accuracy_score(results_df['actual'], results_df['pred_xgb'])
    prec_xgb = precision_score(results_df['actual'], results_df['pred_xgb'])
    rec_xgb = recall_score(results_df['actual'], results_df['pred_xgb'])
    
    print(f"XG Boost:")
    print(f"  Accuracy:  {acc_xgb:.3f}")
    print(f"  Precision: {prec_xgb:.3f}")
    print(f"  Recall:    {rec_xgb:.3f}")
    print()
    
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(axes, 
                               ['pred_lr', 'pred_rf', 'pred_xgb'],
                               ['Logistic Regression', 'Random Forest', 'XGBoost']):
        cm = confusion_matrix(results_df['actual'], results_df[col])
        ConfusionMatrixDisplay(cm, display_labels=['Down', 'Up']).plot(ax=ax)
        ax=ax.set_title(title)
    plt.tight_layout()
    plt.show()

    
    
def signal_backtest(df, Signal, starting_capital):

    df = df.copy()
        
    # Identify crossover points
    df['Golden_Cross'] = (Signal == 1) & (Signal.shift(1) == 0)  # MA20 crosses above MA50
    df['Death_Cross'] = (Signal == 0) & (Signal.shift(1) == 1)   # MA20 crosses below MA50
    
    # Create a column for the current day's relationship between MA20 and MA50
    df['Cash'] = 0.0
    df['Portfolio_Value'] = 0.0
    df['Shares'] = 0.0
    
    cash = starting_capital
    shares = 0.0
    
    for idx, row in df.iterrows():
        price = row['Close']
        port_val = shares * price
        
        if row['Golden_Cross']:
            shares = cash // price
            port_val = shares * price
            cash -= port_val
        
        if row['Death_Cross']:
            cash += shares * price
            shares = 0
            port_val = 0
        
        df.loc[idx, 'Cash'] = cash
        df.loc[idx, 'Portfolio_Value'] = port_val
        df.loc[idx, 'Shares'] = shares
    
    df['Total'] = df['Portfolio_Value'] + df['Cash']
    df['Buy_Hold'] = (starting_capital / df['Close'].iloc[0]) * df['Close']
    
    return df