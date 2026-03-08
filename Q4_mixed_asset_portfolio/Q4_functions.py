import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from scipy.optimize import minimize


def returns_volatility(df):
    
    trading_days = 252
    
    returns = df.pct_change().dropna()
    
    mu = returns.mean()
    sigma = returns.std()
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()
    
    mu_annual = (1 + mu)**trading_days - 1
    sigma_annual = sigma * np.sqrt(trading_days)
    cov_annual = cov_matrix * trading_days
    
    # combine into table
    stats = pd.concat([mu_annual, sigma_annual], axis=1)
    stats.columns = ['Annualized_Return', 'Annualized_Volatility']
    display(stats)
        
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cov_annual, annot=True, fmt='.3f', cmap="viridis")
    plt.title("Covariance Matrix (Annual)")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="viridis")
    plt.title("Correlation Matrix (Daily)")
    
    plt.show()

    return mu_annual, sigma_annual, cov_annual


def portfolio_return(w, mu):
    return w @ mu


def portfolio_volatility(w, cov):
    return np.sqrt(w @ cov @ w)


def frontier_optimizer(mu_annual, sigma_annual, cov_annual, target_returns):
    frontier_volatilities = []
    frontier_returns = []
    frontier_weights = []
    
    for target in target_returns:
        result = min_variance(target, mu_annual, cov_annual)
        if result.success:
            frontier_volatilities.append(portfolio_volatility(result.x, cov_annual))
            frontier_returns.append(target)
            frontier_weights.append(result.x)
    
    return frontier_volatilities, frontier_returns, frontier_weights


def portfolio_value(initial_capital,df,w):

    # 1. Capital allocated per asset
    capital_alloc = initial_capital * w
    # 2. First-day prices
    first_prices = df.iloc[0]
    # 3. Integer number of shares
    shares = np.floor(capital_alloc / first_prices)
    # 4. Daily position values
    position_values = df * shares
    # 5. Portfolio value over time
    portfolio_value = position_values.sum(axis=1)

    return portfolio_value

    
def max_sharpe(mu, cov, rf=0.04):
    n = len(mu)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    
    result = minimize(neg_sharpe, w0, args=(mu, cov, rf),
                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result

    
def min_variance(target_return, mu, cov):
    n = len(mu)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - target_return}
    ]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n  # equal weight starting point
    
    result = minimize(portfolio_volatility, w0, args=(cov,),
                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def neg_sharpe(w, mu, cov, rf=0.04):
    ret = portfolio_return(w, mu)
    vol = portfolio_volatility(w, cov)
    return -(ret - rf) / vol

