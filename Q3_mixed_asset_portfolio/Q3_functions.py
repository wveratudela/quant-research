import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from scipy.optimize import minimize


def portfolio_return(w, mu):
    return w @ mu


def portfolio_volatility(w, cov):
    return np.sqrt(w @ cov @ w)


def global_min_variance(cov):
    n = len(cov)

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        }
    ]

    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    return minimize(
        portfolio_volatility,
        w0,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )


def frontier_optimizer(
    mu_annual,
    cov_annual,
    target_returns,
):

    frontier_volatilities = []
    frontier_returns = []
    frontier_weights = []

    for target in target_returns:

        result = min_variance(
            target,
            mu_annual,
            cov_annual,
        )

        if result.success:
            frontier_volatilities.append(
                portfolio_volatility(
                    result.x,
                    cov_annual,
                )
            )

            frontier_returns.append(target)
            frontier_weights.append(result.x)

    return (
        frontier_volatilities,
        frontier_returns,
        frontier_weights,
    )


def portfolio_value(mu, sigma, end, n_sims=10_000, Y=1):

    sim_start = end
    sim_end   = end + pd.DateOffset(years=Y)
    dates = pd.bdate_range(start=sim_start.date(), end=sim_end.date(), freq="B")

    dt = 1 / 252
    trading_days = len(dates)
    Z = np.random.normal(0, 1, size=(trading_days, n_sims))

    mu_daily = (mu - 0.5 * sigma**2) * dt
    sigma_daily = sigma * np.sqrt(dt)
    daily_returns = np.exp(mu_daily + sigma_daily * Z)

    paths = 100 * daily_returns.cumprod(axis=0)
    paths = pd.DataFrame(paths)

    paths.index = dates

    return paths

    
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


def historical_portfolio_value(
    prices,
    weights,
    starting_capital,
):
    normalized_prices = (
        prices / prices.iloc[0]
    )

    portfolio_growth = (
        normalized_prices
        @ np.asarray(weights)
    )

    return (
        starting_capital
        * portfolio_growth
    )