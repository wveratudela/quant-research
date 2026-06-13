import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf



def returns_volatility(df, trading_days=252):
    
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

    return mu_annual, sigma_annual, cov_annual, corr_matrix



def forward_returns(mu, cov, dist, n_sims, n_steps, df_t=4):

    """
    Cholesky-correlated forward returns (linear), from annual mu and cov.

    mu      : (N,) annual expected returns
    cov     : (N,N) annual covariance matrix
    dist    : 'normal' or 't'
    n_steps : number of time steps (e.g. days)
    n_sims  : number of simulations
    dt      : time step size (default 1/252)

    R       : (n_steps, n_sims, N) Correlated forward returns per step (not compounded).
    """

    dt = 1/252

    mu  = np.asarray(mu)
    cov = np.asarray(cov)
    n_assets = len(mu)

    # annual → step covariance
    cov_step = cov * dt
    L = np.linalg.cholesky(cov_step)  # (N,N)

    # uncorrelated shocks
    if dist == "normal":
        Z = np.random.normal(0, 1, size=(n_steps, n_sims, n_assets))
    elif dist == "t":
        Z = np.random.standard_t(df_t, size=(n_steps, n_sims, n_assets))
        # scale t to unit variance if you want strict cov_step matching
        Z *= np.sqrt((df_t - 2) / df_t)
    else:
        raise ValueError("dist must be 'normal' or 't'")

    # introduce correlation: (n_steps, n_sims, N) @ (N,N)^T
    eps = Z @ L.T

    # linear forward returns: μΔt + ε
    R = mu * dt + eps

    return R



def historical_returns(data, event, start_date=None, end_date=None):

    crisis_windows = [
        {"name": "Great Depression",           "start": "1929-08-01", "end": "1939-12-31"},
        {"name": "Black Monday",               "start": "1987-10-14", "end": "1987-12-31"},
        {"name": "1990-1991 Recession",        "start": "1990-07-01", "end": "1991-03-31"},
        {"name": "Emerging-Market Crisis",     "start": "1997-07-02", "end": "1998-12-31"},
        {"name": "Dot-com Bubble Burst",       "start": "2000-03-10", "end": "2002-10-04"},
        {"name": "Early 2000s Recession",      "start": "2001-03-01", "end": "2001-11-30"},
        {"name": "Great Recession",            "start": "2007-12-01", "end": "2009-06-30"},
        {"name": "European Debt Crisis",       "start": "2010-04-01", "end": "2012-07-26"},
        {"name": "COVID-19 Recession",         "start": "2020-02-01", "end": "2020-04-30"},
        {"name": "2022 Rate Shock",            "start": "2022-01-03", "end": "2022-12-31"},
    ]

    w = next(x for x in crisis_windows if x["name"] == event)

    df_historical = data.loc[w["start"] : w["end"]]

    R_historical = np.log(df_historical / df_historical.shift(1))

    # If the entire window is outside the data range → fail fast
    if R_historical.dropna(how="all").empty:
        raise ValueError(
            f"Window {w['name']} ({w['start']} → {w['end']}) "
            "is outside the available data range."
        )

    # Key fix: assets that didn't exist → return 0 contribution
    R_historical = R_historical.fillna(0)

    return R_historical



def loss_calculator(returns, base, portfolio_value):

    loss_pct = - (returns - base)
    loss_pnl = portfolio_value * loss_pct

    return loss_pct, loss_pnl



def losses_var_es(losses_pnl, losses_pct, CI=95):

    VaR_95 = np.quantile(losses_pnl, CI/100)                 # loss quantile
    ES_95  = losses_pnl[losses_pnl >= VaR_95].mean()            # average loss beyond VaR

    VaR_95_rel = np.quantile(losses_pct, CI/100)
    ES_95_rel  = losses_pct[losses_pct >= VaR_95_rel].mean()

    return VaR_95, ES_95, VaR_95_rel, ES_95_rel



def tail_risk_contribution(hist_returns, weights, tickers, X):
    
    hist_clean = hist_returns.ffill().dropna(how='all').fillna(0)
    returns = np.cumprod(1 + hist_clean)

    port_cumulative = np.cumprod(1 + hist_clean @ weights)       # shape (365,) — one value per day
    cutoff = np.quantile(port_cumulative, X)
    tail_mask = port_cumulative <= cutoff                        # boolean Series, shape (365,)

    # now apply mask to asset-level returns
    tail = returns[tail_mask]                                    # shape (n_tail, 10) — clean

    # loss contribution per asset per scenario = -(w_i * r_i)
    asset_loss_contrib = -((tail - 1) * weights)                 # shape (n_tail, n_assets)
    avg_asset_loss_contrib = asset_loss_contrib.mean(axis=0)     # shape (n_assets,)

    # normalize to 100% (sums to 100%)
    total = avg_asset_loss_contrib.sum()
    pct = avg_asset_loss_contrib * 100

    tail_contrib = pd.Series(pct, index=tickers).sort_values(ascending=False)

    return tail_contrib



def simulate_annual_returns(hist_returns, weights, n_sims, seed=314):
    """
    Simulate annual multivariate log-returns using Ledoit–Wolf shrunk covariance.

    hist_returns : DataFrame (T × N) daily log returns
    n_sims       : number of simulations
    """
    # Mean vector (annual)
    mu_daily = hist_returns.mean().to_numpy()          # shape (N,)
    mu_annual = mu_daily * 252

    # Shrunk covariance (daily)
    lw = LedoitWolf().fit(hist_returns.values)
    cov_daily_shrunk = lw.covariance_                  # shape (N, N)

    # Annualize
    cov_annual = cov_daily_shrunk * 252

    # Simulate annual log-returns
    rng = np.random.default_rng(seed)
    sim_log_ret = rng.multivariate_normal(mu_annual, cov_annual, size=n_sims)

    # Convert to simple returns
    sim_simple_ret = np.exp(sim_log_ret) - 1           # shape (n_sims, N)

    sim_portfolio_ret = sim_simple_ret @ weights

    return sim_portfolio_ret, mu_annual, cov_annual