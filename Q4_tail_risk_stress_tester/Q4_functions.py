import numpy as np



def format_weight(x):
    return "-" if np.isclose(x, 0.0) else f"{x:.1%}"


def format_tail(x):
    return "-" if np.isclose(x, 0.0) else f"{x:.2f}"


def forward_returns(
    mu,
    cov,
    dist,
    n_sims,
    n_steps,
    df_t=6,
    seed=42,
):
    """
    Simulate correlated simple returns from annual expected returns
    and covariance.
    """

    dt = 1 / 252

    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n_assets = len(mu)

    cov_step = cov * dt
    L = np.linalg.cholesky(cov_step)

    rng = np.random.default_rng(seed)

    if dist == "normal":
        Z = rng.normal(
            0,
            1,
            size=(n_steps, n_sims, n_assets),
        )

    elif dist == "t":
        Z = rng.standard_t(
            df_t,
            size=(n_steps, n_sims, n_assets),
        )

        Z *= np.sqrt((df_t - 2) / df_t)

    else:
        raise ValueError("dist must be 'normal' or 't'")

    eps = Z @ L.T

    return mu * dt + eps


def historical_returns(data, event):
    crisis_windows = [
        {"name": "Great Depression",        "start": "1929-08-01", "end": "1939-12-31"},
        {"name": "Black Monday",            "start": "1987-10-14", "end": "1987-12-31"},
        {"name": "1990-1991 Recession",     "start": "1990-07-01", "end": "1991-03-31"},
        {"name": "Emerging-Market Crisis",  "start": "1997-07-02", "end": "1998-12-31"},
        {"name": "Dot-com Bubble Burst",    "start": "2000-03-10", "end": "2002-10-04"},
        {"name": "Early 2000s Recession",   "start": "2001-03-01", "end": "2001-11-30"},
        {"name": "Great Recession",         "start": "2007-12-01", "end": "2009-06-30"},
        {"name": "European Debt Crisis",    "start": "2010-04-01", "end": "2012-07-26"},
        {"name": "COVID-19 Recession",      "start": "2020-02-01", "end": "2020-04-30"},
        {"name": "2022 Rate Shock",         "start": "2022-01-03", "end": "2022-12-31"},
    ]

    window = next(
        x for x in crisis_windows
        if x["name"] == event
    )

    crisis_prices = data.loc[
        window["start"]:window["end"]
    ]

    if crisis_prices.empty:
        raise ValueError(
            f"{event} is outside the available data range."
        )

    crisis_returns = crisis_prices.pct_change().dropna()

    return crisis_returns


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


def tail_risk_contribution(hist_returns, weights, tail_fraction=0.10):

    asset_growth = (1 + hist_returns).cumprod()
    portfolio_growth = asset_growth @ weights

    cutoff = portfolio_growth.quantile(tail_fraction)
    tail_mask = portfolio_growth <= cutoff

    tail_asset_growth = asset_growth.loc[tail_mask]

    asset_loss_contrib = -(tail_asset_growth - 1).mul(weights, axis=1)
    avg_contrib = asset_loss_contrib.mean()

    return (avg_contrib * 100).sort_values(ascending=False)