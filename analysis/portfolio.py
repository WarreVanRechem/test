import numpy as np
from scipy.optimize import minimize


def portfolio_volatility(weights, cov):
    return np.sqrt(weights.T @ cov @ weights)


def optimize_portfolio(returns):
    cov = returns.cov() * 252
    n = len(cov)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n

    result = minimize(
        portfolio_volatility,
        x0=np.ones(n) / n,
        args=(cov,),
        bounds=bounds,
        constraints=constraints,
    )

    return result.x
