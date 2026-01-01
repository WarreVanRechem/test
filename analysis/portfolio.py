from scipy.optimize import minimize




def portfolio_volatility(weights, cov):
    return np.sqrt(weights.T @ cov @ weights)




def optimize_portfolio(returns):
    cov = returns.cov() * 252
    n = len(cov)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    bounds = [(0,1)] * n
    res = minimize(portfolio_volatility, n*[1/n], args=(cov,), bounds=bounds, constraints=cons)
    return res.x