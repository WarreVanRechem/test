def normalize(val, low, high):
    return max(0, min(100, 100 * (val - low) / (high - low)))




def investment_score(price, fair, rsi, drawdown):
    valuation = normalize((fair-price)/price, -0.5, 0.5)
    momentum = normalize(rsi, 30, 70)
    risk = normalize(-drawdown, 0, 0.6)
    return round(0.3*momentum + 0.3*valuation + 0.4*risk, 1)