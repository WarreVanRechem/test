def normalize(value, min_val, max_val):
    if value is None:
        return 0
    return max(0, min(100, 100 * (value - min_val) / (max_val - min_val)))


def investment_score(price, fair_value, rsi, drawdown):
    valuation = normalize(
        (fair_value - price) / price if fair_value else 0, -0.5, 0.5
    )
    momentum = normalize(rsi, 30, 70)
    risk = normalize(-drawdown, 0, 0.6)

    score = (
        0.30 * momentum +
        0.30 * valuation +
        0.40 * risk
    )

    return round(score, 1)
