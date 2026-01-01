# ==================================================
# FILE: analysis/scoring.py
# ==================================================

def normalize(value, min_val, max_val):
    if value is None:
        return 0
    return max(0, min(100, 100 * (value - min_val) / (max_val - min_val)))


def investment_score(price, fair_value, rsi, drawdown):
    valuation_score = normalize((fair_value - price) / price if fair_value else 0, -0.5, 0.5)
    momentum_score = normalize(rsi, 30, 70)
    risk_score = normalize(-drawdown, 0, 0.6)

    score = (
        0.30 * momentum_score +
        0.30 * valuation_score +
        0.40 * risk_score
    )

    return round(score, 1)

