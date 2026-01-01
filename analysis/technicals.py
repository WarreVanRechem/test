def indicators(df):
    df['SMA200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df




def regime(price, sma200, rsi):
    if price > sma200 and rsi > 50:
        return "Bull"
    elif price < sma200 and rsi < 50:
        return "Bear"
    return "Range"