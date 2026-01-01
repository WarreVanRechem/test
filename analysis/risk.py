def max_drawdown(df):
    return (df['Close'] / df['Close'].cummax() - 1).min()




def value_at_risk(returns, capital=10000, level=5):
    return np.percentile(returns, level) * capital




def atr_stop(df, mult=2):
    tr = pd.concat([
        df['High']-df['Low'],
        (df['High']-df['Close'].shift()).abs(),
        (df['Low']-df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    return atr * mult