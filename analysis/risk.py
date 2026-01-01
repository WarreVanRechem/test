import numpy as np
import pandas as pd


def max_drawdown(df):
    return (df["Close"] / df["Close"].cummax() - 1).min()


def atr_stop(df, multiplier=2):
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(14).mean().iloc[-1]
    return atr * multiplier
