import numpy as np
import pandas as pd

def calculate_atr_stop(df, multiplier=2):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(14).mean().iloc[-1] * multiplier
