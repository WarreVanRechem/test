import yfinance as yf
import numpy as np

def get_financial_trends(ticker):
    try:
        f = yf.Ticker(ticker).financials.T
        cols = ['Total Revenue', 'Net Income']
        df = f[cols].dropna()
        df.index = df.index.year
        return df.sort_index()
    except:
        return None

def calculate_graham_number(info):
    try:
        eps = info.get('trailingEps')
        bv = info.get('bookValue')
        if eps and bv and eps > 0 and bv > 0:
            return np.sqrt(22.5 * eps * bv)
    except:
        pass
    return None
