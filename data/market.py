import yfinance as yf
import pandas as pd
import numpy as np




def get_price_history(ticker, period="5y"):
    df = yf.Ticker(ticker).history(period=period)
    return df if not df.empty else None




def get_returns(df):
    return df['Close'].pct_change().dropna()




def get_macro():
    tickers = {"S&P500": "^GSPC", "Nasdaq": "^IXIC", "Gold": "GC=F", "Oil": "CL=F", "10Y": "^TNX"}
    out = {}
    for k, t in tickers.items():
        h = yf.Ticker(t).history(period="2d")
        if len(h) >= 2:
            out[k] = ((h['Close'].iloc[-1]-h['Close'].iloc[-2]) / h['Close'].iloc[-2]) * 100
    return out