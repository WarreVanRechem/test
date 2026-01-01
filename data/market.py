import yfinance as yf
import pandas as pd
import numpy as np


def get_price_history(ticker, period="5y"):
    df = yf.Ticker(ticker).history(period=period)
    return df if not df.empty else None


def get_returns(df):
    return df["Close"].pct_change().dropna()
