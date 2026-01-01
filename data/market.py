import yfinance as yf
import streamlit as st
from analysis.technicals import calculate_atr_stop
from analysis.fundamentals import calculate_graham_number

@st.cache_data(ttl=3600)
def get_zenith_data(ticker):
    s = yf.Ticker(ticker)
    df = s.history(period="7y")
    info = s.info
    if df.empty:
        return None, None, None, None, None, None, None, "Geen data", None

    cur = df['Close'].iloc[-1]
    fair = calculate_graham_number(info)

    df['SMA200'] = df['Close'].rolling(200).mean()
    df['RSI'] = 100 - (100 / (1 + (
        df['Close'].diff().clip(lower=0).rolling(14).mean() /
        -df['Close'].diff().clip(upper=0).rolling(14).mean()
    )))

    atr = calculate_atr_stop(df)
    high = df['High'].tail(50).max()

    sniper = {
        "entry_price": df['Close'].rolling(20).mean().iloc[-1],
        "stop_loss": cur - atr,
        "take_profit": high,
        "rr_ratio": (high-cur)/atr if atr else 0
    }

    metrics = {
        "name": info.get("longName", ticker),
        "price": cur,
        "rsi": df['RSI'].iloc[-1]
    }

    fund = {"fair_value": fair}
    ws = {"upside": ((info.get('targetMeanPrice',0)-cur)/cur)*100 if info.get('targetMeanPrice') else 0}

    return df, metrics, fund, ws, None, sniper, None, None, []
