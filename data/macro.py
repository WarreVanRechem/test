import yfinance as yf
import streamlit as st

@st.cache_data(ttl=600)
def get_macro_data():
    tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Goud": "GC=F",
        "Olie": "CL=F",
        "10Y Rente": "^TNX"
    }
    data = {}
    for name, t in tickers.items():
        try:
            h = yf.Ticker(t).history(period="2d")
            p, prev = h['Close'].iloc[-1], h['Close'].iloc[-2]
            data[name] = (p, ((p-prev)/prev)*100)
        except:
            data[name] = (0,0)
    return data
