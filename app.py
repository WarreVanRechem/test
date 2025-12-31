import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
import feedparser
from scipy.stats import norm
import warnings

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal", layout="wide", page_icon="📈")
warnings.filterwarnings("ignore")

@st.cache_resource
def load_ai():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

ai_pipe = load_ai()

# --- ANALYSE FUNCTIES ---
def get_data(ticker):
    stock = yf.Ticker(ticker)
    # VERANDERING 1: We halen nu 7 jaar data op om zeker 5 jaar schone data over te houden na berekeningen
    df = stock.history(period="7y")
    market = yf.Ticker("^GSPC").history(period="7y")
    
    if df.empty: return None, None, None
    
    # Voortschrijdend gemiddelde berekenen
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['Returns'] = df['Close'].pct_change()
    
    # Risico Metrics
    downside = df.loc[df['Returns'] < 0, 'Returns']
    sortino = (df['Returns'].mean() * 252) / (downside.std() * np.sqrt(252)) if not downside.empty else 0
    var_95 = np.percentile(df['Returns'].dropna(), 5)
    
    metrics = {
        "price": df['Close'].iloc[-1],
        "sma200": df['SMA200'].iloc[-1],
        "sortino": sortino,
        "var": var_95,
        "market_bull": market['Close'].iloc[-1] > market['Close'].rolling(200).mean().iloc[-1]
    }
    return df, metrics, stock

def get_insiders(stock):
    try:
        insider = stock.insider_transactions
        if insider is None or insider.empty: return 0, 0
        recent = insider.head(20)
        buys = recent[recent['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
        sells = recent[recent['Text'].str.contains("Sale", case=False, na=False)].shape[0]
        return buys, sells
    except: return 0, 0

# --- INTERFACE ---
st.title("💎 Zenith Institutional Terminal")
st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Ticker Symbool", "RDW").upper()
capital = st.sidebar.number_input("Inzet Kapitaal ($)", value=10000)

if st.sidebar.button("Start Deep Analysis"):
    df, metrics, stock = get_data(ticker)
    
    if df is not None:
        buys, sells = get_insiders(stock)
        
        # Scoring Logic
        score = 0
        pros, cons = [], []
        
        if metrics['market_bull']: score += 20; pros.append("Brede markt is Bullish")
        else: cons.append("Markt-omgeving is riskant")
            
        if metrics['price'] > metrics['sma200']: score += 30; pros.append("Trend is Positief (Boven 200MA)")
        else: cons.append("Trend is Negatief (Onder 200MA)")
            
        if buys > sells: score += 20; pros.append(f"Insiders kopen ({buys} aankopen)")
        if metrics['sortino'] > 1: score += 20; pros.append("Goede Risk/Reward verhouding")

        # Resultaten
        col1, col2, col3 = st.columns(3)
        col1.metric("Zenith Score", f"{score}/100")
        col2.metric("Huidige Prijs", f"${metrics['price']:.2f}")
        col3.metric("Max Dagverlies (VaR)", f"${abs(metrics['var'] * capital):.0f}")

        # VERANDERING 2: We selecteren de laatste 5 jaar aan handelsdagen (ongeveer 1260 dagen)
        plot_df = df.tail(1260)

        # Grafiek
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=plot_df.index, 
            open=plot_df['Open'], 
            high=plot_df['High'], 
            low=plot_df['Low'], 
            close=plot_df['Close'], 
            name="Prijs"
        ))
        fig.add_trace(go.Scatter(
            x=plot_df.index, 
            y=plot_df['SMA200'], 
            line=dict(color='orange', width=2), 
            name="200 MA"
        ))
        
        fig.update_layout(
            title=f"Koersverloop 5 Jaar: {ticker}",
            template="plotly_dark", 
            height=600,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pros & Cons
        c1, c2 = st.columns(2)
        with c1:
            st.success("### Sterke Punten")
            for p in pros: st.write(f"✅ {p}")
        with c2:
            st.error("### Risico Factoren")
            for c in cons: st.write(f"❌ {c}")
    else:
        st.error("Ticker niet gevonden of Yahoo limiet bereikt.")
