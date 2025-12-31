import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import pipeline
import warnings
from datetime import datetime, timedelta

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

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data(ticker):
    stock = yf.Ticker(ticker)
    # We halen 7 jaar op om genoeg data te hebben voor een zuivere 200MA over 5 jaar
    df = stock.history(period="7y")
    market = yf.Ticker("^GSPC").history(period="7y")
    
    if df.empty: return None, None, None
    
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['Returns'] = df['Close'].pct_change()
    
    # Metrics berekenen
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

# --- INTERFACE ---
st.title("💎 Zenith Institutional Terminal")
st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Ticker Symbool", "RDW").upper()
capital = st.sidebar.number_input("Inzet Kapitaal ($)", value=10000)

if st.sidebar.button("Start Deep Analysis"):
    df, metrics, stock = get_data(ticker)
    
    if df is not None:
        # Score & Tekst (Pros/Cons)
        score = 0
        pros, cons = [], []
        if metrics['market_bull']: score += 20; pros.append("Brede markt is Bullish")
        else: cons.append("Markt-omgeving is riskant")
        if metrics['price'] > metrics['sma200']: score += 30; pros.append("Trend is Positief (Boven 200MA)")
        else: cons.append("Trend is Negatief (Onder 200MA)")
        if metrics['sortino'] > 1: score += 20; pros.append("Goede Risk/Reward verhouding")

        # Header stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Zenith Score", f"{score}/100")
        col2.metric("Huidige Prijs", f"${metrics['price']:.2f}")
        col3.metric("Max Dagverlies (VaR)", f"${abs(metrics['var'] * capital):.0f}")

        # --- GRAFIEK FIX VOOR 5 JAAR ---
        # We pakken de data van de laatste 5 jaar
        end_date = df.index[-1]
        start_date = end_date - pd.DateOffset(years=5)
        plot_df = df.loc[start_date:end_date]

        fig = go.Figure()
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open'],
            high=plot_df['High'],
            low=plot_df['Low'],
            close=plot_df['Close'],
            name="Prijs"
        ))
        # 200 MA (Gele lijn)
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df['SMA200'],
            line=dict(color='#FFD700', width=2),
            name="200 MA"
        ))

        # FIX: Dwing de X-as om exact tussen start_date en end_date te blijven
        fig.update_layout(
            title=f"Gedetailleerde 5-jaars Analyse: {ticker}",
            template="plotly_dark",
            height=600,
            xaxis=dict(
                type='date',
                range=[start_date, end_date], # Forceer het bereik
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(autorange=True, fixedrange=False),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Onderste sectie
        c1, c2 = st.columns(2)
        with c1:
            st.success("### Sterke Punten")
            for p in pros: st.write(f"✅ {p}")
        with c2:
            st.error("### Risico Factoren")
            for c in cons: st.write(f"❌ {c}")
    else:
        st.error("Geen data gevonden. Controleer de ticker.")
