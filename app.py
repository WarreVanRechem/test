import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import pipeline
import feedparser
import warnings
import time
import random

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

# --- DATA FETCHING MET CACHE ---
@st.cache_data(ttl=3600)
def get_zenith_data(ticker):
    """Haalt data op met extra historie voor de 200MA."""
    try:
        # We halen 3 jaar op om ZEKER te weten dat we 200 dagen hebben voor de lijn
        stock = yf.Ticker(ticker)
        df = stock.history(period="3y") 
        market = yf.Ticker("^GSPC").history(period="3y")
        
        if df.empty:
            return None, None, None
            
        # BEREKENING GELE LIJN (SMA200)
        # We gebruiken 'min_periods=1' zodat hij ook lijnen tekent als er minder data is,
        # maar voor de echte 200MA hebben we de 3 jaar historie hierboven nodig.
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        return df, market, stock
    except:
        return None, None, None

# --- INTERFACE ---
st.title("💎 Zenith Institutional Terminal")
ticker = st.sidebar.text_input("Ticker Symbool", "RDW").upper()
run_btn = st.sidebar.button("Update Dashboard")

if run_btn:
    df, market, stock = get_zenith_data(ticker)
    
    if df is not None:
        # Laatste metrics
        current_price = df['Close'].iloc[-1]
        sma200_val = df['SMA200'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Huidige Prijs", f"${current_price:.2f}")
        
        # Laat zien of we boven of onder de gele lijn zitten
        diff = current_price - sma200_val
        status = "BOVEN 200MA" if diff > 0 else "ONDER 200MA"
        col2.metric("Trend Status", status, f"{diff:.2f}")
        
        col3.metric("200 MA Waarde", f"${sma200_val:.2f}")

        # GRAFIEK
        st.subheader(f"Technische Analyse: {ticker}")
        
        # We tonen alleen het laatste jaar in de grafiek voor de duidelijkheid, 
        # maar de SMA200 is berekend op de volledige 3 jaar data.
        plot_df = df.tail(252) 
        
        fig = go.Figure()
        
        # Kaarsen
        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open'],
            high=plot_df['High'],
            low=plot_df['Low'],
            close=plot_df['Close'],
            name="Koers"
        ))
        
        # DE GELE LIJN (SMA200)
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df['SMA200'],
            line=dict(color='#FFD700', width=3), # Fel geel/goud
            name="200-daags Gemiddelde (Trend)"
        ))

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Extra: Insider info kort tonen
        try:
            insider = stock.insider_transactions
            if insider is not None and not insider.empty:
                st.subheader("Bazen (Insiders)")
                st.write(insider.head(5))
        except:
            pass
    else:
        st.warning("Kon geen data vinden. Probeer een andere ticker of ververs de pagina.")
