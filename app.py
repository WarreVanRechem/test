import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import pipeline
import warnings
from datetime import datetime

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

# --- DATA FETCHING (Gecacht: geeft alleen simpele types terug) ---
@st.cache_data(ttl=3600)
def get_analysis_data(ticker):
    # We gebruiken een lokale variabele voor de ticker, we slaan hem niet op
    stock = yf.Ticker(ticker)
    
    # We halen 7 jaar op om een zuivere 5-jaars grafiek inclusief 200MA te tonen
    df = stock.history(period="7y")
    market = yf.Ticker("^GSPC").history(period="7y")
    
    if df.empty:
        return None, None
    
    # Berekeningen
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
    
    # We geven GEEN 'stock' object terug, dit voorkomt de Unserializable error
    return df, metrics

def get_insider_info(ticker):
    """Haalt insider transacties op zonder caching."""
    try:
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is None or insider.empty: 
            return 0, 0
        recent = insider.head(20)
        buys = recent[recent['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
        sells = recent[recent['Text'].str.contains("Sale", case=False, na=False)].shape[0]
        return buys, sells
    except:
        return 0, 0

# --- INTERFACE ---
st.title("💎 Zenith Institutional Terminal")
st.sidebar.header("Parameters")
ticker_input = st.sidebar.text_input("Ticker Symbool", "RDW").upper()
capital = st.sidebar.number_input("Inzet Kapitaal ($)", value=10000)

if st.sidebar.button("Start Deep Analysis"):
    # We roepen de gecachte data aan
    df, metrics = get_analysis_data(ticker_input)
    
    if df is not None:
        # We roepen de niet-gecachte insider info aan
        buys, sells = get_insider_info(ticker_input)
        
        # Score berekening
        score = 0
        pros, cons = [], []
        if metrics['market_bull']: 
            score += 20
            pros.append("Brede markt is Bullish")
        else: 
            cons.append("Markt-omgeving is riskant")
            
        if metrics['price'] > metrics['sma200']: 
            score += 30
            pros.append("Trend is Positief (Boven 200MA)")
        else: 
            cons.append("Trend is Negatief (Onder 200MA)")
            
        if buys > sells: 
            score += 20
            pros.append(f"Insiders kopen bij ({buys} transacties)")

        # Resultaten balk
        col1, col2, col3 = st.columns(3)
        col1.metric("Zenith Score", f"{score}/100")
        col2.metric("Huidige Prijs", f"${metrics['price']:.2f}")
        col3.metric("Potentieel Dagverlies (VaR)", f"${abs(metrics['var'] * capital):.0f}")

        # --- GRAFIEK: EXACT 5 JAAR ---
        # Bepaal het bereik voor de X-as
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

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis=dict(
                type='date',
                range=[start_date, end_date], # Forceer het bereik op 5 jaar
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(autorange=True, fixedrange=False),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pros & Cons sectie
        c1, c2 = st.columns(2)
        with c1:
            st.success("### Sterke Punten")
            for p in pros: st.write(f"✅ {p}")
        with c2:
            st.error("### Risico Factoren")
            for c in cons: st.write(f"❌ {c}")
    else:
        st.error("Kon geen data vinden voor deze ticker. Controleer het symbool.")
