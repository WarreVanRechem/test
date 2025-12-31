import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
import feedparser
import warnings
import requests
from datetime import datetime

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal v6.0", layout="wide", page_icon="📈")
warnings.filterwarnings("ignore")

@st.cache_resource
def load_ai():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

ai_pipe = load_ai()

# --- DATA & RSI BEREKENING ---
@st.cache_data(ttl=3600)
def get_zenith_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        market = yf.Ticker("^GSPC").history(period="7y")
        if df.empty: return None, None
        
        # 200MA
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI Berekening (14 dagen)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Returns'] = df['Close'].pct_change()
        metrics = {
            "price": df['Close'].iloc[-1],
            "sma200": df['SMA200'].iloc[-1],
            "rsi": df['RSI'].iloc[-1],
            "market_bull": market['Close'].iloc[-1] > market['Close'].rolling(200).mean().iloc[-1]
        }
        return df, metrics
    except:
        return None, None

def get_external_info(ticker):
    buys, news_results = 0, []
    try:
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is not None and not insider.empty:
            buys = insider.head(10)[insider.head(10)['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
        
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+finance&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(rss_url, headers=headers, timeout=10)
        feed = feedparser.parse(response.content)
        
        for entry in feed.entries[:5]:
            sentiment = "NEUTRAL"
            if ai_pipe:
                try:
                    res = ai_pipe(entry.title[:512])[0]
                    sentiment = res['label'].upper()
                except: pass
            news_results.append({"title": entry.title, "sentiment": sentiment, "link": entry.link})
    except: pass
    return buys, news_results

# --- INTERFACE ---
st.title("💎 Zenith Institutional Terminal")
ticker_input = st.sidebar.text_input("Ticker Symbool", "RDW").upper()
capital = st.sidebar.number_input("Inzet Kapitaal ($)", value=10000)

if st.sidebar.button("Start Deep Analysis"):
    df, metrics = get_zenith_data(ticker_input)
    
    if df is not None:
        with st.spinner('Nieuws en RSI berekenen...'):
            buys, news = get_external_info(ticker_input)
        
        # Scoring
        score = 0
        if metrics['market_bull']: score += 20
        if metrics['price'] > metrics['sma200']: score += 30
        if buys > 0: score += 20
        if 30 < metrics['rsi'] < 70: score += 10 # Bonus voor gezonde RSI
        if metrics['rsi'] < 30: score += 20 # Extra bonus voor 'koopje' (oversold)

        # Header metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Zenith Score", f"{score}/100")
        c2.metric("Huidige Prijs", f"${metrics['price']:.2f}")
        c3.metric("RSI (14D)", f"{metrics['rsi']:.1f}")
        c4.metric("Markt Status", "🟢 Veilig" if metrics['market_bull'] else "🔴 Risico")

        # --- GEAVANCEERDE GRAFIEK MET RSI ---
        end_date = df.index[-1]
        start_date = end_date - pd.DateOffset(years=5)
        plot_df = df.loc[start_date:end_date]

        # Maak subplots: 1 voor koers (80% hoogte), 1 voor RSI (20% hoogte)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # 1. Candlestick & 200MA
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                                     low=plot_df['Low'], close=plot_df['Close'], name="Prijs"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700', width=2), name="200 MA"), row=1, col=1)

        # 2. RSI Lijn
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#9370DB', width=2), name="RSI"), row=2, col=1)

        # RSI Overbought/Oversold Lijnen
        fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought (70)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)", row=2, col=1)

        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Nieuws & Analyse
        st.subheader("📰 AI Nieuws & Thesis")
        n_col, t_col = st.columns([1, 1])
        with n_col:
            for n in news:
                color = "green" if n['sentiment'] == 'POSITIVE' else "red" if n['sentiment'] == 'NEGATIVE' else "white"
                st.markdown(f":{color}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")
        with t_col:
            st.info(f"**RSI Analyse:** Het aandeel heeft een RSI van {metrics['rsi']:.1f}. " + 
                    ("Het aandeel is momenteel 'Oversold' (Goedkoop)." if metrics['rsi'] < 30 else 
                     "Het aandeel is momenteel 'Overbought' (Duur)." if metrics['rsi'] > 70 else 
                     "De prijs bevindt zich in een neutrale zone."))

    else:
        st.error("Geen data gevonden.")
