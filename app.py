import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import pipeline
import feedparser
import warnings
from datetime import datetime

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal", layout="wide", page_icon="📈")
warnings.filterwarnings("ignore")

@st.cache_resource
def load_ai():
    """Laden van het AI-model (alleen de eerste keer)"""
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

ai_pipe = load_ai()

# --- DATA FETCHING (Gecacht voor stabiliteit) ---
@st.cache_data(ttl=3600)
def get_analysis_data(ticker):
    """Haalt koersdata op. Geeft GEEN Ticker-object terug (voorkomt crash)"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        market = yf.Ticker("^GSPC").history(period="7y")
        
        if df.empty:
            return None, None
        
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['Returns'] = df['Close'].pct_change()
        
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
        return df, metrics
    except:
        return None, None

def get_external_info(ticker):
    """Haalt nieuws en insider data op (niet gecacht)"""
    # 1. Insider Data
    buys, sells = 0, 0
    news_results = []
    try:
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is not None and not insider.empty:
            recent = insider.head(20)
            buys = recent[recent['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
            sells = recent[recent['Text'].str.contains("Sale", case=False, na=False)].shape[0]
            
        # 2. Nieuws & AI Sentiment
        rss_url = f"https://news.google.com/rss/search?q={ticker}+finance&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:5]: # Top 5 koppen
            sentiment = "Neutral"
            if ai_pipe:
                res = ai_pipe(entry.title)[0]
                sentiment = res['label']
            news_results.append({"title": entry.title, "sentiment": sentiment, "link": entry.link})
            
    except:
        pass
    return buys, sells, news_results

# --- INTERFACE ---
st.title("💎 Zenith Institutional Terminal")
st.sidebar.header("Parameters")
ticker_input = st.sidebar.text_input("Ticker Symbool", "RDW").upper()
capital = st.sidebar.number_input("Inzet Kapitaal ($)", value=10000)

if st.sidebar.button("Start Deep Analysis"):
    df, metrics = get_analysis_data(ticker_input)
    
    if df is not None:
        buys, sells, news = get_external_info(ticker_input)
        
        # Scoring
        score = 0
        pros, cons = [], []
        if metrics['market_bull']: score += 20; pros.append("Markt is Bullish")
        else: cons.append("Markt-omgeving is riskant")
        if metrics['price'] > metrics['sma200']: score += 30; pros.append("Trend is Positief (> 200MA)")
        else: cons.append("Trend is Negatief (< 200MA)")
        if buys > sells: score += 20; pros.append(f"Insiders kopen ({buys} tx)")
        
        # Sentiment Bonus
        pos_news = sum(1 for n in news if n['sentiment'] == 'positive')
        if pos_news >= 2: score += 10; pros.append("AI Sentiment: Overwegend positief nieuws")

        # UI: Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Zenith Score", f"{score}/100")
        col2.metric("Huidige Prijs", f"${metrics['price']:.2f}")
        col3.metric("Nieuws Sentiment", f"{pos_news} Positief")

        # UI: Grafiek (5 jaar)
        end_date = df.index[-1]
        start_date = end_date - pd.DateOffset(years=5)
        plot_df = df.loc[start_date:end_date]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                                     low=plot_df['Low'], close=plot_df['Close'], name="Prijs"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700', width=2), name="200 MA"))
        fig.update_layout(template="plotly_dark", height=500, xaxis=dict(range=[start_date, end_date], rangeslider=dict(visible=False)))
        st.plotly_chart(fig, use_container_width=True)

        # UI: Nieuws Sectie
        st.subheader("📰 Recent AI-Geanalyseerd Nieuws")
        for n in news:
            icon = "🟢" if n['sentiment'] == 'positive' else "🔴" if n['sentiment'] == 'negative' else "⚪"
            st.markdown(f"{icon} **{n['sentiment'].upper()}**: [{n['title']}]({n['link']})")

        # UI: Pros & Cons
        c1, c2 = st.columns(2)
        with c1:
            st.success("### Sterke Punten")
            for p in pros: st.write(f"✅ {p}")
        with c2:
            st.error("### Risico Factoren")
            for c in cons: st.write(f"❌ {c}")
    else:
        st.error("Geen data gevonden.")
