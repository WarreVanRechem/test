import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import pipeline
import feedparser
import warnings
import requests
from datetime import datetime

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal", layout="wide", page_icon="📈")
warnings.filterwarnings("ignore")

@st.cache_resource
def load_ai():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:
        st.sidebar.error(f"AI Model Error: {e}")
        return None

ai_pipe = load_ai()

# --- DATA FETCHING (Gecacht) ---
@st.cache_data(ttl=3600)
def get_analysis_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        market = yf.Ticker("^GSPC").history(period="7y")
        if df.empty: return None, None
        
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['Returns'] = df['Close'].pct_change()
        
        metrics = {
            "price": df['Close'].iloc[-1],
            "sma200": df['SMA200'].iloc[-1],
            "var": np.percentile(df['Returns'].dropna(), 5),
            "market_bull": market['Close'].iloc[-1] > market['Close'].rolling(200).mean().iloc[-1]
        }
        return df, metrics
    except:
        return None, None

def get_external_info(ticker):
    buys, sells = 0, 0
    news_results = []
    
    # 1. Insider Data
    try:
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is not None and not insider.empty:
            recent = insider.head(10)
            buys = recent[recent['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
            sells = recent[recent['Text'].str.contains("Sale", case=False, na=False)].shape[0]
    except: pass

    # 2. Nieuws met Browser-vermomming
    try:
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+finance&hl=en-US&gl=US&ceid=US:en"
        # We gebruiken een User-Agent om te voorkomen dat Google News ons blokkeert
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(rss_url, headers=headers, timeout=10)
        feed = feedparser.parse(response.content)
        
        if not feed.entries:
            st.warning("Geen nieuws gevonden op Google News voor deze ticker.")
        
        for entry in feed.entries[:5]:
            sentiment_label = "NEUTRAL"
            if ai_pipe:
                try:
                    res = ai_pipe(entry.title[:512])[0] # Beperk lengte voor AI
                    sentiment_label = res['label'].upper()
                except: pass
            news_results.append({"title": entry.title, "sentiment": sentiment_label, "link": entry.link})
    except Exception as e:
        st.error(f"Nieuwsfout: {e}")
        
    return buys, sells, news_results

# --- INTERFACE ---
st.title("💎 Zenith Institutional Terminal")
ticker_input = st.sidebar.text_input("Ticker Symbool", "RDW").upper()
capital = st.sidebar.number_input("Inzet Kapitaal ($)", value=10000)

if st.sidebar.button("Start Deep Analysis"):
    df, metrics = get_analysis_data(ticker_input)
    
    if df is not None:
        with st.spinner('Nieuws en insiders ophalen...'):
            buys, sells, news = get_external_info(ticker_input)
        
        # Scoring
        score = 0
        pros, cons = [], []
        if metrics['market_bull']: score += 20; pros.append("Markt is Bullish")
        if metrics['price'] > metrics['sma200']: score += 30; pros.append("Trend is Positief (> 200MA)")
        else: cons.append("Trend is Negatief (< 200MA)")
        if buys > sells: score += 20; pros.append(f"Insiders kopen ({buys} tx)")
        
        pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE')
        if pos_news >= 1: score += 10; pros.append(f"{pos_news} Positieve nieuwsberichten gevonden")

        # UI
        c1, c2, c3 = st.columns(3)
        c1.metric("Zenith Score", f"{score}/100")
        c2.metric("Huidige Prijs", f"${metrics['price']:.2f}")
        c3.metric("Nieuws Sentiment", f"{pos_news} Positief")

        # Grafiek (5 jaar)
        end_date = df.index[-1]
        start_date = end_date - pd.DateOffset(years=5)
        plot_df = df.loc[start_date:end_date]
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Prijs"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700', width=2), name="200 MA"))
        fig.update_layout(template="plotly_dark", height=500, xaxis=dict(range=[start_date, end_date], rangeslider=dict(visible=False)))
        st.plotly_chart(fig, use_container_width=True)

        # Nieuws Lijst
        st.subheader("📰 Recent AI-Geanalyseerd Nieuws")
        if not news:
            st.info("Er kon geen recent nieuws worden geladen. Probeer het over een paar minuten opnieuw.")
        for n in news:
            color = "green" if n['sentiment'] == 'POSITIVE' else "red" if n['sentiment'] == 'NEGATIVE' else "gray"
            st.markdown(f":{color}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")

        # Pros/Cons
        p_col, c_col = st.columns(2)
        with p_col:
            st.success("### Sterke Punten")
            for p in pros: st.write(f"✅ {p}")
        with c_col:
            st.error("### Risico Factoren")
            for c in cons: st.write(f"❌ {c}")
    else:
        st.error("Geen data gevonden voor deze ticker.")
