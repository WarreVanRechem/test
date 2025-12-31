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
st.set_page_config(page_title="Zenith door Warre V.R.", layout="wide", page_icon="💎")
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
def get_zenith_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        market = yf.Ticker("^GSPC").history(period="7y")
        if df.empty: return None, None
        
        # Indicator Berekeningen
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI (14)
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
            "market_bull": market['Close'].iloc[-1] > market['Close'].rolling(200).mean().iloc[-1],
            "var": np.percentile(df['Returns'].dropna(), 5)
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
        
        # Nieuws
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+finance&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(rss_url, headers=headers, timeout=5)
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

# --- SIDEBAR MET OPTIES ---
st.sidebar.header("Instellingen")

# 1. VALUTA KEUZE (NIEUW)
currency_mode = st.sidebar.radio("Valuta Weergave", ["USD ($)", "EUR (€)"])
curr_symbol = "$" if "USD" in currency_mode else "€"

ticker_input = st.sidebar.text_input("Ticker Symbool", "RDW").upper()
# Kapitaal past zich nu aan aan het gekozen symbool
capital = st.sidebar.number_input(f"Inzet Kapitaal ({curr_symbol})", value=10000)

run_btn = st.sidebar.button("Start Deep Analysis")

st.sidebar.markdown("---")
st.sidebar.markdown("### Credits")
st.sidebar.markdown("Created by **Warre Van Rechem**")
st.sidebar.markdown("[Connect on LinkedIn](https://www.linkedin.com/in/warre-van-rechem-928723298/)")
# -----------------------------

if run_btn:
    df, metrics = get_zenith_data(ticker_input)
    
    if df is not None:
        with st.spinner('Analyseren van data, insiders en nieuws...'):
            buys, news = get_external_info(ticker_input)
        
        # --- SCORING ---
        score = 0
        pros, cons = [], []
        
        if metrics['market_bull']: score += 20; pros.append("Markt (S&P500) is Bullish")
        else: cons.append("Markt is onzeker")
            
        if metrics['price'] > metrics['sma200']: score += 30; pros.append("Trend is Positief (> 200MA)")
        else: cons.append("Trend is Negatief (< 200MA)")
            
        if metrics['rsi'] < 30: score += 20; pros.append(f"RSI is Oversold ({metrics['rsi']:.1f}) - Koopkans?")
        elif metrics['rsi'] > 70: cons.append(f"RSI is Overbought ({metrics['rsi']:.1f}) - Te duur?")
        
        if buys > 0: score += 20; pros.append(f"Insiders kopen ({buys}x)")
        
        pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE')
        if pos_news >= 2: score += 10; pros.append("Positief nieuws sentiment")

        # --- METRICS MET JUISTE VALUTA ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Zenith Score", f"{score}/100")
        # Hier gebruiken we nu curr_symbol in plaats van hardcoded $
        c2.metric("Prijs", f"{curr_symbol}{metrics['price']:.2f}")
        c3.metric("RSI", f"{metrics['rsi']:.1f}")
        c4.metric("Risk (VaR)", f"{curr_symbol}{abs(metrics['var'] * capital):.0f}")

        # --- GRAFIEK ---
        end_date = df.index[-1]
        start_date = end_date - pd.DateOffset(years=5)
        plot_df = df.loc[start_date:end_date]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                                     low=plot_df['Low'], close=plot_df['Close'], name="Prijs"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700', width=2), name="200 MA"), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#9370DB', width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- PROS & CONS ---
        st.subheader("⚖️ De Analyse")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.success("### ✅ Sterke Punten")
            if not pros: st.write("Geen sterke punten.")
            for p in pros: st.write(f"• {p}")
        with col_cons:
            st.error("### ❌ Risico Punten")
            if not cons: st.write("Geen grote risico's.")
            for c in cons: st.write(f"• {c}")

        # --- NIEUWS ---
        st.subheader("📰 Laatste Nieuws")
        for n in news:
            color = "green" if n['sentiment'] == 'POSITIVE' else "red" if n['sentiment'] == 'NEGATIVE' else "gray"
            st.markdown(f":{color}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")
            
    else:
        st.error("Geen data. Probeer over 1 minuut opnieuw.")

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2025 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")
