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

# --- PAGE CONFIG ---
st.set_page_config(page_title="ZenithTrader Pro", layout="wide", page_icon="💎")
warnings.filterwarnings("ignore")

# --- CACHING (Zodat AI niet elke keer opnieuw laadt) ---
@st.cache_resource
def load_ai():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

ai_pipe = load_ai()

# --- DE FUNCTIES (Hetzelfde als voorheen, maar aangepast voor Streamlit) ---
def get_analysis(ticker):
    stock = yf.Ticker(ticker)
    
    # Data ophalen
    try:
        df = stock.history(period="5y")
        market = yf.Ticker("^GSPC").history(period="2y")
    except:
        return None, None, None, "Fout bij ophalen data"

    if df.empty: return None, None, None, "Geen data gevonden voor ticker"

    # Berekeningen
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Returns'] = df['Close'].pct_change()
    
    # Z-Score
    mean = df['Close'].rolling(252).mean()
    std = df['Close'].rolling(252).std()
    z_score = (df['Close'].iloc[-1] - mean.iloc[-1]) / std.iloc[-1]
    
    # Sortino
    downside = df.loc[df['Returns'] < 0, 'Returns']
    sortino = (df['Returns'].mean() * 252) / (downside.std() * np.sqrt(252)) if downside.std() > 0 else 0

    metrics = {
        "current": df['Close'].iloc[-1],
        "sma200": df['SMA200'].iloc[-1],
        "z_score": z_score,
        "sortino": sortino,
        "market_trend": market['Close'].iloc[-1] > market['Close'].rolling(200).mean().iloc[-1]
    }
    return df, metrics, stock, None

def get_insider(stock):
    try:
        insider = stock.insider_transactions
        if insider is None or insider.empty: return 0, "Geen data"
        recent = insider.head(20)
        buys = recent[recent['Text'].astype(str).str.contains("Purchase", case=False, na=False)].shape[0]
        sells = recent[recent['Text'].astype(str).str.contains("Sale", case=False, na=False)].shape[0]
        return buys, sells
    except:
        return 0, 0

def get_sentiment(ticker):
    if not ai_pipe: return 50
    rss = f"https://news.google.com/rss/search?q={ticker}+finance&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss)
    scores = []
    for entry in feed.entries[:5]:
        res = ai_pipe(entry.title)[0]
        score = 1 if res['label'] == 'positive' else -1 if res['label'] == 'negative' else 0
        scores.append(score * res['score'])
    avg = np.mean(scores) if scores else 0
    return (avg + 1) * 50

# --- DE GUI (GEBRUIKERSINTERFACE) ---
st.title("💎 ZenithTrader: Institutional Analysis")
st.markdown("Vul een ticker in en laat de AI het werk doen.")

# 1. Sidebar voor Invoer
with st.sidebar:
    st.header("Instellingen")
    ticker_input = st.text_input("Aandeel Ticker (bijv. NVDA, RDW)", "RDW").upper()
    run_btn = st.button("🚀 Start Analyse", type="primary")

# 2. Hoofd Scherm
if run_btn:
    with st.spinner(f'Analyseren van {ticker_input}... (AI aan het lezen)'):
        df, metrics, stock, error = get_analysis(ticker_input)
        
        if error:
            st.error(error)
        else:
            # Insiders & Sentiment ophalen
            buys, sells = get_insider(stock)
            sent_score = get_sentiment(ticker_input)

            # Score Berekening
            score = 0
            if metrics['market_trend']: score += 20
            if metrics['sortino'] > 1.5: score += 20
            if metrics['z_score'] < -1.5: score += 20
            if metrics['current'] > metrics['sma200']: score += 20
            if buys > sells: score += 10
            if sent_score > 60: score += 20
            
            score = min(100, max(0, score))
            verdict = "STRONG BUY" if score >= 80 else "BUY" if score >= 60 else "HOLD" if score >= 40 else "SELL"

            # --- RESULTATEN TONEN ---
            
            # KPI Kaarten
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Zenith Score", f"{score}/100", verdict)
            col2.metric("Huidige Koers", f"${metrics['current']:.2f}")
            col3.metric("Insiders (20 tx)", f"{buys} Gekocht", f"-{sells} Verkocht")
            col4.metric("AI Sentiment", f"{sent_score:.0f}/100")

            # Grafiek
            st.subheader("Technische Trend")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index[-252:], open=df['Open'][-252:], high=df['High'][-252:],
                                         low=df['Low'][-252:], close=df['Close'][-252:], name='Koers'))
            fig.add_trace(go.Scatter(x=df.index[-252:], y=df['SMA200'][-252:], line=dict(color='orange'), name='200 MA'))
            fig.update_layout(height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # Analyse Tekst
            st.subheader("📝 Investment Thesis")
            c1, c2 = st.columns(2)
            
            with c1:
                st.success("✅ WAAROM KOPEN (PROS)")
                if metrics['market_trend']: st.write("• Markt (S&P500) is veilig")
                if metrics['sortino'] > 1.5: st.write(f"• Goede Risk/Reward ({metrics['sortino']:.1f})")
                if buys > sells: st.write(f"• Insiders kopen ({buys} vs {sells})")
                if metrics['z_score'] < -1.5: st.write("• Statistisch goedkoop")
                
            with c2:
                st.error("❌ WAAROM OPPASSEN (CONS)")
                if metrics['current'] < metrics['sma200']: st.write("• Trend is dalend (Onder 200MA)")
                if sells > buys + 5: st.write("• Insiders verkopen veel")
                if metrics['z_score'] > 1.5: st.write("• Statistisch te duur")

            # Disclaimer
            st.caption("Disclaimer: Dit is een AI-gegenereerde analyse en geen financieel advies.")
