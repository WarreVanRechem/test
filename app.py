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
st.set_page_config(page_title="Zenith Terminal v7.0", layout="wide", page_icon="💎")
warnings.filterwarnings("ignore")

# --- INITIALISEER SESSIE VOOR PORTFOLIO ---
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

@st.cache_resource
def load_ai():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

ai_pipe = load_ai()

# --- DATA FUNCTIES ---
@st.cache_data(ttl=3600)
def get_zenith_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        market = yf.Ticker("^GSPC").history(period="7y")
        if df.empty: return None, None
        
        df['SMA200'] = df['Close'].rolling(window=200).mean()
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

def get_current_price(ticker):
    """Haalt supersnel alleen de huidige prijs op voor portfolio."""
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
    except: pass
    return 0.0

# --- SIDEBAR NAVIGATIE ---
st.sidebar.header("Navigatie")
page = st.sidebar.radio("Ga naar:", ["🔎 Markt Analyse", "💼 Mijn Portfolio"])

st.sidebar.markdown("---")
# Valuta Instelling (Geldt voor de hele app)
currency_mode = st.sidebar.radio("Valuta Weergave", ["USD ($)", "EUR (€)"])
curr_symbol = "$" if "USD" in currency_mode else "€"

# Credits
st.sidebar.markdown("---")
st.sidebar.markdown("### Credits")
st.sidebar.markdown("Created by **Warre Van Rechem**")
st.sidebar.markdown("[Connect on LinkedIn](https://www.linkedin.com/in/warre-van-rechem-928723298/)")
st.sidebar.error("⚠️ **DISCLAIMER:** Geen financieel advies. Educatief gebruik.")

# ==========================================
# PAGINA 1: MARKT ANALYSE (DE OUDE TOOL)
# ==========================================
if page == "🔎 Markt Analyse":
    st.title("💎 Zenith Institutional Terminal") 
    st.warning("⚠️ **Wettelijke Disclaimer:** Deze analyse is gebaseerd op AI en historische data. Doe altijd uw eigen onderzoek.")

    col_input, col_cap = st.columns(2)
    with col_input:
        ticker_input = st.text_input("Ticker Symbool", "RDW").upper()
    with col_cap:
        capital = st.number_input(f"Virtueel Kapitaal ({curr_symbol})", value=10000)
    
    if st.button("Start Deep Analysis"):
        df, metrics = get_zenith_data(ticker_input)
        
        if df is not None:
            with st.spinner('Analyseren...'):
                buys, news = get_external_info(ticker_input)
            
            # Scoring
            score = 0
            pros, cons = [], []
            if metrics['market_bull']: score += 20; pros.append("Markt (S&P500) is Bullish")
            else: cons.append("Markt is onzeker")
            if metrics['price'] > metrics['sma200']: score += 30; pros.append("Trend is Positief (> 200MA)")
            else: cons.append("Trend is Negatief (< 200MA)")
            if metrics['rsi'] < 30: score += 20; pros.append(f"RSI is Oversold ({metrics['rsi']:.1f})")
            elif metrics['rsi'] > 70: cons.append(f"RSI is Overbought ({metrics['rsi']:.1f})")
            if buys > 0: score += 20; pros.append(f"Insiders kopen ({buys}x)")
            pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE')
            if pos_news >= 2: score += 10; pros.append("Positief nieuws sentiment")

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Zenith Score", f"{score}/100")
            c2.metric("Prijs", f"{curr_symbol}{metrics['price']:.2f}")
            c3.metric("RSI", f"{metrics['rsi']:.1f}")
            c4.metric("Risk (VaR)", f"{curr_symbol}{abs(metrics['var'] * capital):.0f}")

            # Grafiek
            end_date = df.index[-1]
            start_date = end_date - pd.DateOffset(years=5)
            plot_df = df.loc[start_date:end_date]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Prijs"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700', width=2), name="200 MA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#9370DB', width=2), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Pros/Cons & Nieuws
            c_pros, c_cons = st.columns(2)
            with c_pros:
                st.success("### ✅ Sterke Punten")
                for p in pros: st.write(f"• {p}")
            with c_cons:
                st.error("### ❌ Risico Punten")
                for c in cons: st.write(f"• {c}")
                
            st.subheader("📰 Laatste Nieuws")
            for n in news:
                color = "green" if n['sentiment'] == 'POSITIVE' else "red" if n['sentiment'] == 'NEGATIVE' else "gray"
                st.markdown(f":{color}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")
        else:
            st.error("Geen data gevonden.")

# ==========================================
# PAGINA 2: MIJN PORTFOLIO (NIEUW)
# ==========================================
elif page == "💼 Mijn Portfolio":
    st.title("💼 Mijn Portfolio Manager")
    st.info("Voeg hier je aandelen toe om je totale waarde en winst/verlies te volgen.")

    # INPUT SECTIE
    with st.expander("➕ Aandeel Toevoegen", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1: p_ticker = st.text_input("Ticker (bv. AAPL)", key="p_ticker").upper()
        with c2: p_amount = st.number_input("Aantal", min_value=0.01, step=1.0, key="p_amount")
        with c3: p_avg = st.number_input(f"Gem. Aankoopprijs ({curr_symbol})", min_value=0.0, step=0.1, key="p_avg")
        with c4:
            st.write("") # Spacing
            st.write("") 
            if st.button("Toevoegen"):
                if p_ticker and p_amount > 0:
                    # Toevoegen aan sessie
                    st.session_state['portfolio'].append({
                        "Ticker": p_ticker,
                        "Aantal": p_amount,
                        "Koopprijs": p_avg
                    })
                    st.success(f"{p_ticker} toegevoegd!")
                    st.rerun()

    # PORTFOLIO OVERZICHT
    if len(st.session_state['portfolio']) > 0:
        st.markdown("---")
        portfolio_data = []
        total_value = 0
        total_cost = 0

        # Loop door opgeslagen aandelen
        for item in st.session_state['portfolio']:
            current_price = get_current_price(item['Ticker'])
            cur_val = current_price * item['Aantal']
            cost_val = item['Koopprijs'] * item['Aantal']
            
            total_value += cur_val
            total_cost += cost_val
            
            profit_loss = cur_val - cost_val
            profit_pct = ((current_price - item['Koopprijs']) / item['Koopprijs']) * 100 if item['Koopprijs'] > 0 else 0

            portfolio_data.append({
                "Ticker": item['Ticker'],
                "Aantal": item['Aantal'],
                "Koopprijs": f"{curr_symbol}{item['Koopprijs']:.2f}",
                "Huidige Prijs": f"{curr_symbol}{current_price:.2f}",
                "Waarde": f"{curr_symbol}{cur_val:.2f}",
                "Winst/Verlies": f"{curr_symbol}{profit_loss:.2f} ({profit_pct:.1f}%)"
            })

        # Toon Tabel
        st.table(pd.DataFrame(portfolio_data))

        # Totaal Metrics
        tot_profit = total_value - total_cost
        tot_profit_pct = (tot_profit / total_cost) * 100 if total_cost > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Totale Waarde", f"{curr_symbol}{total_value:.2f}")
        m2.metric("Totale Inleg", f"{curr_symbol}{total_cost:.2f}")
        m3.metric("Totaal Winst/Verlies", f"{curr_symbol}{tot_profit:.2f}", f"{tot_profit_pct:.1f}%")
        
        # Reset Knop
        if st.button("🗑️ Portfolio Wissen"):
            st.session_state['portfolio'] = []
            st.rerun()
            
    else:
        st.write("Je portfolio is nog leeg. Voeg hierboven een aandeel toe.")

st.markdown("---")
st.markdown("© 2025 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")
