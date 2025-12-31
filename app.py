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

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal v11.0", layout="wide", page_icon="💎")
warnings.filterwarnings("ignore")

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

@st.cache_resource
def load_ai():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

ai_pipe = load_ai()

PRESETS = {
    "🇺🇸 Big Tech & AI": "NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, AMD, PLTR",
    "🇪🇺 AEX & Bel20": "ASML.AS, ADYEN.AS, BES.AS, SHELL.AS, KBC.BR, UCB.BR, SOLB.BR",
    "🚀 High Growth": "COIN, MSTR, SMCI, HOOD, PLTR, SOFI, RIVN",
    "🛡️ Defensive": "KO, JNJ, PEP, MCD, O, V, BRK-B"
}

# --- THESIS ENGINE ---
def generate_thesis(ticker, metrics, buys, pos_news, fundamentals):
    """Genereert een narratief op basis van Technisch, AI én Fundamenteel."""
    thesis = []
    signal_strength = "NEUTRAAL"
    
    # 1. Technisch
    if metrics['price'] > metrics['sma200']:
        trend_text = "Technisch zit het aandeel in een Bullish trend (>200MA)."
    else:
        trend_text = "Technisch is de trend zwak (onder de 200MA)."
    
    # 2. Fundamenteel (Nieuw in v11)
    pe_text = ""
    if fundamentals['pe'] > 0:
        if fundamentals['pe'] > 50: pe_text = "Het aandeel is wel erg duur (P/E > 50)."
        elif fundamentals['pe'] < 15: pe_text = "Het aandeel is fundamenteel goedkoop (P/E < 15)."
    
    # 3. Combinatie Scenarios
    # Scenario A: The "Value Dip" (Trend UP + RSI Low + Cheap PE)
    if metrics['price'] > metrics['sma200'] and metrics['rsi'] < 40 and fundamentals['pe'] < 25:
        thesis.append(f"🔥 **GOUDEN KANS:** {trend_text} De RSI is laag én de waardering is redelijk. Dit is een klassieke 'Buy the Dip'.")
        signal_strength = "STERK KOPEN"
        
    # Scenario B: The "Expensive Hype" (Trend UP + RSI High + High PE)
    elif metrics['rsi'] > 75 and fundamentals['pe'] > 60:
        thesis.append(f"⚠️ **BUBBEL GEVAAR:** {trend_text} Maar met een P/E van {fundamentals['pe']:.0f} en een RSI van {metrics['rsi']:.0f} is dit perfectie ingeprijsd. Winst nemen?")
        signal_strength = "VERKOPEN / AFROVEN"

    # Default
    else:
        thesis.append(f"ℹ️ **ANALYSE:** {trend_text} {pe_text} RSI is {metrics['rsi']:.0f}.")
        if buys > 0: thesis.append(f"Insiders kochten recent {buys}x, wat vertrouwen geeft.")
        if pos_news >= 2: thesis.append("Het nieuws-sentiment is positief.")

    return " ".join(thesis), signal_strength

# --- DATA FUNCTIES ---
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info.last_price
        if price and not pd.isna(price) and price > 0: return price
        hist = stock.history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
    except: pass
    return 0.0

@st.cache_data(ttl=3600)
def get_zenith_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        market = yf.Ticker("^GSPC").history(period="7y")
        if df.empty: return None, None, None
        
        # Fundamentele Info (Nieuw)
        info = stock.info
        fundamentals = {
            "pe": info.get('trailingPE', 0),
            "market_cap": info.get('marketCap', 0),
            "dividend": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            "sector": info.get('sector', "Onbekend"),
            "profit_margin": info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
        }

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
        return df, metrics, fundamentals
    except: return None, None, None

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

# --- INTERFACE ---
st.sidebar.error("⚠️ **DISCLAIMER:** Geen financieel advies. Educatief gebruik.")
st.sidebar.header("Navigatie")
page = st.sidebar.radio("Ga naar:", ["🔎 Markt Analyse", "💼 Mijn Portfolio", "📡 Deep Scanner"])

# --- NIEUW: RISK CALCULATOR IN SIDEBAR ---
st.sidebar.markdown("---")
with st.sidebar.expander("🧮 Position Size Calculator"):
    st.write("Bereken je positiegrootte")
    acc_size = st.number_input("Account Grootte", value=10000)
    risk_pct = st.slider("Risico %", 0.5, 5.0, 1.0)
    entry_p = st.number_input("Instap Prijs", value=100.0)
    stop_p = st.number_input("Stop Loss", value=95.0)
    
    if stop_p < entry_p:
        risk_per_share = entry_p - stop_p
        total_risk_amount = acc_size * (risk_pct / 100)
        shares_to_buy = total_risk_amount / risk_per_share
        st.markdown(f"**Koop:** `{int(shares_to_buy)}` aandelen")
        st.caption(f"Je riskeert ${total_risk_amount:.2f}")
    else:
        st.warning("Stop Loss moet lager zijn dan Instap.")
# -----------------------------------------

currency_mode = st.sidebar.radio("Valuta", ["USD ($)", "EUR (€)"])
curr_symbol = "$" if "USD" in currency_mode else "€"

st.sidebar.markdown("---")
st.sidebar.markdown("Created by **Warre Van Rechem**")

# ==========================================
# PAGINA 1: ANALYSE (PRO)
# ==========================================
if page == "🔎 Markt Analyse":
    st.title("💎 Zenith Institutional Terminal") 
    
    col_input, col_cap = st.columns(2)
    with col_input: ticker_input = st.text_input("Ticker", "RDW").upper()
    with col_cap: capital = st.number_input(f"Virtueel Kapitaal ({curr_symbol})", value=10000)
    
    if st.button("Start Deep Analysis"):
        df, metrics, fundamentals = get_zenith_data(ticker_input)
        
        if df is not None:
            with st.spinner('Fundamentele & AI Analyse draait...'):
                buys, news = get_external_info(ticker_input)
            
            # Scoring Logic
            score = 0
            if metrics['market_bull']: score += 15
            if metrics['price'] > metrics['sma200']: score += 25
            if metrics['rsi'] < 30: score += 15
            if buys > 0: score += 15
            pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE')
            if pos_news >= 2: score += 10
            # Fundamentele Bonus
            if 0 < fundamentals['pe'] < 25: score += 10
            if fundamentals['profit_margin'] > 20: score += 10
            
            thesis_text, signal = generate_thesis(ticker_input, metrics, buys, pos_news, fundamentals)

            # HEADER METRICS
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1: st.metric("Zenith Score", f"{score}/100")
            with c2: 
                sig_color = "green" if "KOPEN" in signal else "red" if "VERKOPEN" in signal else "orange"
                st.markdown(f"### Advies: :{sig_color}[**{signal}**]")
            with c3: st.metric("Prijs", f"{curr_symbol}{metrics['price']:.2f}")

            # FUNDAMENTELE RIJ (NIEUW)
            st.markdown("##### 🏢 Fundamentele Gezondheid")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Sector", fundamentals['sector'])
            f2.metric("P/E Ratio (Waardering)", f"{fundamentals['pe']:.1f}")
            f3.metric("Winstmarge", f"{fundamentals['profit_margin']:.1f}%")
            f4.metric("Dividend", f"{fundamentals['dividend']:.2f}%")
            st.markdown("---")

            # AI THESIS
            st.info(f"🤖 **Zenith AI Thesis:**\n\n{thesis_text}")

            # GRAFIEK MET VOLUME (NIEUW)
            end_date = df.index[-1]
            start_date = end_date - pd.DateOffset(years=3) # 3 jaar voor snelheid
            plot_df = df.loc[start_date:end_date]
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                                row_heights=[0.6, 0.2, 0.2]) # Koers, Volume, RSI

            # 1. Candlestick
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Prijs"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700', width=2), name="200 MA"), row=1, col=1)
            
            # 2. Volume (NIEUW)
            colors = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in plot_df.iterrows()]
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
            
            # 3. RSI
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#9370DB', width=2), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
            
            fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # NIEUWS
            st.subheader("📰 Relevant Nieuws")
            for n in news:
                color = "green" if n['sentiment'] == 'POSITIVE' else "red" if n['sentiment'] == 'NEGATIVE' else "gray"
                st.markdown(f":{color}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")
        else: st.error("Geen data gevonden.")

# ==========================================
# PAGINA 2: PORTFOLIO
# ==========================================
elif page == "💼 Mijn Portfolio":
    st.title("💼 Portfolio Manager")
    with st.expander("➕ Aandeel Toevoegen", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1: p_ticker = st.text_input("Ticker", key="p_ticker").upper()
        with c2: p_amount = st.number_input("Aantal", min_value=0.0, step=1.0, key="p_amount")
        with c3: p_avg = st.number_input(f"Koopprijs ({curr_symbol})", min_value=0.0, step=0.1, key="p_avg")
        with c4:
            st.write("")
            st.write("")
            if st.button("Toevoegen"):
                if p_ticker and p_amount > 0:
                    st.session_state['portfolio'].append({"Ticker": p_ticker, "Aantal": p_amount, "Koopprijs": p_avg})
                    st.success("Toegevoegd!")
                    st.rerun()

    if len(st.session_state['portfolio']) > 0:
        st.markdown("---")
        portfolio_data = []
        total_value, total_cost = 0, 0
        prog_bar = st.progress(0)
        
        for i, item in enumerate(st.session_state['portfolio']):
            prog_bar.progress((i + 1) / len(st.session_state['portfolio']))
            current_price = get_current_price(item['Ticker'])
            cur_val = current_price * item['Aantal']
            cost_val = item['Koopprijs'] * item['Aantal']
            total_value += cur_val
            total_cost += cost_val
            profit = cur_val - cost_val
            profit_pct = ((current_price - item['Koopprijs']) / item['Koopprijs']) * 100 if item['Koopprijs'] > 0 else 0
            
            color = "green" if profit >= 0 else "red"
            portfolio_data.append({
                "Ticker": item['Ticker'],
                "Aantal": item['Aantal'],
                "Koopprijs": f"{curr_symbol}{item['Koopprijs']:.2f}",
                "Huidige Prijs": f"{curr_symbol}{current_price:.2f}",
                "Waarde": f"{curr_symbol}{cur_val:.2f}",
                "Winst/Verlies": f":{color}[{curr_symbol}{profit:.2f} ({profit_pct:.1f}%)]"
            })
        prog_bar.empty()
        st.write(pd.DataFrame(portfolio_data).to_markdown(index=False), unsafe_allow_html=True)
        
        tot_profit = total_value - total_cost
        tot_profit_pct = (tot_profit / total_cost) * 100 if total_cost > 0 else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Waarde", f"{curr_symbol}{total_value:.2f}")
        m2.metric("Inleg", f"{curr_symbol}{total_cost:.2f}")
        m3.metric("Winst", f"{curr_symbol}{tot_profit:.2f}", f"{tot_profit_pct:.1f}%")
        
        if st.button("🗑️ Wissen"):
            st.session_state['portfolio'] = []
            st.rerun()
    else: st.write("Leeg.")

# ==========================================
# PAGINA 3: DEEP SCANNER
# ==========================================
elif page == "📡 Deep Scanner":
    st.title("📡 Zenith Market Scanner")
    preset = st.selectbox("📂 Kies Markt:", list(PRESETS.keys()))
    default_text = PRESETS.get(preset, "")
    scan_input = st.text_area("Tickers:", default_text, height=80)

    if st.button("🚀 Start Scan"):
        tickers = [t.strip().upper() for t in scan_input.split(",") if t.strip()]
        results = []
        my_bar = st.progress(0, text="Starten...")
        
        for i, ticker in enumerate(tickers):
            my_bar.progress((i)/len(tickers), text=f"🔍 {ticker}...")
            df, metrics, fund = get_zenith_data(ticker)
            if df is not None:
                buys, news = get_external_info(ticker)
                score = 0
                if metrics['market_bull']: score += 15
                if metrics['price'] > metrics['sma200']: score += 25
                if metrics['rsi'] < 30: score += 15
                if buys > 0: score += 15
                pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE')
                if pos_news >= 2: score += 10
                # Fundamentals in score
                if 0 < fund['pe'] < 25: score += 10
                
                results.append({
                    "Ticker": ticker,
                    "Prijs": metrics['price'],
                    "RSI": round(metrics['rsi'], 1),
                    "P/E Ratio": round(fund['pe'], 1),
                    "Insiders": f"{buys} Buys" if buys > 0 else "-",
                    "Score": score
                })
        my_bar.progress(1.0, text="Klaar!")
        if results:
            st.dataframe(pd.DataFrame(results).sort_values(by="Score", ascending=False), use_container_width=True, hide_index=True)
        else: st.error("Geen data.")

st.markdown("---")
st.markdown("© 2025 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")
