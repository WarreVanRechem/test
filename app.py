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
st.set_page_config(page_title="Zenith Terminal v8.1", layout="wide", page_icon="💎")
warnings.filterwarnings("ignore")

# --- INITIALISEER SESSIE ---
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

@st.cache_resource
def load_ai():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

ai_pipe = load_ai()

# --- HULPFUNCTIES ---
def get_current_price(ticker):
    """Haalt de prijs op (Robuust)."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info.last_price
        if price and not pd.isna(price) and price > 0: return price
        hist = stock.history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        hist_5d = stock.history(period="5d")
        if not hist_5d.empty: return hist_5d['Close'].iloc[-1]
    except: pass
    return 0.0

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
        
        market_bull = False
        if not market.empty:
            market_bull = market['Close'].iloc[-1] > market['Close'].rolling(200).mean().iloc[-1]

        metrics = {
            "price": df['Close'].iloc[-1],
            "sma200": df['SMA200'].iloc[-1],
            "rsi": df['RSI'].iloc[-1],
            "market_bull": market_bull,
            "var": np.percentile(df['Returns'].dropna(), 5)
        }
        return df, metrics
    except:
        return None, None

def get_external_info(ticker):
    """Haalt Insiders EN Nieuws op (De zware functie)."""
    buys, news_results = 0, []
    try:
        # 1. Insider Check
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is not None and not insider.empty:
            buys = insider.head(10)[insider.head(10)['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
        
        # 2. AI Nieuws Check
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+finance&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(rss_url, headers=headers, timeout=5)
        feed = feedparser.parse(response.content)
        
        for entry in feed.entries[:5]: # Max 5 artikelen voor snelheid
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

st.sidebar.markdown("---")
currency_mode = st.sidebar.radio("Valuta", ["USD ($)", "EUR (€)"])
curr_symbol = "$" if "USD" in currency_mode else "€"

st.sidebar.markdown("---")
st.sidebar.markdown("### Credits")
st.sidebar.markdown("Created by **Warre Van Rechem**")
st.sidebar.markdown("[Connect on LinkedIn](https://www.linkedin.com/in/warre-van-rechem-928723298/)")

# ==========================================
# PAGINA 1: MARKT ANALYSE
# ==========================================
if page == "🔎 Markt Analyse":
    st.title("💎 Zenith Institutional Terminal") 
    st.warning("⚠️ **Disclaimer:** Deze tool is uitsluitend educatief. Géén financieel advies.")

    col_input, col_cap = st.columns(2)
    with col_input: ticker_input = st.text_input("Ticker", "RDW").upper()
    with col_cap: capital = st.number_input(f"Virtueel Kapitaal ({curr_symbol})", value=10000)
    
    if st.button("Start Deep Analysis"):
        df, metrics = get_zenith_data(ticker_input)
        
        if df is not None:
            with st.spinner('Bezig met volledige AI scan...'):
                buys, news = get_external_info(ticker_input)
            
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

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Zenith Score", f"{score}/100")
            c2.metric("Prijs", f"{curr_symbol}{metrics['price']:.2f}")
            c3.metric("RSI", f"{metrics['rsi']:.1f}")
            c4.metric("Risk (VaR)", f"{curr_symbol}{abs(metrics['var'] * capital):.0f}")

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
# PAGINA 2: MIJN PORTFOLIO
# ==========================================
elif page == "💼 Mijn Portfolio":
    st.title("💼 Mijn Portfolio Manager")
    
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
                    st.success(f"{p_ticker} toegevoegd!")
                    st.rerun()

    if len(st.session_state['portfolio']) > 0:
        st.markdown("---")
        portfolio_data = []
        total_value = 0
        total_cost = 0

        prog_bar = st.progress(0)
        for i, item in enumerate(st.session_state['portfolio']):
            prog_bar.progress((i + 1) / len(st.session_state['portfolio']))
            current_price = get_current_price(item['Ticker'])
            cur_val = current_price * item['Aantal']
            cost_val = item['Koopprijs'] * item['Aantal']
            total_value += cur_val
            total_cost += cost_val
            profit_loss = cur_val - cost_val
            profit_pct = ((current_price - item['Koopprijs']) / item['Koopprijs']) * 100 if item['Koopprijs'] > 0 else 0
            
            color = "green" if profit_loss >= 0 else "red"
            profit_str = f":{color}[{curr_symbol}{profit_loss:.2f} ({profit_pct:.1f}%)]"

            portfolio_data.append({
                "Ticker": item['Ticker'],
                "Aantal": item['Aantal'],
                "Koopprijs": f"{curr_symbol}{item['Koopprijs']:.2f}",
                "Huidige Prijs": f"{curr_symbol}{current_price:.2f}",
                "Waarde": f"{curr_symbol}{cur_val:.2f}",
                "Winst/Verlies": profit_str
            })
        prog_bar.empty()

        st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)

        tot_profit = total_value - total_cost
        tot_profit_pct = (tot_profit / total_cost) * 100 if total_cost > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Totale Waarde", f"{curr_symbol}{total_value:.2f}")
        m2.metric("Totale Inleg", f"{curr_symbol}{total_cost:.2f}")
        m3.metric("Totaal Winst", f"{curr_symbol}{tot_profit:.2f}", f"{tot_profit_pct:.1f}%")
        
        if st.button("🗑️ Portfolio Wissen"):
            st.session_state['portfolio'] = []
            st.rerun()
    else: st.write("Portfolio is leeg.")

# ==========================================
# PAGINA 3: DEEP SCANNER (FULL POWER)
# ==========================================
elif page == "📡 Deep Scanner":
    st.title("📡 Zenith Deep Market Scanner")
    st.info("⚠️ **Let op:** Deze scanner analyseert ALLES: Data + Nieuws (AI) + Insiders. Dit kan 3-5 seconden per aandeel duren.")

    default_tickers = "AAPL, NVDA, TSLA, AMD, MSFT, ASML.AS"
    scan_input = st.text_area("Voer tickers in (gescheiden door komma's)", default_tickers)

    if st.button("🚀 Start Deep Scan"):
        tickers_to_scan = [t.strip().upper() for t in scan_input.split(",") if t.strip()]
        
        results = []
        progress_text = "Analyseren..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, ticker in enumerate(tickers_to_scan):
            # Update de balk met tekst zodat je weet waar hij is
            my_bar.progress((i) / len(tickers_to_scan), text=f"🔍 Analyseren van {ticker} (Tech + AI + Insiders)...")
            
            # 1. Technische Data
            df, metrics = get_zenith_data(ticker)
            
            # 2. Insiders & Nieuws (De "Zware" taak)
            if df is not None:
                buys, news = get_external_info(ticker) # Dit duurt even!
                
                # 3. Bereken de VOLLEDIGE Zenith Score
                score = 0
                
                # A. Markt
                if metrics['market_bull']: score += 20
                
                # B. Trend
                if metrics['price'] > metrics['sma200']: score += 30
                
                # C. RSI
                if metrics['rsi'] < 30: score += 20
                elif metrics['rsi'] > 70: pass # Geen punten aftrek in de scanner voor netheid, maar geen bonus
                else: score += 0
                
                # D. Insiders (Nu actief in de scanner!)
                if buys > 0: score += 20
                
                # E. AI Sentiment (Nu actief in de scanner!)
                pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE')
                if pos_news >= 2: score += 10
                
                # Sla op
                results.append({
                    "Ticker": ticker,
                    "Prijs": metrics['price'],
                    "RSI": round(metrics['rsi'], 1),
                    "Insiders": f"{buys} Buys" if buys > 0 else "-",
                    "AI Sentiment": f"{pos_news} Positief",
                    "Totale Score": score
                })
        
        # Klaar!
        my_bar.progress(1.0, text="Scan Voltooid!")
        
        if results:
            scan_df = pd.DataFrame(results).sort_values(by="Totale Score", ascending=False)
            st.success(f"✅ Deep Scan voltooid! {len(results)} aandelen volledig doorgelicht.")
            
            # Mooie tabel
            st.dataframe(
                scan_df,
                column_config={
                    "Totale Score": st.column_config.ProgressColumn(
                        "Zenith Score",
                        format="%d",
                        min_value=0,
                        max_value=100,
                    ),
                    "Prijs": st.column_config.NumberColumn(format=f"{curr_symbol}%.2f"),
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.error("Geen data gevonden.")

st.markdown("---")
st.markdown("© 2025 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")
