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
st.set_page_config(page_title="Zenith Terminal v17.1", layout="wide", page_icon="💎")
warnings.filterwarnings("ignore")

# --- SESSION STATE ---
if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
if 'nav_page' not in st.session_state: st.session_state['nav_page'] = "🔎 Markt Analyse"
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "RDW"

def start_analysis_for(ticker):
    st.session_state['selected_ticker'] = ticker
    st.session_state['nav_page'] = "🔎 Markt Analyse"
    st.session_state['auto_run'] = True

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

# --- MACRO DATA FUNCTIE ---
@st.cache_data(ttl=600)
def get_macro_data():
    tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Goud": "GC=F",
        "Olie (WTI)": "CL=F",
        "10Y Rente": "^TNX"
    }
    data = {}
    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            price = t.fast_info.last_price
            prev = t.fast_info.previous_close
            change = ((price - prev) / prev) * 100
            data[name] = (price, change)
        except:
            data[name] = (0, 0)
    return data

# --- THESIS ENGINE (UPDATED) ---
def generate_thesis(ticker, metrics, buys, pos_news, fundamentals, wall_street):
    thesis = []
    signal_strength = "NEUTRAAL" # Default
    
    # 1. Technisch
    trend_text = "Technisch Bullish (>200MA)." if metrics['price'] > metrics['sma200'] else "Technisch zwak (<200MA)."
    
    # 2. Wall Street
    ws_text = ""
    if wall_street['target'] > 0:
        if wall_street['upside'] > 10:
            ws_text = f"Analisten zien {wall_street['upside']:.1f}% upside."
        elif wall_street['upside'] < 0:
            ws_text = f"Koers ligt boven het koersdoel (${wall_street['target']:.2f})."
            
    # 3. Dividend
    div_text = ""
    if fundamentals['dividend'] > 4.0:
        div_text = f"Dividendrendement is aantrekkelijk ({fundamentals['dividend']:.2f}%)."
    
    # --- LOGICA UPDATE: IETS SOEPELER ---
    # Strong Buy
    if metrics['price'] > metrics['sma200'] and wall_street['upside'] > 10:
        thesis.append(f"🔥 **STERK:** {trend_text} {ws_text} Fundamentals ondersteunen de groei.")
        signal_strength = "STERK KOPEN"
        
    # Speculative Buy (Oversold)
    elif metrics['rsi'] < 30:
        thesis.append(f"🛒 **KOOPKANS:** Het aandeel is zwaar afgestraft (RSI < 30). Mogelijk een goed instapmoment voor een rebound.")
        signal_strength = "KOOP (DIP)"

    # Sell / Avoid
    elif metrics['price'] < metrics['sma200'] and wall_street['upside'] < 5:
        thesis.append(f"⚠️ **OPGELET:** Trend is neerwaarts en analisten zien weinig potentieel.")
        signal_strength = "AFBLIJVEN / VERKOPEN"
        
    # Neutraal (Default)
    else:
        thesis.append(f"ℹ️ **HOUDEN:** {trend_text} {ws_text} Geen uitgesproken signaal.")
        if buys > 0: thesis.append(f"Positief: Insiders kochten {buys}x.")

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
        if df.empty: return None, None, None, None, None
        
        info = stock.info
        current_p = df['Close'].iloc[-1]

        # Dividend Fix
        div_rate = info.get('dividendRate') 
        if div_rate is None: div_rate = info.get('trailingAnnualDividendRate')
        dividend_pct = 0.0
        if div_rate is not None and current_p > 0:
            dividend_pct = (div_rate / current_p) * 100
        else:
            raw_div = info.get('dividendYield')
            if raw_div is not None:
                dividend_pct = raw_div * 100 if raw_div < 0.5 else raw_div

        fundamentals = {
            "pe": info.get('trailingPE', 0),
            "market_cap": info.get('marketCap', 0),
            "dividend": dividend_pct, 
            "sector": info.get('sector', "Onbekend"),
            "profit_margin": (info.get('profitMargins') or 0) * 100
        }
        
        # Wall Street
        target_p = info.get('targetMeanPrice', 0)
        if target_p is None: target_p = 0
        upside = ((target_p - current_p) / current_p) * 100 if target_p > 0 else 0
        
        wall_street = {
            "target": target_p,
            "recommendation": info.get('recommendationKey', 'none').upper(),
            "upside": upside
        }

        # Metrics
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Returns'] = df['Close'].pct_change()
        
        # Alpha Calc
        start_compare = df.index[-500] if len(df) > 500 else df.index[0]
        market_subset = market.loc[df.index]
        df['Rel_Perf'] = df['Close'] / df['Close'].iloc[0]
        df['Market_Perf'] = (market_subset['Close'] / market_subset['Close'].iloc[0]) * df['Close'].iloc[0] 
        
        metrics = {
            "price": current_p,
            "sma200": df['SMA200'].iloc[-1],
            "rsi": df['RSI'].iloc[-1],
            "market_bull": market['Close'].iloc[-1] > market['Close'].rolling(200).mean().iloc[-1],
            "var": np.percentile(df['Returns'].dropna(), 5)
        }
        return df, metrics, fundamentals, wall_street, market
    except Exception as e: 
        return None, None, None, None, None

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
st.sidebar.header("Navigatie")
page = st.sidebar.radio("Ga naar:", ["🔎 Markt Analyse", "💼 Mijn Portfolio", "📡 Deep Scanner"], key="nav_page")

with st.sidebar.expander("🧮 Risk Calculator"):
    acc_size = st.number_input("Account", value=10000)
    risk_pct = st.slider("Risico %", 0.5, 5.0, 1.0)
    entry_p = st.number_input("Instap", value=100.0)
    stop_p = st.number_input("Stop Loss", value=95.0)
    if stop_p < entry_p:
        risk_per_share = entry_p - stop_p
        shares = (acc_size * (risk_pct/100)) / risk_per_share
        st.write(f"**Koop:** {int(shares)} stuks")

currency_mode = st.sidebar.radio("Valuta", ["USD ($)", "EUR (€)"])
curr_symbol = "$" if "USD" in currency_mode else "€"
st.sidebar.markdown("---")
st.sidebar.markdown("Created by **Warre Van Rechem**")

# --- MACRO HEADER ---
st.title("💎 Zenith Institutional Terminal") 
macro = get_macro_data()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("S&P 500", f"{macro['S&P 500'][0]:.0f}", f"{macro['S&P 500'][1]:.2f}%")
m2.metric("Nasdaq", f"{macro['Nasdaq'][0]:.0f}", f"{macro['Nasdaq'][1]:.2f}%")
m3.metric("Goud", f"${macro['Goud'][0]:.0f}", f"{macro['Goud'][1]:.2f}%")
m4.metric("Olie", f"${macro['Olie (WTI)'][0]:.2f}", f"{macro['Olie (WTI)'][1]:.2f}%")
m5.metric("10Y Rente", f"{macro['10Y Rente'][0]:.2f}%", f"{macro['10Y Rente'][1]:.2f}%")
st.markdown("---")

# ==========================================
# PAGINA 1: ANALYSE
# ==========================================
if page == "🔎 Markt Analyse":
    col_input, col_cap = st.columns(2)
    with col_input: ticker_input = st.text_input("Ticker", value=st.session_state['selected_ticker']).upper()
    with col_cap: capital = st.number_input(f"Virtueel Kapitaal ({curr_symbol})", value=10000)
    
    auto_run = st.session_state.get('auto_run', False)
    
    if st.button("Start Deep Analysis") or auto_run:
        if auto_run: st.session_state['auto_run'] = False
        
        df, metrics, fund, wall_street, market_data = get_zenith_data(ticker_input)
        
        if df is not None:
            with st.spinner('Analyseren...'):
                buys, news = get_external_info(ticker_input)
            
            score = 0
            if metrics['market_bull']: score += 15
            if metrics['price'] > metrics['sma200']: score += 20
            if metrics['rsi'] < 30: score += 15
            if buys > 0: score += 15
            pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE')
            if pos_news >= 2: score += 10
            if 0 < fund['pe'] < 25: score += 10
            if wall_street['upside'] > 10: score += 15
            
            thesis_text, signal = generate_thesis(ticker_input, metrics, buys, pos_news, fund, wall_street)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Zenith Score", f"{score}/100")
            
            # --- HIER ZIT DE FIX VOOR DE KLEUR ---
            if "KOPEN" in signal or "KOOP" in signal:
                sig_color = "green"
            elif "VERKOPEN" in signal or "AFBLIJVEN" in signal:
                sig_color = "red"
            else:
                sig_color = "orange" # Nu orange ipv off
            
            c2.markdown(f"**Advies:** :{sig_color}[{signal}]")
            c3.metric("Huidige Prijs", f"{curr_symbol}{metrics['price']:.2f}")
            c4.metric("Analisten Doel", f"{curr_symbol}{wall_street['target']:.2f}", f"{wall_street['upside']:.1f}% Upside")

            st.markdown("---")
            col_thesis, col_fund = st.columns([2, 1])
            with col_thesis:
                st.subheader("📝 Zenith AI Thesis")
                st.info(f"{thesis_text}")
                st.caption(f"**Wall Street Consensus:** {wall_street['recommendation'].replace('_', ' ')}")
            with col_fund:
                st.subheader("🏢 Fundamenteel")
                st.metric("P/E Ratio", f"{fund['pe']:.2f}")
                st.metric("Dividend Yield", f"{fund['dividend']:.2f}%")
                st.metric("Winstmarge", f"{fund['profit_margin']:.1f}%")

            st.markdown("---")
            st.subheader("📈 Alpha Grafiek (vs S&P 500)")
            end_date = df.index[-1]
            start_date = end_date - pd.DateOffset(years=2)
            plot_df = df.loc[start_date:end_date]
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Prijs"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700', width=2), name="200 MA"), row=1, col=1)
            scale_factor = plot_df['Close'].iloc[0] / market_data.loc[plot_df.index[0]]['Close']
            scaled_market = market_data.loc[plot_df.index]['Close'] * scale_factor
            fig.add_trace(go.Scatter(x=plot_df.index, y=scaled_market, line=dict(color='gray', width=1, dash='dot'), name="S&P 500 (Ref)"), row=1, col=1)

            colors = ['green' if r['Open'] < r['Close'] else 'red' for i, r in plot_df.iterrows()]
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#9370DB', width=2), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
            fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("📰 Laatste Nieuws")
            if news:
                n_cols = st.columns(2)
                for i, n in enumerate(news):
                    col = n_cols[i % 2]
                    color = "green" if n['sentiment'] == 'POSITIVE' else "red" if n['sentiment'] == 'NEGATIVE' else "gray"
                    col.markdown(f":{color}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")
        else: st.error("Geen data.")

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
        
        df_port = pd.DataFrame(portfolio_data)
        st.write(df_port.to_markdown(index=False), unsafe_allow_html=True)
        st.download_button("📥 Download Portfolio (CSV)", df_port.to_csv(index=False), "portfolio.csv", "text/csv")
        
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
# PAGINA 3: SCANNER
# ==========================================
elif page == "📡 Deep Scanner":
    st.title("📡 Zenith Market Scanner")
    preset = st.selectbox("📂 Kies Markt:", list(PRESETS.keys()))
    scan_input = st.text_area("Tickers:", PRESETS.get(preset, ""), height=80)

    if st.button("🚀 Start Scan"):
        tickers = [t.strip().upper() for t in scan_input.split(",") if t.strip()]
        results = []
        my_bar = st.progress(0, text="Starten...")
        for i, ticker in enumerate(tickers):
            my_bar.progress((i)/len(tickers), text=f"🔍 {ticker}...")
            df, metrics, fund, ws, _ = get_zenith_data(ticker)
            if df is not None:
                buys, news = get_external_info(ticker)
                
                score = 0
                reasons = []
                if metrics['market_bull']: score += 15
                if metrics['price'] > metrics['sma200']: 
                    score += 20; reasons.append("🚀 Trend")
                if metrics['rsi'] < 30: 
                    score += 15; reasons.append("📉 Oversold")
                if buys > 0: 
                    score += 15; reasons.append("🏛️ Insiders")
                pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE')
                if pos_news >= 2: 
                    score += 10; reasons.append("🤖 AI Positief")
                if 0 < fund['pe'] < 25: 
                    score += 10; reasons.append("💰 Goedkoop")
                if ws['upside'] > 10: 
                    score += 15; reasons.append("💼 Wall St")
                
                advies = "NEUTRAAL"
                if score >= 70: advies = "🟢 STERK KOPEN"
                elif score >= 50: advies = "🟡 KOPEN / HOUDEN"
                else: advies = "🔴 AFBLIJVEN"
                
                results.append({
                    "Ticker": ticker,
                    "Prijs": metrics['price'],
                    "Analist Doel": f"{curr_symbol}{ws['target']:.2f}",
                    "Score": score,
                    "Advies": advies,
                    "Reden": " + ".join(reasons) if reasons else "Geen triggers"
                })
        my_bar.progress(1.0, text="Klaar!")
        if results: st.session_state['scan_results'] = results 
        else: st.error("Geen data.")

    if 'scan_results' in st.session_state and st.session_state['scan_results']:
        results = st.session_state['scan_results']
        df_scan = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        
        st.dataframe(
            df_scan, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100),
                "Prijs": st.column_config.NumberColumn("Prijs", format=f"{curr_symbol}%.2f")
            }
        )
        st.download_button("📥 Download Scan Resultaten (CSV)", df_scan.to_csv(index=False), "scanner_results.csv", "text/csv")

        st.markdown("---")
        st.subheader("🔍 Wil je een aandeel dieper analyseren?")
        c1, c2 = st.columns([3, 1])
        options = [r['Ticker'] for r in results]
        selected_scan = c1.selectbox("Kies uit de lijst:", options)
        c2.button("🚀 Analyseer Nu", on_click=start_analysis_for, args=(selected_scan,))

st.markdown("---")
st.markdown("© 2025 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")
