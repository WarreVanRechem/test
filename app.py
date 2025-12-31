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

# Try-except voor scipy (nodig voor Portfolio Optimalisatie)
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal v21.0 Complete", layout="wide", page_icon="💎")
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
    try: return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except: return None

ai_pipe = load_ai()

PRESETS = {
    "🇺🇸 Big Tech & AI": "NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, AMD, PLTR",
    "🇪🇺 AEX & Bel20": "ASML.AS, ADYEN.AS, BESI.AS, SHELL.AS, KBC.BR, UCB.BR, SOLB.BR, ABI.BR, INGA.AS",
    "🚀 High Growth": "COIN, MSTR, SMCI, HOOD, PLTR, SOFI, RIVN",
    "🛡️ Defensive": "KO, JNJ, PEP, MCD, O, V, BRK-B"
}

# --- QUANT FUNCTIONS ---

def run_backtest(ticker, period="5y"):
    """Backtest de Zenith Strategie (Trend + RSI Dip)."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty: return "Geen data."
        if len(df) < 250: return "Te weinig data (<250 dagen)."
        
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        balance = 10000; shares = 0; trades = []
        in_position = False
        
        # Start pas na 200 dagen (SMA nodig)
        for i in range(201, len(df)):
            price = df['Close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            sma = df['SMA200'].iloc[i]
            date = df.index[i]
            
            if pd.isna(sma) or pd.isna(rsi): continue

            # Buy Signal
            if not in_position and price > sma and rsi < 35:
                shares = balance / price
                balance = 0; in_position = True
                trades.append({"Date": date, "Type": "BUY", "Price": price})
            # Sell Signal
            elif in_position and (rsi > 75 or price < sma * 0.95):
                balance = shares * price
                shares = 0; in_position = False
                trades.append({"Date": date, "Type": "SELL", "Price": price})
                
        final_val = balance if not in_position else shares * df['Close'].iloc[-1]
        ret = ((final_val - 10000) / 10000) * 100
        
        start_p = df['Close'].iloc[201]
        end_p = df['Close'].iloc[-1]
        bh_ret = ((end_p - start_p) / start_p) * 100
        
        return {"return": ret, "bh_return": bh_ret, "trades": len(trades), "final_value": final_val, "history": df}
    except Exception as e: return f"Error: {str(e)}"

def run_monte_carlo(ticker, simulations=200): # 200 sims is sneller voor webapp
    try:
        data = yf.Ticker(ticker).history(period="1y")['Close']
        returns = data.pct_change().dropna()
        if returns.empty: return None
        mu, sigma = returns.mean(), returns.std()
        sim_runs = []
        for _ in range(simulations):
            prices = [data.iloc[-1]]
            for _ in range(252): # 1 jaar vooruit
                prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
            sim_runs.append(prices)
        return np.array(sim_runs)
    except: return None

def optimize_portfolio(tickers):
    if not SCIPY_AVAILABLE: return "SCIPY_MISSING"
    try:
        data = yf.download(tickers, period="1y")['Close'].dropna()
        if data.empty or len(tickers) < 2: return None
        returns = data.pct_change()
        mean_ret = returns.mean(); cov_mat = returns.cov()
        
        def neg_sharpe(weights):
            p_ret = np.sum(mean_ret * weights) * 252
            p_var = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
            return -(p_ret - 0.04) / p_var if p_var > 0 else 0
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        res = minimize(neg_sharpe, len(tickers)*[1./len(tickers)], bounds=bounds, constraints=constraints)
        return dict(zip(tickers, res.x))
    except: return None

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

@st.cache_data(ttl=600)
def get_macro_data():
    tickers = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Goud": "GC=F", "Olie": "CL=F", "10Y Rente": "^TNX"}
    data = {}
    for name, t in tickers.items():
        try:
            obj = yf.Ticker(t)
            p = obj.fast_info.last_price
            prev = obj.fast_info.previous_close
            if p and prev: data[name] = (p, ((p-prev)/prev)*100)
            else: data[name] = (0,0)
        except: data[name] = (0,0)
    return data

@st.cache_data(ttl=3600)
def get_zenith_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        if df.empty: return None, None, None, None, None, None, "Geen data."
        
        info = stock.info
        current_p = df['Close'].iloc[-1]
        
        # Fundamentals
        d_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate')
        d_yield = (d_rate/current_p)*100 if (d_rate and current_p>0) else (info.get('dividendYield',0)*100 if info.get('dividendYield',0)<0.5 else info.get('dividendYield',0))
        fund = {"pe": info.get('trailingPE', 0), "dividend": d_yield, "sector": info.get('sector', "-"), "profit_margin": (info.get('profitMargins') or 0)*100}
        
        ws = {"target": info.get('targetMeanPrice', 0) or 0, "rec": info.get('recommendationKey', 'none').upper()}
        ws["upside"] = ((ws["target"] - current_p)/current_p)*100 if ws["target"] else 0

        # Technicals
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['StdDev'] = df['Close'].rolling(20).std()
        df['Upper'] = df['SMA20'] + (df['StdDev']*2)
        df['Lower'] = df['SMA20'] - (df['StdDev']*2)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Sniper
        entry = df['Lower'].iloc[-1]; low = df['Low'].tail(50).min(); high = df['High'].tail(50).max()
        if pd.isna(entry): entry = current_p
        risk = entry - (min(low, entry)*0.98); reward = high - entry
        sniper = {
            "entry_price": entry, "current_diff": ((current_p - entry)/current_p)*100,
            "stop_loss": min(low, entry)*0.98, "take_profit": high, "rr_ratio": reward/risk if risk > 0 else 0
        }

        # Market Alpha
        try:
            mkt = yf.Ticker("^GSPC").history(period="7y")
            if not mkt.empty:
                mkt_aligned = mkt['Close'].reindex(df.index, method='nearest')
                df['Market_Perf'] = (mkt_aligned / mkt_aligned.iloc[0]) * df['Close'].iloc[0]
                m_bull = mkt['Close'].iloc[-1] > mkt['Close'].rolling(200).mean().iloc[-1]
            else: df['Market_Perf'] = df['Close']; m_bull = True
        except: df['Market_Perf'] = df['Close']; m_bull = True

        metrics = {"name": info.get('longName', ticker), "price": current_p, "sma200": df['SMA200'].iloc[-1], "rsi": df['RSI'].iloc[-1], "market_bull": m_bull}
        return df, metrics, fund, ws, None, sniper, None
    except Exception as e: return None, None, None, None, None, None, str(e)

def get_external_info(ticker):
    buys, news = 0, []
    try:
        stock = yf.Ticker(ticker)
        ins = stock.insider_transactions
        if ins is not None and not ins.empty: buys = ins.head(10)[ins.head(10)['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
        feed = feedparser.parse(requests.get(f"https://news.google.com/rss/search?q={ticker}+stock+finance&hl=en-US&gl=US&ceid=US:en", headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).content)
        for e in feed.entries[:5]:
            sent = "NEUTRAL"
            if ai_pipe: 
                try: sent = ai_pipe(e.title[:512])[0]['label'].upper()
                except: pass
            news.append({"title": e.title, "sentiment": sent, "link": e.link})
    except: pass
    return buys, news

def generate_thesis(metrics, sniper, ws, buys):
    thesis = []
    sig = "NEUTRAAL"
    uptrend = metrics['price'] > metrics['sma200']
    trend_txt = "Trend is Bullish 🟢" if uptrend else "Trend is Bearish 🔴"
    diff = sniper['current_diff']
    in_zone = diff < 1.5
    
    if uptrend and in_zone: thesis.append(f"🔥 **PERFECT:** {trend_txt}. Prijs in 'Buy Zone'."); sig = "STERK KOPEN"
    elif not uptrend and in_zone: thesis.append(f"⚠️ **RISICO:** {trend_txt}, maar prijs op support. Speculatief."); sig = "SPECULATIEF KOPEN"
    elif uptrend and not in_zone: thesis.append(f"✅ **HOUDEN:** {trend_txt}. Wacht op dip (-{diff:.1f}%)."); sig = "HOUDEN / WACHTEN"
    else: thesis.append(f"🛑 **AFBLIJVEN:** {trend_txt}."); sig = "AFBLIJVEN"
    
    if ws['upside'] > 15: thesis.append(f"Analisten zien {ws['upside']:.0f}% upside.")
    return " ".join(thesis), sig

# --- UI ---
st.sidebar.header("Navigatie")
page = st.sidebar.radio("Ga naar:", ["🔎 Markt Analyse", "💼 Mijn Portfolio", "📡 Deep Scanner"], key="nav_page")

with st.sidebar.expander("🧮 Risk Calculator"):
    acc = st.number_input("Account", 10000); risk = st.slider("Risico %", 0.5, 5.0, 1.0)
    ent = st.number_input("Instap", 100.0); stp = st.number_input("Stop", 95.0)
    if stp < ent: st.write(f"**Koop:** {int((acc*(risk/100))/(ent-stp))} stuks")

curr_sym = "$" if "USD" in st.sidebar.radio("Valuta", ["USD ($)", "EUR (€)"]) else "€"
st.sidebar.markdown("---"); st.sidebar.markdown("Created by **Warre Van Rechem**")

st.title("💎 Zenith Institutional Terminal")
mac = get_macro_data()
c = st.columns(5)
for i, m in enumerate(["S&P 500", "Nasdaq", "Goud", "Olie", "10Y Rente"]):
    v, ch = mac.get(m, (0,0))
    c[i].metric(m, f"{v:.2f}", f"{ch:.2f}%")
st.markdown("---")

if page == "🔎 Markt Analyse":
    c1, c2 = st.columns(2)
    with c1: tick = st.text_input("Ticker", value=st.session_state['selected_ticker']).upper()
    with c2: cap = st.number_input(f"Virtueel Kapitaal ({curr_sym})", 10000)
    
    auto = st.session_state.get('auto_run', False)
    if st.button("Start Deep Analysis") or auto:
        if auto: st.session_state['auto_run'] = False
        df, met, fund, ws, _, snip, err = get_zenith_data(tick)
        
        if err: st.error(f"⚠️ {err}")
        elif df is not None:
            with st.spinner('Analyseren...'): buys, news = get_external_info(tick)
            
            score = 0
            if met['price'] > met['sma200']: score += 25
            if met['rsi'] < 35: score += 20
            elif met['rsi'] > 70: score -= 10
            if buys > 0: score += 15
            if ws['upside'] > 10: score += 15
            if snip['rr_ratio'] > 2: score += 15
            
            thesis, sig = generate_thesis(met, snip, ws, buys)
            
            st.markdown(f"## 🏢 {met['name']} ({tick})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Zenith Score", f"{score}/100")
            col = "green" if "KOPEN" in sig else "orange" if "SPECULATIEF" in sig else "red" if "AFBLIJVEN" in sig else "blue"
            m2.markdown(f"**Advies:** :{col}[{sig}]")
            m3.metric("Prijs", f"{curr_sym}{met['price']:.2f}")
            m4.metric("Analist Doel", f"{curr_sym}{ws['target']:.2f}", f"{ws['upside']:.1f}%")
            
            st.markdown("---")
            st.subheader("🎯 Sniper Setup & Fundamentals")
            s1, s2, s3, s4 = st.columns(4)
            msg = "✅ NU KOPEN!" if snip['current_diff'] < 1.5 else f"Wacht (-{snip['current_diff']:.1f}%)"
            s1.metric("Entry (Lower Band)", f"{curr_sym}{snip['entry_price']:.2f}", msg)
            s2.metric("Stop Loss", f"{curr_sym}{snip['stop_loss']:.2f}")
            s3.metric("Take Profit", f"{curr_sym}{snip['take_profit']:.2f}")
            rr_c = "green" if snip['rr_ratio']>=2 else "orange"
            s4.markdown(f"**R/R:** :{rr_c}[1 : {snip['rr_ratio']:.1f}]")
            
            st.info(f"**Zenith Thesis:** {thesis}")
            
            # --- DE GROTE GRAFIEK IS TERUG! (3 ROWS) ---
            st.subheader("📈 Technical Chart (Full View)")
            end = df.index[-1]; start = end - pd.DateOffset(years=1); plot_df = df.loc[start:end]
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            # Row 1: Price, BB, SMA, Alpha
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Lower'], line=dict(color='rgba(0,255,0,0.3)'), name="Lower Band"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Upper'], line=dict(color='rgba(255,0,0,0.3)'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name="Upper Band"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700', width=2), name="200 MA"), row=1, col=1)
            # Alpha Line Check
            if 'Market_Perf' in plot_df.columns:
                # Schaal de markt zodat hij start op zelfde punt als aandeel
                scale = plot_df['Close'].iloc[0] / plot_df['Market_Perf'].iloc[0]
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Market_Perf']*scale, line=dict(color='white', width=1, dash='dot'), name="S&P 500 (Ref)"), row=1, col=1)
            
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Prijs"), row=1, col=1)
            
            # Row 2: Volume (TERUG!)
            cols = ['green' if r['Open'] < r['Close'] else 'red' for i, r in plot_df.iterrows()]
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=cols, name="Volume"), row=2, col=1)
            
            # Row 3: RSI
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#9370DB'), name="RSI"), row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            
            fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # --- EXTRA TOOLS IN TABS ---
            t1, t2, t3 = st.tabs(["🔙 Backtest", "🔮 Monte Carlo", "📰 Nieuws"])
            
            with t1:
                if st.button("🚀 Draai Backtest (5 Jaar)"):
                    res = run_backtest(tick)
                    if isinstance(res, str): st.error(res)
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Strategy Return", f"{res['return']:.1f}%")
                        c2.metric("Buy & Hold", f"{res['bh_return']:.1f}%")
                        c3.metric("Trades", res['trades'])
                        st.line_chart(res['history']['Close'])
            
            with t2:
                if st.button("🔮 Draai Simulatie"):
                    sims = run_monte_carlo(tick)
                    if sims is not None:
                        f_mc = go.Figure()
                        for i in range(50): f_mc.add_trace(go.Scatter(y=sims[i], mode='lines', line=dict(width=1, color='rgba(0,255,255,0.1)'), showlegend=False))
                        avg = np.mean(sims, axis=0)
                        f_mc.add_trace(go.Scatter(y=avg, mode='lines', line=dict(width=3, color='yellow'), name='Verwachting'))
                        f_mc.update_layout(template="plotly_dark", title="1 Jaar Toekomst (1000 Scenario's)")
                        st.plotly_chart(f_mc, use_container_width=True)
            
            with t3:
                for n in news:
                    clr = "green" if n['sentiment'] == 'POSITIVE' else "red" if n['sentiment'] == 'NEGATIVE' else "gray"
                    st.markdown(f":{clr}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")
        else: st.error("Fout bij laden data.")

elif page == "💼 Mijn Portfolio":
    st.title("💼 Portfolio Manager")
    with st.expander("➕ Toevoegen", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1: t = st.text_input("Ticker", key="pt").upper()
        with c2: a = st.number_input("Aantal", 0.0, step=1.0)
        with c3: p = st.number_input("Prijs", 0.0)
        with c4: 
            if st.button("Add"): 
                st.session_state['portfolio'].append({"Ticker": t, "Aantal": a, "Koopprijs": p})
                st.success("Added!"); st.rerun()
    
    if st.session_state['portfolio']:
        p_data = []
        tot_v = 0; tot_c = 0
        tickers = []
        for i in st.session_state['portfolio']:
            cur = get_current_price(i['Ticker'])
            val = cur * i['Aantal']; cost = i['Koopprijs'] * i['Aantal']
            tot_v += val; tot_c += cost
            prof = val - cost; pct = (prof/cost)*100 if cost>0 else 0
            clr = "green" if prof>=0 else "red"
            tickers.append(i['Ticker'])
            p_data.append({"Ticker": i['Ticker'], "Aantal": i['Aantal'], "Waarde": f"{curr_sym}{val:.2f}", "Winst": f":{clr}[{curr_sym}{prof:.2f} ({pct:.1f}%)]"})
        
        st.write(pd.DataFrame(p_data).to_markdown(index=False), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Waarde", f"{curr_sym}{tot_v:.2f}")
        m2.metric("Inleg", f"{curr_sym}{tot_c:.2f}")
        m3.metric("Winst", f"{curr_sym}{tot_v-tot_c:.2f}", f"{((tot_v-tot_c)/tot_c)*100 if tot_c>0 else 0:.1f}%")
        
        if st.button("Optimaliseer Mix"):
            w = optimize_portfolio(tickers)
            if isinstance(w, str): st.error(w)
            elif w: 
                df_w = pd.DataFrame(list(w.items()), columns=['Ticker', 'Ideaal'])
                st.bar_chart(df_w.set_index('Ticker'))
            else: st.warning("Minimaal 2 tickers nodig.")
        
        if st.button("Wissen"): st.session_state['portfolio'] = []; st.rerun()

elif page == "📡 Deep Scanner":
    st.title("Scanner")
    pre = st.selectbox("Markt", list(PRESETS.keys()))
    txt = st.text_area("Tickers", PRESETS[pre])
    if st.button("Scan"):
        lst = [x.strip().upper() for x in txt.split(',')]
        res = []
        bar = st.progress(0)
        for i, t in enumerate(lst):
            bar.progress((i)/len(lst))
            try:
                df, met, _, ws, _, snip, _ = get_zenith_data(t)
                if df is not None:
                    sc = 0
                    if met['price']>met['sma200']: sc+=20
                    if met['rsi']<30: sc+=15
                    if ws['upside']>10: sc+=15
                    adv = "KOPEN" if sc>=70 else "HOUDEN" if sc>=50 else "AFBLIJVEN"
                    res.append({"Ticker": t, "Prijs": met['price'], "Score": sc, "Advies": adv})
            except: continue
        bar.empty()
        st.session_state['res'] = res
    
    if 'res' in st.session_state:
        df = pd.DataFrame(st.session_state['res']).sort_values('Score', ascending=False)
        st.dataframe(df, use_container_width=True)
        sel = st.selectbox("Kies:", df['Ticker'])
        st.button("Analyseer", on_click=start_analysis_for, args=(sel,))
