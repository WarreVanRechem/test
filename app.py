import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from transformers import pipeline
import feedparser
import warnings
import requests
import time

# Try-except voor scipy voor portfolio optimalisatie
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal v26.1 Full", layout="wide", page_icon="💎")
warnings.filterwarnings("ignore")

# --- DISCLAIMER & CREDITS ---
st.sidebar.error("⚠️ **DISCLAIMER:** Geen financieel advies. Educatief gebruik.")
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")

# --- SESSION STATE ---
if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
if 'nav_page' not in st.session_state: st.session_state['nav_page'] = "🔎 Markt Analyse"
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "RDW"
if 'analysis_active' not in st.session_state: st.session_state['analysis_active'] = False

# --- NAVIGATIE FUNCTIES ---
def start_analysis_for(ticker):
    st.session_state['selected_ticker'] = ticker
    st.session_state['nav_page'] = "🔎 Markt Analyse"
    st.session_state['analysis_active'] = True

def reset_analysis():
    st.session_state['analysis_active'] = False

# --- AI MODEL LADEN ---
@st.cache_resource
def load_ai():
    try: return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except: return None

ai_pipe = load_ai()

# --- PRESETS VOOR SCANNER ---
PRESETS = {
    "🇺🇸 Big Tech & AI": "NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, AMD, PLTR",
    "🇪🇺 AEX & Bel20": "ASML.AS, ADYEN.AS, BESI.AS, SHELL.AS, KBC.BR, UCB.BR, SOLB.BR, ABI.BR, INGA.AS",
    "🚀 High Growth": "COIN, MSTR, SMCI, HOOD, PLTR, SOFI, RIVN",
    "🛡️ Defensive": "KO, JNJ, PEP, MCD, O, V, BRK-B"
}

# --- ANALYSE FUNCTIES ---

def get_financial_trends(ticker):
    """Haalt omzet en winst trends op voor de laatste 4 jaar."""
    try:
        s = yf.Ticker(ticker)
        f = s.financials.T
        if f.empty: return None
        cols = [c for c in ['Total Revenue', 'Net Income'] if c in f.columns]
        df = f[cols].dropna()
        df.index = df.index.year
        return df.sort_index()
    except: return None

def calculate_atr_stop(df, multiplier=2):
    """Berekent een Stop Loss op basis van de beweeglijkheid (ATR)."""
    try:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        return atr * multiplier
    except: return 0.0

def get_smart_peers(ticker, info):
    if not info: return ["^GSPC"]
    sector = info.get('sector', '').lower(); industry = info.get('industry', '').lower()
    peers = []
    if 'semicon' in industry: peers = ["NVDA", "AMD", "INTC", "TSM", "ASML"]
    elif 'software' in industry or 'technology' in sector: peers = ["MSFT", "AAPL", "GOOGL", "ORCL", "ADBE"]
    elif 'bank' in industry: peers = ["JPM", "BAC", "C", "WFC", "HSBC"]
    elif 'oil' in industry or 'energy' in sector: peers = ["XOM", "CVX", "SHEL", "TTE", "BP"]
    elif 'auto' in industry: peers = ["TSLA", "TM", "F", "GM", "STLA"]
    elif 'drug' in industry or 'healthcare' in sector: peers = ["LLY", "JNJ", "PFE", "MRK", "NVS"]
    if not peers:
        if 'tech' in sector: peers = ["XLK"]
        elif 'health' in sector: peers = ["XLV"]
        elif 'financ' in sector: peers = ["XLF"]
        elif 'energy' in sector: peers = ["XLE"]
        else: peers = ["^GSPC"]
    peers = [p for p in peers if p.upper() != ticker.upper() and p.split('.')[0] != ticker.split('.')[0]]
    return peers[:4]

def compare_peers(main_ticker, peers_list):
    try:
        df = yf.download([main_ticker] + peers_list, period="1y")['Close']
        if df.empty: return None
        return df.apply(lambda x: ((x / x.iloc[0]) - 1) * 100)
    except: return None

def calculate_graham_number(info):
    try:
        eps = info.get('trailingEps'); bvps = info.get('bookValue')
        if eps is None or bvps is None or eps <= 0 or bvps <= 0: return None
        return np.sqrt(22.5 * eps * bvps)
    except: return None

def run_backtest(ticker, period="5y"):
    try:
        stock = yf.Ticker(ticker); df = stock.history(period=period)
        if df.empty or len(df) < 250: return "Te weinig data."
        df['SMA200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff(); gain = (delta.where(delta>0,0)).rolling(14).mean(); loss = (-delta.where(delta<0,0)).rolling(14).mean()
        df['RSI'] = 100-(100/(1+(gain/loss)))
        balance=10000; shares=0; trades=[]; in_pos=False
        for i in range(201, len(df)):
            p = df['Close'].iloc[i]; rsi = df['RSI'].iloc[i]; sma = df['SMA200'].iloc[i]
            if pd.isna(sma): continue
            if not in_pos and p > sma and rsi < 35:
                shares = balance/p; balance=0; in_pos=True; trades.append({"Date":df.index[i],"Type":"BUY","Price":p})
            elif in_pos and (rsi > 75 or p < sma*0.95):
                balance = shares*p; shares=0; in_pos=False; trades.append({"Date":df.index[i],"Type":"SELL","Price":p})
        final = balance if not in_pos else shares*df['Close'].iloc[-1]
        return {"return": ((final-10000)/10000)*100, "bh_return": ((df['Close'].iloc[-1]-df['Close'].iloc[201])/df['Close'].iloc[201])*100, "trades": len(trades), "final_value": final, "history": df}
    except: return "Backtest Error"

def run_monte_carlo(ticker):
    try:
        d = yf.Ticker(ticker).history(period="1y")['Close']
        ret = d.pct_change().dropna()
        sims = []; mu=ret.mean(); sig=ret.std()
        for _ in range(200):
            p = [d.iloc[-1]]
            for _ in range(252): p.append(p[-1]*(1+np.random.normal(mu,sig)))
            sims.append(p)
        return np.array(sims)
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

def get_current_price(ticker):
    try:
        obj = yf.Ticker(ticker)
        p = obj.fast_info.last_price
        if not pd.isna(p) and p > 0: return p
        h = obj.history(period="1d")
        if not h.empty: return h['Close'].iloc[-1]
    except: pass
    return 0.0

@st.cache_data(ttl=600)
def get_macro_data():
    tickers = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Goud": "GC=F", "Olie": "CL=F", "10Y Rente": "^TNX"}
    data = {}
    for name, t in tickers.items():
        try:
            obj = yf.Ticker(t)
            try:
                p = obj.fast_info.last_price
                prev = obj.fast_info.previous_close
                if not p or not prev: raise ValueError("Geen fast data")
            except:
                hist = obj.history(period="2d")
                if len(hist) >= 2:
                    p = hist['Close'].iloc[-1]; prev = hist['Close'].iloc[-2]
                else: p = 0; prev = 0
            
            if p and prev: data[name] = (p, ((p-prev)/prev)*100)
            else: data[name] = (0,0)
        except: data[name] = (0,0)
    return data

@st.cache_data(ttl=3600)
def get_zenith_data(ticker):
    try:
        s = yf.Ticker(ticker); df = s.history(period="7y"); i = s.info
        if df.empty: return None, None, None, None, None, None, None, "Geen data", None
        cur = df['Close'].iloc[-1]
        
        fair_value = calculate_graham_number(i)
        d_rate = i.get('dividendRate') or i.get('trailingAnnualDividendRate')
        d_yield = (d_rate/cur)*100 if (d_rate and cur>0) else (i.get('dividendYield',0)*100)
        fund = {"pe": i.get('trailingPE',0), "div": d_yield, "sec": i.get('sector','-'), "prof": (i.get('profitMargins')or 0)*100, "fair_value": fair_value}
        
        ws = {"target": i.get('targetMeanPrice',0) or 0, "rec": i.get('recommendationKey','none').upper()}
        ws["upside"] = ((ws["target"]-cur)/cur)*100 if ws["target"] else 0
        
        df['SMA200'] = df['Close'].rolling(200).mean(); df['SMA20'] = df['Close'].rolling(20).mean()
        df['std'] = df['Close'].rolling(20).std(); df['U'] = df['SMA20']+(df['std']*2); df['L'] = df['SMA20']-(df['std']*2)
        delta = df['Close'].diff(); rs = (delta.where(delta>0,0).rolling(14).mean()) / (-delta.where(delta<0,0).rolling(14).mean())
        df['RSI'] = 100-(100/(1+rs))
        
        # NIEUW: ATR Based Stop Loss
        atr_val = calculate_atr_stop(df)
        ent = df['L'].iloc[-1] if not pd.isna(df['L'].iloc[-1]) else cur
        high = df['High'].tail(50).max()
        
        snip = {
            "entry_price": ent, 
            "current_diff": ((cur-ent)/cur)*100, 
            "stop_loss": cur - atr_val,
            "take_profit": high, 
            "rr_ratio": (high-ent)/(atr_val) if atr_val>0 else 0 
        }
        
        try: 
            m = yf.Ticker("^GSPC").history(period="7y")
            ma = m['Close'].reindex(df.index, method='nearest')
            df['M'] = (ma/ma.iloc[0])*df['Close'].iloc[0]; mb = m['Close'].iloc[-1]>m['Close'].rolling(200).mean().iloc[-1]
        except: df['M']=df['Close']; mb=True
        
        peers = get_smart_peers(ticker, i)
        met = {"name": i.get('longName', ticker), "price": cur, "sma200": df['SMA200'].iloc[-1], "rsi": df['RSI'].iloc[-1], "bull": mb}
        return df, met, fund, ws, None, snip, None, None, peers
    except Exception as e: return None, None, None, None, None, None, None, str(e), None

def get_external_info(ticker):
    try:
        s = yf.Ticker(ticker); b = 0
        ins = s.insider_transactions
        if ins is not None and not ins.empty: b = ins.head(10)[ins.head(10)['Text'].str.contains("Purchase",case=False,na=False)].shape[0]
        f = feedparser.parse(requests.get(f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en", headers={'User-Agent': 'Mozilla/5.0'}, timeout=3).content)
        n = [{"title":e.title, "sentiment": ai_pipe(e.title[:512])[0]['label'].upper() if ai_pipe else "NEUTRAL", "link":e.link} for e in f.entries[:5]]
        return b, n
    except: return 0, []

def generate_thesis(met, snip, ws, buys, fund):
    th = []; sig = "NEUTRAAL"
    upt = met['price']>met['sma200']; zone = snip['current_diff']<1.5 
    val_txt = ""
    if fund['fair_value']:
        if met['price'] < fund['fair_value'] * 0.8: val_txt = "💎 **VALUE:** Aandeel is goedkoop."
        elif met['price'] > fund['fair_value'] * 1.2: val_txt = "⚠️ **WAARDE:** Aandeel is duur."
    
    if upt and zone: th.append(f"🔥 **PERFECT:** Trend Bullish + Buy Zone. {val_txt}"); sig="STERK KOPEN"
    elif not upt and zone: th.append(f"⚠️ **RISICO:** Trend Bearish + Buy Zone. {val_txt}"); sig="SPECULATIEF"
    elif upt and not zone: th.append(f"✅ **HOUDEN:** Wacht op dip. {val_txt}"); sig="HOUDEN"
    else: th.append("🛑 **AFBLIJVEN.**"); sig="AFBLIJVEN"
    if ws['upside']>15: th.append(f"Analisten: {ws['upside']:.0f}% upside.")
    return " ".join(th), sig

# --- UI START ---
st.sidebar.header("Navigatie")
page = st.sidebar.radio("Ga naar:", ["🔎 Markt Analyse", "💼 Mijn Portfolio", "📡 Deep Scanner", "🎓 Leer de Basics"], key="nav_page")

with st.sidebar.expander("🧮 Calculator"):
    acc=st.number_input("Acc",10000); risk=st.slider("Risico",0.5,5.0,1.0); ent=st.number_input("In",100.0); stp=st.number_input("Stop",95.0)
    if stp<ent: st.write(f"**Koop:** {int((acc*(risk/100))/(ent-stp))} stuks")
curr_sym = "$" if "USD" in st.sidebar.radio("Valuta", ["USD", "EUR"]) else "€"

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")

st.title("💎 Zenith Institutional Terminal")
mac = get_macro_data()
cols = st.columns(5)
for i, m in enumerate(["S&P 500", "Nasdaq", "Goud", "Olie", "10Y Rente"]):
    v, ch = mac.get(m, (0,0))
    cols[i].metric(m, f"{v:.2f}", f"{ch:.2f}%")
st.markdown("---")

# --- PAGINA 1: MARK ANALYSE ---
if page == "🔎 Markt Analyse":
    c1, c2 = st.columns(2)
    with c1: tick = st.text_input("Ticker", value=st.session_state['selected_ticker'], on_change=reset_analysis).upper()
    with c2: cap = st.number_input(f"Kapitaal ({curr_sym})", 10000)
    
    if st.button("Start Deep Analysis"): st.session_state['analysis_active'] = True; st.session_state['selected_ticker'] = tick
    
    if st.session_state['analysis_active']:
        df, met, fund, ws, _, snip, _, err, peers = get_zenith_data(st.session_state['selected_ticker'])
        
        if err: st.error(f"⚠️ {err}")
        elif df is not None:
            with st.spinner('Analyseren...'): buys, news = get_external_info(tick)
            
            score = 50 
            if met['price']>met['sma200']: score+=20
            if met['rsi']<35: score+=15
            if fund['fair_value'] and met['price'] < fund['fair_value']: score += 15
            
            thesis, sig = generate_thesis(met, snip, ws, buys, fund)
            
            st.markdown(f"## 🏢 {met['name']} ({tick})")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Score", f"{score}/100")
            clr = "green" if "KOPEN" in sig else "orange" if "SPEC" in sig else "red" if "AFBL" in sig else "blue"
            k2.markdown(f"**Advies:** :{clr}[{sig}]")
            k3.metric("Prijs", f"{curr_sym}{met['price']:.2f}")
            if fund['fair_value']:
                diff_fair = ((fund['fair_value'] - met['price']) / met['price']) * 100
                k4.metric("Fair Value", f"{curr_sym}{fund['fair_value']:.2f}", f"{diff_fair:.1f}%")
            else: k4.metric("Fair Value", "N/A", "Verlieslatend")

            st.info(f"**Zenith Thesis:** {thesis}")
            
            st.markdown("---")
            # NIEUWE SNIPER LAYOUT MET ATR
            st.subheader("🎯 Sniper Entry Setup (ATR Volatiliteit)")
            s1, s2, s3, s4 = st.columns(4)
            msg = "✅ NU KOPEN!" if snip['current_diff'] < 1.5 else f"Wacht (-{snip['current_diff']:.1f}%)"
            s1.metric("1. Entry (Ideaal)", f"{curr_sym}{snip['entry_price']:.2f}", msg)
            s2.metric("2. Stop Loss (ATR)", f"{curr_sym}{snip['stop_loss']:.2f}")
            s3.metric("3. Take Profit", f"{curr_sym}{snip['take_profit']:.2f}")
            rr_c = "green" if snip['rr_ratio']>=2 else "orange"
            s4.markdown(f"**4. Risk/Reward:** :{rr_c}[1 : {snip['rr_ratio']:.1f}]")
            
            # NIEUWE FUNDAMENTELE CHART
            st.subheader("📊 Fundamentele Trends (4 Jaar)")
            fin_df = get_financial_trends(tick)
            if fin_df is not None:
                f_fig = px.bar(fin_df, barmode='group', template="plotly_dark", color_discrete_sequence=['#636EFA', '#00CC96'])
                st.plotly_chart(f_fig, use_container_width=True)
            else: st.warning("Geen fundamentele data beschikbaar.")

            st.subheader("📈 Technical Chart")
            end = df.index[-1]; start = end - pd.DateOffset(years=1); plot_df = df.loc[start:end]
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.2,0.2])
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['L'], line=dict(color='rgba(0,255,0,0.3)'), name="Lower Band"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['U'], line=dict(color='rgba(255,0,0,0.3)'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name="Upper Band"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700'), name="200MA"), row=1, col=1)
            if 'M' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['M'], line=dict(color='white', width=1, dash='dot'), name="S&P500"), row=1, col=1)
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Prijs"), row=1, col=1)
            clrs = ['green' if r['Open']<r['Close'] else 'red' for i,r in plot_df.iterrows()]
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=clrs, name="Vol"), row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#9370DB'), name="RSI"), row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1); fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)
            
            t1, t2, t3, t4 = st.tabs(["⚔️ Peer Battle", "🔙 Backtest", "🔮 Monte Carlo", "📰 Nieuws"])
            with t1:
                st.subheader("Competitie Check")
                if peers:
                    st.write(f"Automatisch vergeleken met: {', '.join(peers)}")
                    if st.button("Laad Vergelijking"):
                        pd_data = compare_peers(tick, peers)
                        if pd_data is not None: st.line_chart(pd_data)
                        else: st.error("Geen data.")
                else: st.warning("Geen concurrenten gevonden.")
            with t2:
                if st.button("🚀 Draai Backtest"):
                    res = run_backtest(tick)
                    if isinstance(res, str): st.error(res)
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Strategy", f"{res['return']:.1f}%"); c2.metric("Buy&Hold", f"{res['bh_return']:.1f}%"); c3.metric("Trades", res['trades'])
                        st.line_chart(res['history']['Close'])
            with t3:
                if st.button("🔮 Simulatie"):
                    sims = run_monte_carlo(tick)
                    if sims is not None:
                        f = go.Figure()
                        for i in range(50): f.add_trace(go.Scatter(y=sims[i], mode='lines', line=dict(width=1, color='rgba(0,255,255,0.1)'), showlegend=False))
                        f.add_trace(go.Scatter(y=np.mean(sims,axis=0), mode='lines', line=dict(width=3, color='yellow'), name='Avg'))
                        f.update_layout(template="plotly_dark"); st.plotly_chart(f, use_container_width=True)
            with t4:
                for n in news:
                    c = "green" if n['sentiment']=="POSITIVE" else "red" if n['sentiment']=="NEGATIVE" else "gray"
                    st.markdown(f":{c}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")

# --- PAGINA 2: PORTFOLIO MANAGER ---
elif page == "💼 Mijn Portfolio":
    st.title("💼 Portfolio Manager & Risk Guard")
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
        
        # --- NIEUW: CORRELATIE MATRIX ---
        if len(tickers) > 1:
            st.subheader("🔗 Correlatie Matrix (Risk Check)")
            try:
                corr_data = yf.download(tickers, period="1y")['Close'].pct_change().corr()
                fig_corr = px.imshow(corr_data, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)
                st.caption("1.0 = Beweegt identiek. < 0.5 = Goede spreiding. -1.0 = Beweegt tegenovergesteld.")
            except: st.warning("Kon correlatie matrix niet laden.")

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
    else: st.write("Leeg.")

# --- PAGINA 3: DEEP SCANNER ---
elif page == "📡 Deep Scanner":
    st.title("Scanner")
    pre = st.selectbox("Markt", list(PRESETS.keys()))
    txt = st.text_area("Tickers", PRESETS[pre])
    if st.button("Scan"):
        lst = [x.strip().upper() for x in txt.split(',')]
        res = []
        bar = st.progress(0)
        failed = [] 
        for i, t in enumerate(lst):
            bar.progress((i)/len(lst))
            time.sleep(0.2) 
            try:
                df, met, fund, ws, _, snip, _, _, _ = get_zenith_data(t)
                if df is not None:
                    sc = 0; reasons = []
                    if met['price'] > met['sma200']: sc += 20; reasons.append("Trend 📈")
                    if met['rsi'] < 30: sc += 15; reasons.append("Oversold 📉")
                    if ws['upside'] > 15: sc += 15; reasons.append("Analisten 💼")
                    if fund['fair_value'] and met['price'] < fund['fair_value']: sc += 15; reasons.append("Value 💎")
                    if snip['rr_ratio'] > 2: sc += 15; reasons.append("Sniper 🎯")
                    if fund['pe'] > 0 and fund['pe'] < 20: sc += 10; reasons.append("Goedkoop 💰")

                    adv = "KOPEN" if sc>=70 else "HOUDEN" if sc>=50 else "AFBLIJVEN"
                    
                    res.append({
                        "Ticker": t, 
                        "Prijs": met['price'], 
                        "Analist Doel": f"{curr_sym}{ws['target']:.2f}",
                        "Upside": f"{ws['upside']:.1f}%",
                        "Score": sc, 
                        "Advies": adv,
                        "Reden": " + ".join(reasons) if reasons else "-"
                    })
                else: failed.append(f"{t}: Geen data")
            except Exception as e:
                failed.append(f"{t}: {str(e)}")
        
        bar.empty()
        st.session_state['res'] = res
        st.session_state['failed'] = failed
    
    if 'res' in st.session_state:
        if not st.session_state['res']:
            st.warning("Geen resultaten gevonden.")
        else:
            df = pd.DataFrame(st.session_state['res']).sort_values('Score', ascending=False)
            st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100),
                    "Prijs": st.column_config.NumberColumn("Prijs", format=f"{curr_sym}%.2f")
                }
            )
            st.download_button("📥 Download (CSV)", df.to_csv(index=False), "results.csv", "text/csv")
            
            st.markdown("---")
            c1, c2 = st.columns([3, 1])
            options = [r['Ticker'] for r in st.session_state['res']]
            if options:
                sel = c1.selectbox("Kies:", options)
                c2.button("🚀 Analyseer Nu", on_click=start_analysis_for, args=(sel,))
        
        if 'failed' in st.session_state and st.session_state['failed']:
            with st.expander("⚠️ Foutrapportage"): st.write(st.session_state['failed'])

# --- PAGINA 4: EDUCATE & BASICS ---
elif page == "🎓 Leer de Basics":
    st.title("🎓 Zenith Academy: Beleggen voor Beginners")
    st.markdown("### Begrijp de data achter je beslissingen")

    with st.expander("💎 1. De 'Eerlijke Prijs' (Graham Number)"):
        st.write("""
        **Wat is het?** Een berekening die kijkt naar de winst en bezittingen om de 'echte' waarde van een aandeel te bepalen.
        * **De Metafoor:** Zie het als de taxatiewaarde van een huis. Als de vraagprijs lager is dan de taxatie, heb je een goede deal.
        * **Zenith Tip:** Wij zoeken aandelen waar de huidige prijs onder de Fair Value ligt.
        """)

    with st.expander("🌡️ 2. De Thermometer (RSI)"):
        st.write("""
        **Wat is het?** De Relative Strength Index (RSI) meet of een aandeel te snel gestegen of gedaald is.
        * **Onder 30:** Het aandeel is 'onderkoeld' (Oversold). Vaak een koopkans.
        * **Boven 70:** Het aandeel is 'oververhit' (Overbought). De kans op een daling is groot.
        """)

    with st.expander("📈 3. De Lange Termijn Trend (SMA 200)"):
        st.write("""
        **Wat is het?** Het gemiddelde van de prijs over de laatste 200 dagen.
        * **Boven de lijn:** De trend is positief (Bullish).
        * **Onder de lijn:** De trend is negatief (Bearish). Professionals kopen meestal alleen als de prijs boven deze lijn zit.
        """)

    with st.expander("🎯 4. De Sniper & ATR Stop Loss"):
        st.write("""
        **Wat is ATR?** De Average True Range meet hoe 'wild' een aandeel beweegt.
        * **Waarom ATR?** Een stabiel aandeel (als Coca-Cola) heeft een krappe stop loss nodig. Een wild aandeel (als Tesla) heeft ruimte nodig om te ademen zonder dat je direct wordt 'uitgeschud'.
        * **Risk/Reward:** Wij mikken op 1:2. Dat betekent dat we bereid zijn €1 te riskeren om €2 te verdienen.
        """)

    with st.expander("🔗 5. Correlatie Matrix (Je Geheime Wapen)"):
        st.write("""
        **Wat is het?** Het laat zien of je aandelen 'vriendjes' zijn.
        * **Correlatie 1.0:** Als aandeel A stijgt, stijgt B ook. Dit is gevaarlijk (geen spreiding).
        * **Correlatie < 0.5:** Je aandelen reageren anders op de markt. Dit is veiliger.
        * **Voorbeeld:** Als je alleen maar AI-aandelen koopt, is je correlatie vaak 0.9. Als de tech-sector valt, valt je hele portfolio.
        """)

    with st.expander("🔮 6. Monte Carlo & Backtesting"):
        st.write("""
        **Backtest:** Een simulatie van "Wat als ik dit in het verleden had gedaan?". Het geeft geen garantie, maar laat zien of een strategie statistisch werkt.
        **Monte Carlo:** Een computer die 200 keer 'met de dobbelstenen gooit' om te zien waar de prijs over een jaar kan eindigen op basis van huidige grilligheid.
        """)

    st.success("💡 **Gouden regel:** Een goede belegger kijkt niet naar de prijs van vandaag, maar naar de trend van morgen en het risico van gisteren.")
