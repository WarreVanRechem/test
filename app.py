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

# Try-except voor scipy
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal v26.0 Elite", layout="wide", page_icon="💎")
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

def start_analysis_for(ticker):
    st.session_state['selected_ticker'] = ticker
    st.session_state['nav_page'] = "🔎 Markt Analyse"
    st.session_state['analysis_active'] = True

def reset_analysis():
    st.session_state['analysis_active'] = False

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
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]
    return atr * multiplier

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
            hist = obj.history(period="2d")
            if len(hist) >= 2:
                p = hist['Close'].iloc[-1]; prev = hist['Close'].iloc[-2]
                data[name] = (p, ((p-prev)/prev)*100)
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
        
        peers = get_smart_peers(ticker, i)
        met = {"name": i.get('longName', ticker), "price": cur, "sma200": df['SMA200'].iloc[-1], "rsi": df['RSI'].iloc[-1]}
        return df, met, fund, ws, None, snip, None, None, peers
    except Exception as e: return None, None, None, None, None, None, None, str(e), None

# --- UI ---
st.sidebar.header("Navigatie")
page = st.sidebar.radio("Ga naar:", ["🔎 Markt Analyse", "💼 Mijn Portfolio", "📡 Deep Scanner", "🎓 Leer de Basics"], key="nav_page")
curr_sym = "$" if "USD" in st.sidebar.radio("Valuta", ["USD", "EUR"]) else "€"

st.title("💎 Zenith Institutional Terminal")
mac = get_macro_data()
cols = st.columns(5)
for i, m in enumerate(["S&P 500", "Nasdaq", "Goud", "Olie", "10Y Rente"]):
    v, ch = mac.get(m, (0,0))
    cols[i].metric(m, f"{v:.2f}", f"{ch:.2f}%")
st.markdown("---")

if page == "🔎 Markt Analyse":
    c1, c2 = st.columns(2)
    with c1: tick = st.text_input("Ticker", value=st.session_state['selected_ticker'], on_change=reset_analysis).upper()
    with c2: cap = st.number_input(f"Kapitaal ({curr_sym})", 10000)
    
    if st.button("Start Deep Analysis"): st.session_state['analysis_active'] = True; st.session_state['selected_ticker'] = tick
    
    if st.session_state['analysis_active']:
        df, met, fund, ws, _, snip, _, err, peers = get_zenith_data(st.session_state['selected_ticker'])
        if err: st.error(f"⚠️ {err}")
        elif df is not None:
            st.markdown(f"## 🏢 {met['name']} ({tick})")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Prijs", f"{curr_sym}{met['price']:.2f}")
            k2.metric("RSI (14)", f"{met['rsi']:.1f}")
            if fund['fair_value']:
                diff = ((fund['fair_value']-met['price'])/met['price'])*100
                k3.metric("Fair Value", f"{curr_sym}{fund['fair_value']:.2f}", f"{diff:.1f}%")
            k4.metric("Upside (Target)", f"{ws['upside']:.1f}%")

            st.subheader("🎯 Sniper Setup & Financiële Gezondheid")
            s1, s2, s3 = st.columns(3)
            s1.metric("Entry (Ideaal)", f"{curr_sym}{snip['entry_price']:.2f}")
            s2.metric("Stop Loss (ATR)", f"{curr_sym}{snip['stop_loss']:.2f}")
            s3.metric("Risk/Reward", f"1 : {snip['rr_ratio']:.1f}")

            fin_df = get_financial_trends(tick)
            if fin_df is not None:
                f_fig = px.bar(fin_df, barmode='group', template="plotly_dark", title="Omzet & Winst Trend (Jaarlijks)")
                st.plotly_chart(f_fig, use_container_width=True)

            st.subheader("📈 Technical Chart")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Prijs"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='gold'), name="200MA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

elif page == "💼 Mijn Portfolio":
    st.title("💼 Portfolio & Risk Monitor")
    if st.session_state['portfolio']:
        tickers = [i['Ticker'] for i in st.session_state['portfolio']]
        if len(tickers) > 1:
            st.subheader("🔗 Correlatie Matrix")
            corr_data = yf.download(tickers, period="1y")['Close'].pct_change().corr()
            st.plotly_chart(px.imshow(corr_data, text_auto=True, template="plotly_dark"), use_container_width=True)
    else:
        st.info("Voeg aandelen toe om je risico te analyseren.")

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
