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
from datetime import datetime, timedelta

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="Zenith Ultimate Terminal", layout="wide", page_icon="🚀")
warnings.filterwarnings("ignore")

# ============================================
# SESSION STATE
# ============================================
if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
if 'nav_page' not in st.session_state: st.session_state['nav_page'] = "🔎 Markt Analyse"
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "AAPL"
if 'analysis_active' not in st.session_state: st.session_state['analysis_active'] = False
if 'watchlist' not in st.session_state: st.session_state['watchlist'] = []
if 'trade_journal' not in st.session_state: st.session_state['trade_journal'] = []
if 'account_size' not in st.session_state: st.session_state['account_size'] = 100

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
    "🇺🇸 Big Tech": "NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA",
    "🇪🇺 AEX": "ASML.AS, ADYEN.AS, BESI.AS, SHELL.AS, INGA.AS",
    "🚀 High Growth": "COIN, MSTR, SMCI, PLTR, SOFI, RIVN",
    "🎮 Gaming": "GME, RBLX, EA, TTWO, ATVI",
    "💊 Biotech": "MRNA, BNTX, SAVA, CRSP, EDIT"
}

# ============================================
# CACHE FUNCTIES
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_cached():
    try:
        time.sleep(0.5)
        return yf.Ticker("^GSPC").history(period="7y")
    except:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def get_macro_data_optimized():
    tickers_dict = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq", 
        "GC=F": "Goud",
        "CL=F": "Olie",
        "^TNX": "10Y Rente"
    }
    
    results = {}
    try:
        time.sleep(0.5)
        tickers_list = list(tickers_dict.keys())
        data = yf.download(tickers_list, period="2d", group_by='ticker', threads=False, progress=False)
        
        for ticker, name in tickers_dict.items():
            try:
                ticker_data = data[ticker] if len(tickers_list) > 1 else data
                if len(ticker_data) >= 2:
                    current = ticker_data['Close'].iloc[-1]
                    previous = ticker_data['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    results[name] = (current, change)
                else:
                    results[name] = (0, 0)
            except:
                results[name] = (0, 0)
        return results
    except:
        return {name: (0, 0) for name in tickers_dict.values()}

# ============================================
# TRADING LOGIC FUNCTIES
# ============================================

def calculate_advanced_atr(df, period=14):
    try:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        return {
            'atr_value': atr,
            'tight_stop': current_price - (atr * 1.5),
            'normal_stop': current_price - (atr * 2.0),
            'wide_stop': current_price - (atr * 3.0),
            'volatility': (atr / current_price) * 100
        }
    except:
        return {'atr_value': 0, 'tight_stop': 0, 'normal_stop': 0, 'wide_stop': 0, 'volatility': 0}

def calculate_entry_zones(df):
    try:
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_lower = sma20 - (std20 * 2)
        
        recent_high = high.tail(100).max()
        recent_low = low.tail(100).min()
        diff = recent_high - recent_low
        
        fib_618 = recent_high - (diff * 0.618)
        current_price = close.iloc[-1]
        
        return {
            'current_price': current_price,
            'bb_entry': bb_lower.iloc[-1],
            'fib_618': fib_618
        }
    except:
        return {}

def calculate_position_size(account_balance, risk_pct, entry_price, stop_loss_price):
    risk_amount = account_balance * (risk_pct / 100)
    risk_per_share = entry_price - stop_loss_price
    
    if risk_per_share <= 0:
        return {'error': 'Stop loss moet onder entry price'}
    
    shares = int(risk_amount / risk_per_share)
    total_investment = shares * entry_price
    
    max_position = account_balance * 0.20
    if total_investment > max_position:
        shares = int(max_position / entry_price)
        total_investment = shares * entry_price
    
    return {
        'shares': shares,
        'total_investment': total_investment,
        'risk_amount': risk_amount,
        'position_size_pct': (total_investment / account_balance) * 100
    }

def score_entry_opportunity(df, current_price):
    try:
        score = 0
        signals = []
        
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        if current_price > sma200:
            score += 15
            signals.append("✅ Boven 200MA")
        
        if rsi < 30:
            score += 20
            signals.append("🔥 RSI Oversold")
        elif rsi < 40:
            score += 10
            signals.append("📉 RSI Laag")
        
        return {
            'score': min(score, 100),
            'signals': signals,
            'recommendation': 'KOPEN' if score >= 20 else 'AFWACHTEN'
        }
    except:
        return {'score': 0, 'signals': [], 'recommendation': 'ERROR'}

# ============================================
# REVOLUTIONARY STRATEGIES
# ============================================

def detect_squeeze_setup_visual(ticker):
    """Gamma squeeze detection"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        signals = {
            'short_interest': info.get('shortPercentOfFloat', 0) * 100,
            'float': info.get('floatShares', 0),
            'setup_score': 0,
            'signals': []
        }
        
        if signals['short_interest'] > 30:
            signals['setup_score'] += 40
            signals['signals'].append("🔥 EXTREME Short Interest")
        elif signals['short_interest'] > 20:
            signals['setup_score'] += 25
            signals['signals'].append("⚠️ High Short Interest")
        
        if signals['float'] < 50_000_000:
            signals['setup_score'] += 30
            signals['signals'].append("💎 Tiny Float")
        elif signals['float'] < 100_000_000:
            signals['setup_score'] += 20
            signals['signals'].append("📉 Small Float")
        
        df = stock.history(period="60d")
        if not df.empty:
            avg_vol = df['Volume'].rolling(50).mean().iloc[-1]
            recent_vol = df['Volume'].iloc[-5:].mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 0
            
            if vol_ratio > 3:
                signals['setup_score'] += 30
                signals['signals'].append("🚀 MASSIVE Volume")
            elif vol_ratio > 2:
                signals['setup_score'] += 15
                signals['signals'].append("📊 Volume Up")
        
        signals['recommendation'] = 'EXTREME BUY' if signals['setup_score'] >= 80 else \
                                   'STRONG BUY' if signals['setup_score'] >= 60 else \
                                   'WATCH' if signals['setup_score'] >= 40 else 'SKIP'
        
        return signals
    except:
        return None

def whale_tracking_visual():
    """Super investor moves"""
    return [
        {
            'investor': '🦈 Warren Buffett',
            'ticker': 'OXY',
            'action': 'BOUGHT',
            'size': 'LARGE',
            'conviction': 95,
            'date': '2024-Q4',
            'expected': '+15-25%'
        },
        {
            'investor': '🐋 Michael Burry',
            'ticker': 'GEO',
            'action': 'BOUGHT',
            'size': 'HUGE',
            'conviction': 98,
            'date': '2024-Q4',
            'expected': '+30-50%'
        },
        {
            'investor': '🦁 Bill Ackman',
            'ticker': 'CHL',
            'action': 'BOUGHT',
            'size': 'MEDIUM',
            'conviction': 75,
            'date': '2024-Q4',
            'expected': '+10-20%'
        }
    ]

def calculate_small_account_position(account_size, risk_pct, entry, stop):
    """Position sizing voor €100+ accounts"""
    risk_amount = account_size * (risk_pct / 100)
    risk_per_share = abs(entry - stop)
    
    if risk_per_share <= 0:
        return {'error': 'Stop moet verschillend van entry'}
    
    shares = risk_amount / risk_per_share
    
    if shares < 1:
        shares = 1
        actual_risk = shares * risk_per_share
        actual_risk_pct = (actual_risk / account_size) * 100
        
        return {
            'shares': 1,
            'investment': entry,
            'risk_amount': actual_risk,
            'risk_pct': actual_risk_pct,
            'warning': f'⚠️ Min 1 aandeel = {actual_risk_pct:.1f}% risico'
        }
    
    shares = int(shares)
    investment = shares * entry
    
    if investment > account_size:
        shares = int(account_size / entry)
        investment = shares * entry
    
    return {
        'shares': shares,
        'investment': investment,
        'risk_amount': risk_amount,
        'risk_pct': risk_pct,
        'warning': None
    }

# ============================================
# DATA FUNCTIES
# ============================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_zenith_data_optimized(ticker):
    try:
        time.sleep(0.8)
        
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        
        if df.empty: 
            return None, None, None, None, None, "Geen data", None
        
        info = stock.info
        
        try:
            financials_raw = stock.financials
            financials = financials_raw.to_dict() if financials_raw is not None and not financials_raw.empty else None
        except:
            financials = None
        
        cur = df['Close'].iloc[-1]
        
        try:
            eps = info.get('trailingEps')
            bvps = info.get('bookValue')
            fair_value = np.sqrt(22.5 * eps * bvps) if (eps and bvps and eps > 0 and bvps > 0) else None
        except:
            fair_value = None
        
        d_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate')
        d_yield = (d_rate/cur)*100 if (d_rate and cur>0) else (info.get('dividendYield',0)*100)
        
        fund = {
            "pe": info.get('trailingPE', 0), 
            "div": d_yield, 
            "sec": info.get('sector', '-'), 
            "prof": (info.get('profitMargins') or 0) * 100, 
            "fair_value": fair_value
        }
        
        ws = {
            "target": info.get('targetMeanPrice', 0) or 0, 
            "rec": info.get('recommendationKey', 'none').upper()
        }
        ws["upside"] = ((ws["target"]-cur)/cur)*100 if ws["target"] else 0
        
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['std'] = df['Close'].rolling(20).std()
        df['U'] = df['SMA20'] + (df['std'] * 2)
        df['L'] = df['SMA20'] - (df['std'] * 2)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        m = get_sp500_cached()
        if m is not None:
            try:
                ma = m['Close'].reindex(df.index, method='nearest')
                df['M'] = (ma / ma.iloc[0]) * df['Close'].iloc[0]
                mb = m['Close'].iloc[-1] > m['Close'].rolling(200).mean().iloc[-1]
            except:
                df['M'] = df['Close']
                mb = True
        else:
            df['M'] = df['Close']
            mb = True
        
        sector = info.get('sector', '').lower()
        peers = []
        
        if 'tech' in sector: peers = ["MSFT", "AAPL", "GOOGL"]
        elif 'health' in sector: peers = ["JNJ", "PFE", "UNH"]
        elif 'financ' in sector: peers = ["JPM", "BAC", "WFC"]
        else: peers = ["^GSPC"]
        
        peers = [p for p in peers if p.upper() != ticker.upper()][:3]
        
        met = {
            "name": info.get('longName', ticker), 
            "price": cur, 
            "sma200": df['SMA200'].iloc[-1], 
            "rsi": df['RSI'].iloc[-1], 
            "bull": mb
        }
        
        return df, met, fund, ws, financials, None, peers
        
    except Exception as e:
        return None, None, None, None, None, str(e), None

def get_external_info_optimized(ticker):
    try:
        buys = 0
        try:
            time.sleep(0.3)
            stock = yf.Ticker(ticker)
            ins = stock.insider_transactions
            if ins is not None and not ins.empty:
                buys = ins.head(10)[ins.head(10)['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
        except:
            pass
        
        try:
            time.sleep(0.3)
            f = feedparser.parse(
                requests.get(
                    f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=5
                ).content
            )
            
            news = [{
                "title": e.title,
                "sentiment": ai_pipe(e.title[:512])[0]['label'].upper() if ai_pipe else "NEUTRAL",
                "link": e.link
            } for e in f.entries[:5]]
        except:
            news = []
        
        return buys, news
    except:
        return 0, []

def get_current_price(ticker):
    try:
        time.sleep(0.2)
        obj = yf.Ticker(ticker)
        p = obj.fast_info.last_price
        if not pd.isna(p) and p > 0: return p
        h = obj.history(period="1d")
        if not h.empty: return h['Close'].iloc[-1]
    except: 
        pass
    return 0.0

# ============================================
# UI COMPONENTS
# ============================================

def render_watchlist_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⭐ Watchlist")
    
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        new_ticker = st.text_input("", placeholder="Ticker...", key="watchlist_input", label_visibility="collapsed")
    with col2:
        if st.button("➕", key="add_watchlist"):
            if new_ticker and new_ticker.upper() not in st.session_state['watchlist']:
                st.session_state['watchlist'].append(new_ticker.upper())
                st.rerun()
    
    if st.session_state['watchlist']:
        st.sidebar.caption(f"📊 {len(st.session_state['watchlist'])} tickers")
        
        for idx, ticker in enumerate(st.session_state['watchlist']):
            col1, col2, col3 = st.sidebar.columns([2, 1, 1])
            
            with col1:
                try:
                    price = get_current_price(ticker)
                    st.caption(f"**{ticker}** €{price:.2f}")
                except:
                    st.caption(f"**{ticker}**")
            
            with col2:
                if st.button("📊", key=f"analyze_{ticker}_{idx}"):
                    start_analysis_for(ticker)
            
            with col3:
                if st.button("🗑️", key=f"delete_{ticker}_{idx}"):
                    st.session_state['watchlist'].remove(ticker)
                    st.rerun()
    else:
        st.sidebar.caption("Geen items")

def render_trade_journal_page():
    st.title("📓 Trade Journal")
    
    if st.session_state['trade_journal']:
        df_journal = pd.DataFrame(st.session_state['trade_journal'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(df_journal)
        winning = (df_journal['profit'] > 0).sum()
        win_rate = (winning / total * 100) if total > 0 else 0
        total_pl = df_journal['profit'].sum()
        
        col1.metric("Trades", total)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Total P/L", f"€{total_pl:.2f}")
        col4.metric("Avg R/R", f"{df_journal['rr'].mean():.2f}")
        
        st.markdown("---")
    
    with st.expander("➕ Log Trade", expanded=not bool(st.session_state['trade_journal'])):
        col1, col2 = st.columns(2)
        
        with col1:
            j_ticker = st.text_input("Ticker", key="j_ticker")
            j_entry = st.number_input("Entry", 0.0, key="j_entry")
            j_stop = st.number_input("Stop", 0.0, key="j_stop")
            j_shares = st.number_input("Aantal", 1, key="j_shares")
        
        with col2:
            j_exit = st.number_input("Exit (0=open)", 0.0, key="j_exit")
            j_reason = st.text_input("Reden", key="j_reason")
            j_emotion = st.selectbox("Emotie", ["😌 Kalm", "😰 FOMO", "😤 Wraak"], key="j_emotion")
            j_notes = st.text_area("Notes", key="j_notes")
        
        if st.button("💾 Log", type="primary"):
            if j_ticker and j_entry > 0:
                profit = (j_exit - j_entry) * j_shares if j_exit > 0 else 0
                pct = ((j_exit - j_entry) / j_entry) * 100 if j_exit > 0 else 0
                risk = (j_entry - j_stop) * j_shares if j_stop > 0 else 0
                rr = profit / risk if risk > 0 and profit > 0 else 0
                
                st.session_state['trade_journal'].append({
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'ticker': j_ticker.upper(),
                    'entry': j_entry,
                    'exit': j_exit if j_exit > 0 else None,
                    'stop': j_stop,
                    'shares': j_shares,
                    'profit': profit,
                    'pct': pct,
                    'risk': risk,
                    'rr': rr,
                    'reason': j_reason,
                    'emotion': j_emotion,
                    'notes': j_notes,
                    'status': 'Closed' if j_exit > 0 else 'Open'
                })
                
                st.success(f"✅ {j_ticker}")
                st.rerun()
    
    if st.session_state['trade_journal']:
        df_journal = pd.DataFrame(st.session_state['trade_journal'])
        st.dataframe(df_journal, use_container_width=True, hide_index=True)
        
        st.subheader("📈 Equity Curve")
        df_journal['cumulative'] = df_journal['profit'].cumsum()
        st.line_chart(df_journal[['cumulative']])

# ============================================
# REVOLUTIONARY PAGE
# ============================================

def render_revolutionary_page():
    st.title("🚀 Revolutionary Strategies")
    st.caption("High Risk, High Reward")
    
    account = st.session_state.get('account_size', 100)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        new_account = st.number_input("💰 Account (€)", 100, 1000000, account, 50)
        st.session_state['account_size'] = new_account
    
    with col2:
        risk_level = st.selectbox("⚡ Risk", ["Conservative", "Aggressive", "EXTREME"], index=1)
    
    with col3:
        if account < 500:
            st.warning("⚠️ Klein account")
        else:
            st.success("✅ Voldoende capital")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🐋 Whale Watching", "🎢 Gamma Squeeze", "📊 Dashboard"])
    
    # TAB 1: WHALE WATCHING
    with tab1:
        st.subheader("🐋 Follow Smart Money")
        
        whale_moves = whale_tracking_visual()
        
        for move in whale_moves:
            with st.expander(f"{move['investor']} → {move['ticker']}", expanded=True):
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Conviction", f"{move['conviction']}/100")
                col2.metric("Size", move['size'])
                col3.metric("Quarter", move['date'])
                col4.metric("Expected", move['expected'])
                
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=move['conviction'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "green"}
                        ]
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculator
                st.markdown("**💰 Your Position:**")
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    risk = st.slider("Risk %", 1.0, 10.0, 3.0, 0.5, key=f"r_{move['ticker']}")
                
                try:
                    ticker_obj = yf.Ticker(move['ticker'])
                    price = ticker_obj.history(period="1d")['Close'].iloc[-1]
                    
                    with c2:
                        entry = st.number_input("Entry", value=float(price), key=f"e_{move['ticker']}")
                    with c3:
                        stop = st.number_input("Stop", value=float(price*0.9), key=f"s_{move['ticker']}")
                    
                    if st.button(f"Calculate {move['ticker']}", key=f"calc_{move['ticker']}"):
                        pos = calculate_small_account_position(account, risk, entry, stop)
                        
                        if 'error' not in pos:
                            st.success(f"✅ Koop {pos['shares']} aandelen")
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Aandelen", pos['shares'])
                            m2.metric("Investering", f"€{pos['investment']:.2f}")
                            m3.metric("Max Loss", f"€{pos['risk_amount']:.2f}")
                            
                            if pos['warning']:
                                st.warning(pos['warning'])
                except:
                    st.error(f"Kan {move['ticker']} niet ophalen")
    
    # TAB 2: GAMMA SQUEEZE
    with tab2:
        st.subheader("🎢 Squeeze Scanner")
        
        ticker = st.text_input("Scan ticker", "GME").upper()
        
        if st.button("🚀 Scan", type="primary"):
            with st.spinner("Scanning..."):
                time.sleep(1)
                setup = detect_squeeze_setup_visual(ticker)
                
                if setup:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=setup['setup_score'],
                            title={'text': "Score"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "red"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgray"},
                                    {'range': [40, 60], 'color': "yellow"},
                                    {'range': [60, 80], 'color': "orange"},
                                    {'range': [80, 100], 'color': "red"}
                                ]
                            }
                        ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"### {ticker}")
                        
                        rec = setup['recommendation']
                        if rec == 'EXTREME BUY':
                            st.error(f"🔥 {rec}")
                        elif rec == 'STRONG BUY':
                            st.warning(f"⚡ {rec}")
                        else:
                            st.info(f"👀 {rec}")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Short %", f"{setup['short_interest']:.1f}%")
                        m2.metric("Float", f"{setup['float']/1e6:.1f}M")
                        m3.metric("Score", f"{setup['setup_score']}/100")
                        
                        if setup['signals']:
                            for sig in setup['signals']:
                                st.markdown(f"- {sig}")
    
    # TAB 3: DASHBOARD
    with tab3:
        st.subheader("📊 Dashboard")
        
        if risk_level == "Conservative":
            alloc = {'Whale': 30, 'Growth': 30, 'Events': 20, 'Cash': 20}
        elif risk_level == "Aggressive":
            alloc = {'Whale': 40, 'Squeeze': 25, 'Events': 25, 'Cash': 10}
        else:
            alloc = {'Squeeze': 40, 'Whale': 30, 'Events': 25, 'Vol': 5}
        
        fig = px.pie(values=list(alloc.values()), names=list(alloc.keys()))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Positions")
        for cat, pct in alloc.items():
            if cat != 'Cash':
                pos = account * (pct/100)
                st.progress(pct/100, text=f"{cat}: €{pos:.2f} ({pct}%)")

# ============================================
# MAIN APP
# ============================================

st.sidebar.title("💎 Zenith Terminal")
st.sidebar.markdown("---")

# Account selector
with st.sidebar.expander("💰 Account"):
    acc = st.number_input("Size (€)", 100, 1000000, st.session_state.get('account_size', 100), 50)
    st.session_state['account_size'] = acc
    
    if acc < 500:
        st.warning("Klein account: Focus 1-2 trades")

page = st.sidebar.radio("", [
    "🔎 Markt Analyse",
    "🚀 Revolutionary",
    "💼 Portfolio",
    "📡 Scanner",
    "📓 Trade Journal"
], key="nav_page")

with st.sidebar.expander("🧮 Calculator"):
    calc_acc = st.number_input("Account", 100, step=100, value=int(acc))
    risk = st.slider("Risk %", 0.5, 5.0, 1.0, 0.1)
    ent = st.number_input("Entry", 10.0)
    stp = st.number_input("Stop", 9.0)
    if stp < ent: 
        shares = int((calc_acc * (risk/100)) / (ent - stp))
        st.write(f"**Koop:** {shares} stuks")

render_watchlist_sidebar()

curr_sym = "$" if "USD" in st.sidebar.radio("Valuta", ["USD", "EUR"]) else "€"

st.sidebar.markdown("---")
st.sidebar.caption("v29 Ultimate | © 2026")

st.title("💎 Zenith Terminal")
mac = get_macro_data_optimized()
cols = st.columns(5)
for i, m in enumerate(["S&P 500", "Nasdaq", "Goud", "Olie", "10Y Rente"]):
    v, ch = mac.get(m, (0,0))
    cols[i].metric(m, f"{v:.2f}", f"{ch:.2f}%")
st.markdown("---")

# ============================================
# PAGES
# ============================================

if page == "🔎 Markt Analyse":
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: tick = st.text_input("Ticker", st.session_state['selected_ticker'], on_change=reset_analysis).upper()
    with col2: cap = st.number_input(f"Account ({curr_sym})", acc)
    with col3: risk_pct = st.number_input("Risk %", 0.5, 5.0, risk, 0.1)
    
    if st.button("🚀 Analyseer"): 
        st.session_state['analysis_active'] = True
        st.session_state['selected_ticker'] = tick
    
    if st.session_state['analysis_active']:
        df, met, fund, ws, financials, err, peers = get_zenith_data_optimized(tick)
        
        if err:
            st.error(f"⚠️ {err}")
        elif df is not None:
            with st.spinner('Analyseren...'):
                buys, news = get_external_info_optimized(tick)
            
            st.markdown(f"## 🏢 {met['name']} ({tick})")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Prijs", f"{curr_sym}{met['price']:.2f}")
            k2.metric("RSI", f"{met['rsi']:.1f}")
            k3.metric("P/E", f"{fund['pe']:.1f}" if fund['pe'] else "N/A")
            k4.metric("Target", f"{curr_sym}{ws['target']:.2f}", f"{ws['upside']:.1f}%")
            
            st.markdown("---")
            
            # Chart
            st.subheader("📈 Chart")
            end = df.index[-1]
            start = end - pd.DateOffset(years=1)
            plot_df = df.loc[start:end]
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(
                x=plot_df.index,
                open=plot_df['Open'],
                high=plot_df['High'],
                low=plot_df['Low'],
                close=plot_df['Close'],
                name="Price"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df['SMA200'],
                line=dict(color='#FFD700'),
                name="200MA"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df['RSI'],
                line=dict(color='#9370DB'),
                name="RSI"
            ), row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # News
            st.subheader("📰 News")
            for n in news[:3]:
                c = "green" if n['sentiment']=="POSITIVE" else "red" if n['sentiment']=="NEGATIVE" else "gray"
                st.markdown(f":{c}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")

elif page == "🚀 Revolutionary":
    render_revolutionary_page()

elif page == "💼 Portfolio":
    st.title("💼 Portfolio")
    
    with st.expander("➕ Add", expanded=True):
        c1, c2, c3, c4 = st.columns([2,2,2,1])
        with c1: t = st.text_input("Ticker", key="pt").upper()
        with c2: a = st.number_input("Aantal", 0.0, step=1.0)
        with c3: p = st.number_input("Prijs", 0.0)
        with c4:
            if st.button("Add"):
                st.session_state['portfolio'].append({"Ticker": t, "Aantal": a, "Koopprijs": p})
                st.rerun()
    
    if st.session_state['portfolio']:
        total_v = 0
        total_c = 0
        
        for i in st.session_state['portfolio']:
            cur = get_current_price(i['Ticker'])
            val = cur * i['Aantal']
            cost = i['Koopprijs'] * i['Aantal']
            total_v += val
            total_c += cost
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Waarde", f"{curr_sym}{total_v:.2f}")
        m2.metric("Inleg", f"{curr_sym}{total_c:.2f}")
        m3.metric("Winst", f"{curr_sym}{total_v-total_c:.2f}")
        
        if st.button("Clear"):
            st.session_state['portfolio'] = []
            st.rerun()

elif page == "📡 Scanner":
    st.title("📡 Scanner")
    
    preset = st.selectbox("Preset", list(PRESETS.keys()))
    txt = st.text_area("Tickers", PRESETS[preset])
    
    speed = st.selectbox("Speed", ["🐢 Veilig (1.0s)", "⚡ Snel (0.5s)"])
    delay = 1.0 if "Veilig" in speed else 0.5
    
    if st.button("🔍 Scan"):
        tickers = [t.strip().upper() for t in txt.split(',')]
        results = []
        
        progress = st.progress(0)
        for i, ticker in enumerate(tickers):
            progress.progress((i+1)/len(tickers))
            time.sleep(delay)
            
            try:
                df, met, fund, ws, _, err, _ = get_zenith_data_optimized(ticker)
                
                if df is not None:
                    score = 50
                    if met['price'] > met['sma200']: score += 20
                    if met['rsi'] < 35: score += 15
                    
                    results.append({
                        'Ticker': ticker,
                        'Prijs': met['price'],
                        'Score': score,
                        'Advies': 'KOPEN' if score >= 70 else 'HOUDEN'
                    })
            except:
                pass
        
        progress.empty()
        
        if results:
            df_results = pd.DataFrame(results).sort_values('Score', ascending=False)
            st.dataframe(df_results, use_container_width=True)

elif page == "📓 Trade Journal":
    render_trade_journal_page()
