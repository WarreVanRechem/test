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
from datetime import datetime
from io import BytesIO

# Try-except voor scipy
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal v28 ULTIMATE", layout="wide", page_icon="💎")
warnings.filterwarnings("ignore")

# --- DISCLAIMER & CREDITS ---
st.sidebar.error("⚠️ **DISCLAIMER:** Geen financieel advies. Educatief gebruik.")
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")

# --- SESSION STATE ---
if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
if 'nav_page' not in st.session_state: st.session_state['nav_page'] = "🔎 Markt Analyse"
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "AAPL"
if 'analysis_active' not in st.session_state: st.session_state['analysis_active'] = False
if 'watchlist' not in st.session_state: st.session_state['watchlist'] = []
if 'trade_journal' not in st.session_state: st.session_state['trade_journal'] = []

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

# ============================================
# OPTIMIZED CACHE FUNCTIONS 🔥
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_cached():
    """Cache S&P500 data voor 1 uur - HUGE performance win!"""
    try:
        return yf.Ticker("^GSPC").history(period="7y")
    except:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def get_macro_data_optimized():
    """Batch download macro data - 5x sneller!"""
    tickers_dict = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq", 
        "GC=F": "Goud",
        "CL=F": "Olie",
        "^TNX": "10Y Rente"
    }
    
    results = {}
    
    try:
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
# ENHANCED TRADING FUNCTIONS
# ============================================

def calculate_advanced_atr(df, period=14):
    """Berekent ATR met meerdere stop loss niveaus"""
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
    """Berekent meerdere strategische entry punten"""
    try:
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_lower = sma20 - (std20 * 2)
        bb_upper = sma20 + (std20 * 2)
        
        recent_prices = close.tail(100)
        support_levels = []
        resistance_levels = []
        
        for i in range(5, len(recent_prices) - 5):
            if recent_prices.iloc[i] == recent_prices.iloc[i-5:i+6].min():
                support_levels.append(recent_prices.iloc[i])
            if recent_prices.iloc[i] == recent_prices.iloc[i-5:i+6].max():
                resistance_levels.append(recent_prices.iloc[i])
        
        def cluster_levels(levels):
            if not levels:
                return []
            levels = sorted(levels)
            clusters = [[levels[0]]]
            for level in levels[1:]:
                if abs(level - clusters[-1][-1]) / clusters[-1][-1] < 0.02:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            return [np.mean(cluster) for cluster in clusters]
        
        support_zones = cluster_levels(support_levels)
        resistance_zones = cluster_levels(resistance_levels)
        
        recent_high = high.tail(100).max()
        recent_low = low.tail(100).min()
        diff = recent_high - recent_low
        
        fib_618 = recent_high - (diff * 0.618)
        fib_500 = recent_high - (diff * 0.500)
        fib_382 = recent_high - (diff * 0.382)
        
        current_price = close.iloc[-1]
        
        return {
            'current_price': current_price,
            'bb_entry': bb_lower.iloc[-1],
            'bb_upper': bb_upper.iloc[-1],
            'nearest_support': min(support_zones, key=lambda x: abs(x - current_price)) if support_zones else None,
            'nearest_resistance': min(resistance_zones, key=lambda x: abs(x - current_price)) if resistance_zones else None,
            'fib_618': fib_618,
            'fib_500': fib_500,
            'fib_382': fib_382,
            'support_zones': support_zones[-3:] if len(support_zones) >= 3 else support_zones,
            'resistance_zones': resistance_zones[:3] if len(resistance_zones) >= 3 else resistance_zones
        }
    except:
        return {}

def calculate_take_profit_levels(entry_price, stop_loss_price, df):
    """Berekent meerdere take profit targets"""
    try:
        risk = entry_price - stop_loss_price
        
        tp1 = entry_price + (risk * 1.5)
        tp2 = entry_price + (risk * 2.0)
        tp3 = entry_price + (risk * 3.0)
        
        swing_high_50d = df['High'].tail(50).max()
        
        return {'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'swing_high': swing_high_50d}
    except:
        return {'tp1': 0, 'tp2': 0, 'tp3': 0, 'swing_high': 0}

def calculate_position_size(account_balance, risk_pct, entry_price, stop_loss_price):
    """Smart position sizing met risk management"""
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
    """Geeft entry opportunity score 0-100"""
    try:
        score = 0
        signals = []
        
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9).mean()
        macd_cross = macd.iloc[-1] > signal_line.iloc[-1]
        
        avg_volume = df['Volume'].tail(20).mean()
        current_volume = df['Volume'].iloc[-1]
        volume_surge = current_volume > avg_volume * 1.5
        
        zones = calculate_entry_zones(df)
        
        if current_price > sma200:
            score += 15
            signals.append("✅ Boven 200MA (Bullish)")
        if current_price > sma50:
            score += 10
            signals.append("✅ Boven 50MA")
        
        if rsi < 30:
            score += 20
            signals.append("🔥 RSI Oversold (<30)")
        elif rsi < 40:
            score += 10
            signals.append("📉 RSI Laag (<40)")
        
        if zones.get('nearest_support'):
            near_support = abs(current_price - zones['nearest_support']) / current_price < 0.02
            if near_support:
                score += 15
                signals.append(f"💎 Bij Support")
        
        if zones.get('fib_618'):
            near_fib = abs(current_price - zones['fib_618']) / current_price < 0.02
            if near_fib:
                score += 10
                signals.append(f"🎯 Fib 0.618 Zone")
        
        if macd_cross:
            score += 10
            signals.append("📈 MACD Bullish")
        
        if volume_surge:
            score += 5
            signals.append("📊 Volume Surge")
        
        return {
            'score': min(score, 100),
            'signals': signals,
            'recommendation': 'STERK KOPEN' if score >= 75 else 'KOPEN' if score >= 60 else 'AFWACHTEN' if score >= 40 else 'NIET KOPEN'
        }
    except:
        return {'score': 0, 'signals': [], 'recommendation': 'ERROR'}

def generate_complete_trade_setup(ticker, df, account_balance, risk_pct=1.0):
    """Genereert complete trade setup"""
    try:
        current_price = df['Close'].iloc[-1]
        
        entry_zones = calculate_entry_zones(df)
        atr_data = calculate_advanced_atr(df)
        scoring = score_entry_opportunity(df, current_price)
        
        entry_options = []
        
        if entry_zones.get('nearest_support'):
            entry_options.append(('Support', entry_zones['nearest_support']))
        
        if entry_zones.get('fib_618'):
            entry_options.append(('Fib 0.618', entry_zones['fib_618']))
        
        entry_options.append(('Bollinger', entry_zones.get('bb_entry', current_price * 0.98)))
        
        valid_entries = [e for e in entry_options if e[1] < current_price]
        if valid_entries:
            best_entry = min(valid_entries, key=lambda x: current_price - x[1])
        else:
            best_entry = ('Current -2%', current_price * 0.98)
        
        entry_price = best_entry[1]
        entry_method = best_entry[0]
        
        stop_loss = atr_data['normal_stop']
        
        tp_levels = calculate_take_profit_levels(entry_price, stop_loss, df)
        
        position = calculate_position_size(account_balance, risk_pct, entry_price, stop_loss)
        
        if 'error' in position:
            return position
        
        risk_per_share = entry_price - stop_loss
        reward = tp_levels['tp2'] - entry_price
        rr_ratio = reward / risk_per_share if risk_per_share > 0 else 0
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'entry_price': entry_price,
            'entry_method': entry_method,
            'distance_to_entry': ((current_price - entry_price) / current_price) * 100,
            'stop_loss': stop_loss,
            'stop_loss_tight': atr_data['tight_stop'],
            'stop_loss_wide': atr_data['wide_stop'],
            'stop_loss_pct': ((entry_price - stop_loss) / entry_price) * 100,
            'atr_value': atr_data['atr_value'],
            'volatility': atr_data['volatility'],
            'tp1': tp_levels['tp1'],
            'tp2': tp_levels['tp2'],
            'tp3': tp_levels['tp3'],
            'rr_ratio': rr_ratio,
            'shares': position['shares'],
            'total_investment': position['total_investment'],
            'position_size_pct': position['position_size_pct'],
            'risk_amount': position['risk_amount'],
            'score': scoring['score'],
            'signals': scoring['signals'],
            'recommendation': scoring['recommendation'],
            'support_zones': entry_zones.get('support_zones', []),
            'resistance_zones': entry_zones.get('resistance_zones', []),
        }
    except Exception as e:
        return {'error': str(e)}

# ============================================
# OPTIMIZED DATA FUNCTIONS (HERGEBRUIK!) 🔥
# ============================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_zenith_data_optimized(ticker):
    """Optimized version met object hergebruik"""
    try:
        time.sleep(0.3)
        
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        
        if df.empty: 
            return None, None, None, None, None, None, "Geen data", None
        
        info = stock.info
        
        try:
            financials = stock.financials
        except:
            financials = None
        
        cur = df['Close'].iloc[-1]
        
        # Fair Value
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
        
        # Use cached S&P500!
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
        industry = info.get('industry', '').lower()
        peers = []
        
        if 'semicon' in industry: 
            peers = ["NVDA", "AMD", "INTC", "TSM", "ASML"]
        elif 'software' in industry or 'technology' in sector: 
            peers = ["MSFT", "AAPL", "GOOGL", "ORCL", "ADBE"]
        elif 'bank' in industry: 
            peers = ["JPM", "BAC", "C", "WFC", "HSBC"]
        elif 'oil' in industry or 'energy' in sector: 
            peers = ["XOM", "CVX", "SHEL", "TTE", "BP"]
        elif 'auto' in industry: 
            peers = ["TSLA", "TM", "F", "GM", "STLA"]
        elif 'drug' in industry or 'healthcare' in sector: 
            peers = ["LLY", "JNJ", "PFE", "MRK", "NVS"]
        
        if not peers:
            if 'tech' in sector: peers = ["XLK"]
            elif 'health' in sector: peers = ["XLV"]
            elif 'financ' in sector: peers = ["XLF"]
            elif 'energy' in sector: peers = ["XLE"]
            else: peers = ["^GSPC"]
        
        peers = [p for p in peers if p.upper() != ticker.upper()][:4]
        
        met = {
            "name": info.get('longName', ticker), 
            "price": cur, 
            "sma200": df['SMA200'].iloc[-1], 
            "rsi": df['RSI'].iloc[-1], 
            "bull": mb
        }
        
        return df, met, fund, ws, stock, financials, None, peers
        
    except Exception as e:
        return None, None, None, None, None, None, str(e), None

def get_external_info_optimized(stock_obj):
    """Hergebruik stock object"""
    try:
        buys = 0
        
        try:
            ins = stock_obj.insider_transactions
            if ins is not None and not ins.empty:
                buys = ins.head(10)[ins.head(10)['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
        except:
            pass
        
        ticker = stock_obj.ticker
        try:
            f = feedparser.parse(
                requests.get(
                    f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=3
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

def get_financial_trends_optimized(financials_obj):
    """Process financials zonder nieuwe API call"""
    try:
        if financials_obj is None or financials_obj.empty:
            return None
        
        f = financials_obj.T
        cols = [c for c in ['Total Revenue', 'Net Income'] if c in f.columns]
        
        if not cols:
            return None
        
        df = f[cols].dropna()
        df.index = df.index.year
        return df.sort_index()
    except:
        return None

# ============================================
# OUDE FUNCTIES (BLIJVEN HETZELFDE)
# ============================================

def get_smart_peers(ticker, info):
    if not info: return ["^GSPC"]
    sector = info.get('sector', '').lower()
    industry = info.get('industry', '').lower()
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
    
    peers = [p for p in peers if p.upper() != ticker.upper()]
    return peers[:4]

def compare_peers(main_ticker, peers_list):
    try:
        df = yf.download([main_ticker] + peers_list, period="1y")['Close']
        if df.empty: return None
        return df.apply(lambda x: ((x / x.iloc[0]) - 1) * 100)
    except: return None

def run_backtest(ticker, period="5y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty or len(df) < 250: return "Te weinig data"
        
        df['SMA200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta>0,0)).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        df['RSI'] = 100-(100/(1+(gain/loss)))
        
        balance=10000
        shares=0
        trades=[]
        in_pos=False
        
        for i in range(201, len(df)):
            p = df['Close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            sma = df['SMA200'].iloc[i]
            
            if pd.isna(sma): continue
            
            if not in_pos and p > sma and rsi < 35:
                shares = balance/p
                balance=0
                in_pos=True
                trades.append({"Date":df.index[i],"Type":"BUY","Price":p})
            elif in_pos and (rsi > 75 or p < sma*0.95):
                balance = shares*p
                shares=0
                in_pos=False
                trades.append({"Date":df.index[i],"Type":"SELL","Price":p})
        
        final = balance if not in_pos else shares*df['Close'].iloc[-1]
        
        return {
            "return": ((final-10000)/10000)*100, 
            "bh_return": ((df['Close'].iloc[-1]-df['Close'].iloc[201])/df['Close'].iloc[201])*100, 
            "trades": len(trades), 
            "final_value": final, 
            "history": df
        }
    except: 
        return "Backtest Error"

def run_monte_carlo(ticker):
    try:
        d = yf.Ticker(ticker).history(period="1y")['Close']
        ret = d.pct_change().dropna()
        sims = []
        mu=ret.mean()
        sig=ret.std()
        
        for _ in range(200):
            p = [d.iloc[-1]]
            for _ in range(252): 
                p.append(p[-1]*(1+np.random.normal(mu,sig)))
            sims.append(p)
        
        return np.array(sims)
    except: 
        return None

def optimize_portfolio(tickers):
    if not SCIPY_AVAILABLE: return "SCIPY_MISSING"
    try:
        data = yf.download(tickers, period="1y")['Close'].dropna()
        if data.empty or len(tickers) < 2: return None
        
        returns = data.pct_change()
        mean_ret = returns.mean()
        cov_mat = returns.cov()
        
        def neg_sharpe(weights):
            p_ret = np.sum(mean_ret * weights) * 252
            p_var = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
            return -(p_ret - 0.04) / p_var if p_var > 0 else 0
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        res = minimize(neg_sharpe, len(tickers)*[1./len(tickers)], bounds=bounds, constraints=constraints)
        
        return dict(zip(tickers, res.x))
    except: 
        return None

def get_current_price(ticker):
    try:
        obj = yf.Ticker(ticker)
        p = obj.fast_info.last_price
        if not pd.isna(p) and p > 0: return p
        h = obj.history(period="1d")
        if not h.empty: return h['Close'].iloc[-1]
    except: pass
    return 0.0

# ============================================
# WATCHLIST SIDEBAR 🔥
# ============================================

def render_watchlist_sidebar():
    """Render watchlist in sidebar"""
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
                if st.button("📊", key=f"analyze_{ticker}_{idx}", help="Analyze"):
                    start_analysis_for(ticker)
            
            with col3:
                if st.button("🗑️", key=f"delete_{ticker}_{idx}", help="Remove"):
                    st.session_state['watchlist'].remove(ticker)
                    st.rerun()
        
        if st.sidebar.button("🔍 Scan Watchlist", use_container_width=True):
            st.session_state['nav_page'] = "📡 Deep Scanner"
            st.rerun()
    else:
        st.sidebar.caption("Geen items in watchlist")

# ============================================
# TRADE JOURNAL 🔥
# ============================================

def render_trade_journal_page():
    """Complete trade journal pagina"""
    st.title("📓 Trade Journal")
    st.caption("Track je trades en leer van je fouten")
    
    if st.session_state['trade_journal']:
        df_journal = pd.DataFrame(st.session_state['trade_journal'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(df_journal)
        winning_trades = (df_journal['profit'] > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pl = df_journal['profit'].sum()
        
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%", "🟢" if win_rate > 50 else "🔴")
        col3.metric("Total P/L", f"€{total_pl:.2f}")
        col4.metric("Avg R/R", f"{df_journal['rr'].mean():.2f}")
        
        st.markdown("---")
    
    with st.expander("➕ Log Nieuwe Trade", expanded=not bool(st.session_state['trade_journal'])):
        col1, col2 = st.columns(2)
        
        with col1:
            j_ticker = st.text_input("Ticker", key="j_ticker")
            j_entry = st.number_input("Entry Prijs", 0.0, key="j_entry")
            j_stop = st.number_input("Stop Loss", 0.0, key="j_stop")
            j_shares = st.number_input("Aantal", 1, key="j_shares")
        
        with col2:
            j_exit = st.number_input("Exit (0 = open)", 0.0, key="j_exit")
            j_reason = st.text_input("Reden", key="j_reason")
            j_emotion = st.selectbox("Emotie", ["😌 Kalm", "😰 FOMO", "😤 Wraak", "🤔 Unsure"], key="j_emotion")
            j_notes = st.text_area("Notes", key="j_notes")
        
        if st.button("💾 Log Trade", type="primary"):
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
                
                st.success(f"✅ Trade logged: {j_ticker}")
                st.rerun()
            else:
                st.error("Vul minimaal Ticker en Entry in!")
    
    if st.session_state['trade_journal']:
        df_journal = pd.DataFrame(st.session_state['trade_journal'])
        
        st.dataframe(df_journal, use_container_width=True, hide_index=True,
            column_config={
                "profit": st.column_config.NumberColumn("P/L", format="€%.2f"),
                "pct": st.column_config.NumberColumn("P/L %", format="%.2f%%"),
                "entry": st.column_config.NumberColumn("Entry", format="€%.2f"),
                "exit": st.column_config.NumberColumn("Exit", format="€%.2f"),
                "rr": st.column_config.NumberColumn("R/R", format="%.2f")
            }
        )
        
        st.subheader("📈 Equity Curve")
        df_journal['cumulative_pl'] = df_journal['profit'].cumsum()
        st.line_chart(df_journal[['cumulative_pl']])
        
        st.download_button("📥 Download Journal", df_journal.to_csv(index=False), "trade_journal.csv", "text/csv")
        
        if st.button("🗑️ Wis Alles"):
            if st.checkbox("Weet je het zeker?"):
                st.session_state['trade_journal'] = []
                st.rerun()
    else:
        st.info("📝 Nog geen trades gelogd.")

# ============================================
# UI START
# ============================================

st.sidebar.header("Navigatie")
page = st.sidebar.radio("Ga naar:", [
    "🔎 Markt Analyse", 
    "💼 Mijn Portfolio", 
    "📡 Deep Scanner",
    "📓 Trade Journal",
    "🎓 Leer de Basics"
], key="nav_page")

with st.sidebar.expander("🧮 Calculator"):
    acc=st.number_input("Account",10000,step=1000)
    risk=st.slider("Risk %",0.5,5.0,1.0,0.1)
    ent=st.number_input("Entry",100.0)
    stp=st.number_input("Stop",95.0)
    if stp<ent: st.write(f"**Koop:** {int((acc*(risk/100))/(ent-stp))} stuks")

render_watchlist_sidebar()

curr_sym = "$" if "USD" in st.sidebar.radio("Valuta", ["USD", "EUR"]) else "€"

with st.sidebar.expander("🔧 Advanced"):
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")

st.title("💎 Zenith Ultimate Terminal v28")
mac = get_macro_data_optimized()
cols = st.columns(5)
for i, m in enumerate(["S&P 500", "Nasdaq", "Goud", "Olie", "10Y Rente"]):
    v, ch = mac.get(m, (0,0))
    cols[i].metric(m, f"{v:.2f}", f"{ch:.2f}%")
st.markdown("---")

# ============================================
# PAGINA 1: MARKT ANALYSE
# ============================================

if page == "🔎 Markt Analyse":
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1: tick = st.text_input("Ticker", value=st.session_state['selected_ticker'], on_change=reset_analysis).upper()
    with c2: cap = st.number_input(f"Account ({curr_sym})", acc)
    with c3: risk_pct = st.number_input("Risk %", 0.5, 5.0, risk, 0.1)
    
    if st.button("🚀 Start Analysis"): 
        st.session_state['analysis_active'] = True
        st.session_state['selected_ticker'] = tick
    
    if st.session_state['analysis_active']:
        df, met, fund, ws, stock_obj, financials, err, peers = get_zenith_data_optimized(st.session_state['selected_ticker'])
        
        if err: 
            st.error(f"⚠️ {err}")
        elif df is not None:
            with st.spinner('Analyseren...'): 
                buys, news = get_external_info_optimized(stock_obj)
                trade_setup = generate_complete_trade_setup(tick, df, cap, risk_pct)
            
            if 'error' in trade_setup:
                st.error(f"Setup Error: {trade_setup['error']}")
            else:
                st.markdown(f"## 🏢 {met['name']} ({tick})")
                
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("AI Score", f"{trade_setup['score']}/100")
                
                sig_color = "green" if "KOPEN" in trade_setup['recommendation'] else "orange" if "AFWACH" in trade_setup['recommendation'] else "red"
                k2.markdown(f"**Signal:** :{sig_color}[{trade_setup['recommendation']}]")
                
                k3.metric("Prijs", f"{curr_sym}{trade_setup['current_price']:.2f}")
                k4.metric("Volatility", f"{trade_setup['volatility']:.2f}%")
                
                if fund['fair_value']:
                    diff_fair = ((fund['fair_value'] - met['price']) / met['price']) * 100
                    k5.metric("Fair Value", f"{curr_sym}{fund['fair_value']:.2f}", f"{diff_fair:.1f}%")
                else: 
                    k5.metric("Fair Value", "N/A", "Loss Making")
                
                if trade_setup['signals']:
                    st.info("**🎯 Signals:** " + " | ".join(trade_setup['signals']))
                
                st.markdown("---")
                
                # TRADE SETUP
                st.subheader("🎯 Professional Trade Setup")
                
                st.markdown("### 1️⃣ ENTRY")
                e1, e2, e3 = st.columns(3)
                entry_status = "🟢 KOOP NU" if trade_setup['distance_to_entry'] < 1 else f"🟡 Wacht -{trade_setup['distance_to_entry']:.1f}%"
                e1.metric("Entry", f"{curr_sym}{trade_setup['entry_price']:.2f}", entry_status)
                e2.metric("Methode", trade_setup['entry_method'])
                e3.metric("Afstand", f"{trade_setup['distance_to_entry']:.2f}%")
                
                st.markdown("### 2️⃣ STOP LOSS")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("🔴 Tight", f"{curr_sym}{trade_setup['stop_loss_tight']:.2f}", "-1.5 ATR")
                s2.metric("🟠 Normal", f"{curr_sym}{trade_setup['stop_loss']:.2f}", "-2.0 ATR ✅")
                s3.metric("🟢 Wide", f"{curr_sym}{trade_setup['stop_loss_wide']:.2f}", "-3.0 ATR")
                s4.metric("Stop %", f"{trade_setup['stop_loss_pct']:.2f}%")
                
                st.markdown("### 3️⃣ TAKE PROFIT")
                t1, t2, t3 = st.columns(3)
                profit_tp1 = ((trade_setup['tp1'] - trade_setup['entry_price']) / trade_setup['entry_price']) * 100
                profit_tp2 = ((trade_setup['tp2'] - trade_setup['entry_price']) / trade_setup['entry_price']) * 100
                profit_tp3 = ((trade_setup['tp3'] - trade_setup['entry_price']) / trade_setup['entry_price']) * 100
                
                t1.metric("TP1 (1/3)", f"{curr_sym}{trade_setup['tp1']:.2f}", f"+{profit_tp1:.1f}%")
                t2.metric("TP2 (1/3)", f"{curr_sym}{trade_setup['tp2']:.2f}", f"+{profit_tp2:.1f}%")
                t3.metric("TP3 (1/3)", f"{curr_sym}{trade_setup['tp3']:.2f}", f"+{profit_tp3:.1f}%")
                
                rr_color = "green" if trade_setup['rr_ratio'] >= 2 else "orange"
                st.markdown(f"**Risk/Reward:** :{rr_color}[1 : {trade_setup['rr_ratio']:.2f}]")
                
                st.markdown("### 4️⃣ POSITION")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Aandelen", f"{trade_setup['shares']}")
                p2.metric("Investering", f"{curr_sym}{trade_setup['total_investment']:.2f}")
                p3.metric("Max Loss", f"{curr_sym}{trade_setup['risk_amount']:.2f}")
                p4.metric("% Account", f"{trade_setup['position_size_pct']:.1f}%")
                
                st.markdown("---")
                
                # FUNDAMENTALS
                st.subheader("📊 Fundamentals")
                fin_df = get_financial_trends_optimized(financials)
                if fin_df is not None:
                    f_fig = px.bar(fin_df, barmode='group', template="plotly_dark", color_discrete_sequence=['#636EFA', '#00CC96'])
                    st.plotly_chart(f_fig, use_container_width=True)
                else: 
                    st.warning("Geen data")
                
                # CHART
                st.subheader("📈 Technical Chart")
                end = df.index[-1]
                start = end - pd.DateOffset(years=1)
                plot_df = df.loc[start:end]
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.2,0.2])
                
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['L'], line=dict(color='rgba(0,255,0,0.3)'), name="Lower"), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['U'], line=dict(color='rgba(255,0,0,0.3)'), fill='tonexty', name="Upper"), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], line=dict(color='#FFD700'), name="200MA"), row=1, col=1)
                
                if 'M' in plot_df.columns: 
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['M'], line=dict(color='white', width=1, dash='dot'), name="S&P500"), row=1, col=1)
                
                fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Price"), row=1, col=1)
                
                fig.add_hline(y=trade_setup['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry", row=1, col=1)
                fig.add_hline(y=trade_setup['stop_loss'], line_dash="dash", line_color="red", annotation_text="Stop", row=1, col=1)
                fig.add_hline(y=trade_setup['tp2'], line_dash="dash", line_color="green", annotation_text="TP2", row=1, col=1)
                
                clrs = ['green' if r['Open']<r['Close'] else 'red' for i,r in plot_df.iterrows()]
                fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=clrs, name="Vol"), row=2, col=1)
                
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#9370DB'), name="RSI"), row=3, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
                
                fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # TABS
                t1, t2, t3, t4 = st.tabs(["⚔️ Peers", "🔙 Backtest", "🔮 Monte Carlo", "📰 News"])
                
                with t1:
                    if peers:
                        st.write(f"Peers: {', '.join(peers)}")
                        if st.button("Load"):
                            pd_data = compare_peers(tick, peers)
                            if pd_data is not None: 
                                st.line_chart(pd_data)
                
                with t2:
                    if st.button("Run Backtest"):
                        res = run_backtest(tick)
                        if isinstance(res, str): 
                            st.error(res)
                        else:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Strategy", f"{res['return']:.1f}%")
                            c2.metric("Buy&Hold", f"{res['bh_return']:.1f}%")
                            c3.metric("Trades", res['trades'])
                            st.line_chart(res['history']['Close'])
                
                with t3:
                    if st.button("Simulate"):
                        sims = run_monte_carlo(tick)
                        if sims is not None:
                            f = go.Figure()
                            for i in range(50): 
                                f.add_trace(go.Scatter(y=sims[i], mode='lines', line=dict(width=1, color='rgba(0,255,255,0.1)'), showlegend=False))
                            f.add_trace(go.Scatter(y=np.mean(sims,axis=0), mode='lines', line=dict(width=3, color='yellow'), name='Avg'))
                            f.update_layout(template="plotly_dark")
                            st.plotly_chart(f, use_container_width=True)
                
                with t4:
                    for n in news:
                        c = "green" if n['sentiment']=="POSITIVE" else "red" if n['sentiment']=="NEGATIVE" else "gray"
                        st.markdown(f":{c}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")

# ============================================
# PAGINA 2: PORTFOLIO
# ============================================

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
                st.success("Added!")
                st.rerun()
    
    if st.session_state['portfolio']:
        p_data = []
        tot_v = 0
        tot_c = 0
        tickers = []
        
        for i in st.session_state['portfolio']:
            cur = get_current_price(i['Ticker'])
            val = cur * i['Aantal']
            cost = i['Koopprijs'] * i['Aantal']
            tot_v += val
            tot_c += cost
            prof = val - cost
            pct = (prof/cost)*100 if cost>0 else 0
            clr = "green" if prof>=0 else "red"
            tickers.append(i['Ticker'])
            p_data.append({
                "Ticker": i['Ticker'], 
                "Aantal": i['Aantal'], 
                "Waarde": f"{curr_sym}{val:.2f}", 
                "Winst": f":{clr}[{curr_sym}{prof:.2f} ({pct:.1f}%)]"
            })
        
        st.write(pd.DataFrame(p_data).to_markdown(index=False), unsafe_allow_html=True)
        
        if len(tickers) > 1:
            st.subheader("🔗 Correlatie")
            try:
                corr_data = yf.download(tickers, period="1y")['Close'].pct_change().corr()
                fig_corr = px.imshow(corr_data, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)
            except: 
                st.warning("Error")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Waarde", f"{curr_sym}{tot_v:.2f}")
        m2.metric("Inleg", f"{curr_sym}{tot_c:.2f}")
        m3.metric("Winst", f"{curr_sym}{tot_v-tot_c:.2f}", f"{((tot_v-tot_c)/tot_c)*100 if tot_c>0 else 0:.1f}%")
        
        if st.button("Optimaliseer"):
            w = optimize_portfolio(tickers)
            if isinstance(w, str): 
                st.error(w)
            elif w: 
                df_w = pd.DataFrame(list(w.items()), columns=['Ticker', 'Ideaal'])
                st.bar_chart(df_w.set_index('Ticker'))
        
        if st.button("Wissen"): 
            st.session_state['portfolio'] = []
            st.rerun()
    else: 
        st.write("Leeg")

# ============================================
# PAGINA 3: SCANNER
# ============================================

elif page == "📡 Deep Scanner":
    st.title("📡 Deep Scanner")
    
    pre = st.selectbox("Markt", list(PRESETS.keys()))
    txt = st.text_area("Tickers", PRESETS[pre])
    
    col1, col2 = st.columns(2)
    with col1:
        scan_speed = st.selectbox("Snelheid:", ["🐢 Veilig (0.8s)", "⚡ Gemiddeld (0.5s)", "🚀 Snel (0.3s)"])
    with col2:
        st.info("💡 Kies 'Veilig' bij errors")
    
    delay = 0.8 if "Veilig" in scan_speed else 0.5 if "Gemiddeld" in scan_speed else 0.3
    
    if st.button("🔍 Scan"):
        ticker_list = [x.strip().upper() for x in txt.split(',')]
        
        results = []
        failed = []
        total = len(ticker_list)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(ticker_list):
            progress_bar.progress((idx + 1) / total)
            status_text.text(f"📊 Scanning {ticker}... ({idx+1}/{total})")
            
            try:
                time.sleep(delay)
                
                df, met, fund, ws, _, _, err, _ = get_zenith_data_optimized(ticker)
                
                if err:
                    failed.append(f"{ticker}: {err}")
                    continue
                
                if df is not None and met is not None:
                    score = 50
                    reasons = []
                    
                    if met['price'] > met['sma200']:
                        score += 20
                        reasons.append("Trend 📈")
                    
                    if met['rsi'] < 30:
                        score += 15
                        reasons.append("Oversold")
                    elif met['rsi'] < 40:
                        score += 10
                        reasons.append("RSI Laag")
                    
                    if ws['upside'] > 15:
                        score += 15
                        reasons.append("Analisten")
                    
                    if fund['fair_value'] and met['price'] < fund['fair_value']:
                        score += 15
                        reasons.append("Value")
                    
                    if fund['pe'] > 0 and fund['pe'] < 20:
                        score += 10
                        reasons.append("Goedkoop")
                    
                    adv = "KOPEN" if score >= 70 else "HOUDEN" if score >= 50 else "AFBLIJVEN"
                    
                    results.append({
                        "Ticker": ticker,
                        "Prijs": met['price'],
                        "Target": f"{curr_sym}{ws['target']:.2f}" if ws['target'] else "N/A",
                        "Upside": f"{ws['upside']:.1f}%",
                        "Score": score,
                        "Advies": adv,
                        "Reden": " + ".join(reasons) if reasons else "-"
                    })
                else:
                    failed.append(f"{ticker}: Geen data")
                    
            except Exception as e:
                failed.append(f"{ticker}: {str(e)}")
                time.sleep(1)
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state['res'] = results
        st.session_state['failed'] = failed
    
    if 'res' in st.session_state:
        if not st.session_state['res']:
            st.warning("Geen resultaten")
        else:
            df_results = pd.DataFrame(st.session_state['res']).sort_values('Score', ascending=False)
            
            st.dataframe(df_results, use_container_width=True, hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100),
                    "Prijs": st.column_config.NumberColumn("Prijs", format=f"{curr_sym}%.2f")
                }
            )
            
            st.download_button("📥 CSV", df_results.to_csv(index=False), "scan.csv", "text/csv")
            
            st.markdown("---")
            c1, c2 = st.columns([3, 1])
            options = [r['Ticker'] for r in st.session_state['res']]
            
            if options:
                sel = c1.selectbox("Analyseer:", options)
                c2.button("🚀 Go", on_click=start_analysis_for, args=(sel,))
        
        if 'failed' in st.session_state and st.session_state['failed']:
            with st.expander(f"⚠️ Errors ({len(st.session_state['failed'])})"):
                for error in st.session_state['failed']:
                    st.text(error)

# ============================================
# PAGINA 4: TRADE JOURNAL
# ============================================

elif page == "📓 Trade Journal":
    render_trade_journal_page()

# ============================================
# PAGINA 5: EDUCATIE
# ============================================

elif page == "🎓 Leer de Basics":
    st.title("🎓 Zenith Academy")
    
    with st.expander("🎯 1. Entry Strategieën"):
        st.write("""
        **Fibonacci 0.618 (Golden Ratio)**
        - Meest betrouwbare retracement level
        - Gebruik in combinatie met support
        
        **Support/Resistance**
        - Plekken waar prijs eerder keerde
        - Hoe vaker getest, hoe sterker
        
        **Bollinger Bands**
        - Lower band = mogelijk oversold
        - Squeeze = breakout komt
        """)
    
    with st.expander("🛑 2. Stop Loss"):
        st.write("""
        **ATR-based (Beste)**
        - Tight (1.5 ATR): Day traders
        - Normal (2.0 ATR): Swing traders ✅
        - Wide (3.0 ATR): Position traders
        
        **Waarom ATR?**
        - Past zich aan markt aan
        - Voorkomt uitschudden
        """)
    
    with st.expander("💰 3. Position Sizing"):
        st.write("""
        **1% Rule**
        - Max 1-2% risico per trade
        - €10,000 account = €100-200 risico
        
        **Berekening:**
        Aandelen = (Account × Risk%) / (Entry - Stop)
        
        **Max 20% per positie**
        """)
    
    with st.expander("🎯 4. Take Profit"):
        st.write("""
        **Ladder Strategie:**
        1. TP1 (1.5R): 1/3 uit, stop naar breakeven
        2. TP2 (2R): 1/3 uit, stop naar TP1
        3. TP3 (3R): Laat lopen met trailing
        """)
    
    st.success("💡 **Golden Rules:** Risk 1-2% | Min 1:2 RR | ATR stops | Ladder exits | Diversify")
