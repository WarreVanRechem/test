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

st.set_page_config(page_title="Zenith Revolutionary Terminal", layout="wide", page_icon="🚀")
warnings.filterwarnings("ignore")

# Session state
if 'portfolio' not in st.session_state: st.session_state['portfolio'] = []
if 'nav_page' not in st.session_state: st.session_state['nav_page'] = "🔎 Markt Analyse"
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "AAPL"
if 'analysis_active' not in st.session_state: st.session_state['analysis_active'] = False
if 'watchlist' not in st.session_state: st.session_state['watchlist'] = []
if 'trade_journal' not in st.session_state: st.session_state['trade_journal'] = []
if 'account_size' not in st.session_state: st.session_state['account_size'] = 100  # Start met €100!

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
    "🇺🇸 Big Tech": "NVDA, AAPL, MSFT, GOOGL, AMZN",
    "🎮 Gaming": "GME, RBLX, EA, TTWO, ATVI",
    "💊 Biotech": "MRNA, BNTX, SAVA, CRSP, EDIT",
    "⚡ EV & Energy": "TSLA, RIVN, LCID, PLUG, ENPH",
    "🚀 Meme Stocks": "GME, AMC, BBBY, DWAC, SPCE"
}

# ============================================
# REVOLUTIONARY STRATEGIES - VISUAL
# ============================================

def detect_squeeze_setup_visual(ticker):
    """Gamma squeeze detection met visuals"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        signals = {
            'short_interest': info.get('shortPercentOfFloat', 0) * 100,
            'float': info.get('floatShares', 0),
            'setup_score': 0,
            'signals': []
        }
        
        # Short interest check
        if signals['short_interest'] > 30:
            signals['setup_score'] += 40
            signals['signals'].append("🔥 EXTREME Short Interest")
        elif signals['short_interest'] > 20:
            signals['setup_score'] += 25
            signals['signals'].append("⚠️ High Short Interest")
        
        # Float check
        if signals['float'] < 50_000_000:
            signals['setup_score'] += 30
            signals['signals'].append("💎 Tiny Float")
        elif signals['float'] < 100_000_000:
            signals['setup_score'] += 20
            signals['signals'].append("📉 Small Float")
        
        # Volume check
        df = stock.history(period="60d")
        if not df.empty:
            avg_vol = df['Volume'].rolling(50).mean().iloc[-1]
            recent_vol = df['Volume'].iloc[-5:].mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 0
            
            if vol_ratio > 3:
                signals['setup_score'] += 30
                signals['signals'].append("🚀 MASSIVE Volume Spike")
            elif vol_ratio > 2:
                signals['setup_score'] += 15
                signals['signals'].append("📊 Volume Increasing")
        
        signals['recommendation'] = 'EXTREME BUY' if signals['setup_score'] >= 80 else \
                                   'STRONG BUY' if signals['setup_score'] >= 60 else \
                                   'WATCH' if signals['setup_score'] >= 40 else \
                                   'SKIP'
        
        return signals
    except:
        return None

def whale_tracking_visual():
    """Track super investor moves"""
    
    # Demo data - in productie zou dit real SEC filings zijn
    whale_moves = [
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
    
    return whale_moves

def calculate_small_account_position(account_size, risk_pct, entry, stop):
    """Position sizing voor kleine accounts (€100+)"""
    
    # Voor kleine accounts: fixed fractional method
    risk_amount = account_size * (risk_pct / 100)
    risk_per_share = abs(entry - stop)
    
    if risk_per_share <= 0:
        return {'error': 'Stop moet verschillend zijn van entry'}
    
    # Calculate shares
    shares = risk_amount / risk_per_share
    
    # Voor hele kleine accounts: minimaal 1 aandeel
    if shares < 1:
        shares = 1
        actual_risk = shares * risk_per_share
        actual_risk_pct = (actual_risk / account_size) * 100
        
        return {
            'shares': int(shares),
            'investment': shares * entry,
            'risk_amount': actual_risk,
            'risk_pct': actual_risk_pct,
            'warning': f'⚠️ Min 1 aandeel = {actual_risk_pct:.1f}% risico (hoger dan gewenst)'
        }
    
    # Normal calculation
    shares = int(shares)
    investment = shares * entry
    
    # Check if investment > account (shouldn't happen but safety)
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

def render_revolutionary_page():
    """🚀 Revolutionary Strategies Page - VISUEEL"""
    
    st.title("🚀 Revolutionary Strategies")
    st.caption("High Risk, High Reward - Voor ervaren traders")
    
    # Account size check
    account = st.session_state.get('account_size', 100)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        new_account = st.number_input("💰 Account Size (€)", 100, 1000000, account, 50)
        st.session_state['account_size'] = new_account
    
    with col2:
        risk_level = st.selectbox("⚡ Risk Level", 
            ["🐢 Conservative (20-30%)", "⚡ Aggressive (40-80%)", "🚀 EXTREME (100-300%)"],
            index=1)
    
    with col3:
        if account < 500:
            st.warning("⚠️ Klein account: Focus op 1-2 trades")
        elif account < 2000:
            st.info("📊 Medium account: 3-5 posities mogelijk")
        else:
            st.success("💎 Groot account: Volledige diversificatie")
    
    st.markdown("---")
    
    # Strategy tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🐋 Whale Watching", 
        "🎢 Gamma Squeeze", 
        "📰 Event Calendar",
        "🌍 Macro Regime",
        "📊 Master Dashboard"
    ])
    
    # TAB 1: WHALE WATCHING
    with tab1:
        st.subheader("🐋 Follow the Smart Money")
        st.caption("Kopieer legendary investors VOOR de massa het ziet")
        
        whale_moves = whale_tracking_visual()
        
        for move in whale_moves:
            with st.expander(f"{move['investor']} → {move['ticker']} ({move['action']})", expanded=True):
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Conviction", f"{move['conviction']}/100")
                col2.metric("Position Size", move['size'])
                col3.metric("Quarter", move['date'])
                col4.metric("Expected Move", move['expected'])
                
                # Conviction bar
                fig_conv = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=move['conviction'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_conv.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_conv, use_container_width=True)
                
                # Position calculator
                st.markdown("**💰 Calculate Your Position:**")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    risk_pct = st.slider(f"Risk % {move['ticker']}", 1.0, 10.0, 3.0, 0.5, key=f"risk_{move['ticker']}")
                
                # Get current price
                try:
                    ticker_obj = yf.Ticker(move['ticker'])
                    current_price = ticker_obj.history(period="1d")['Close'].iloc[-1]
                    
                    with col_b:
                        entry_price = st.number_input(f"Entry €", value=float(current_price), key=f"entry_{move['ticker']}")
                    
                    with col_c:
                        stop_price = st.number_input(f"Stop €", value=float(current_price * 0.90), key=f"stop_{move['ticker']}")
                    
                    # Calculate position
                    if st.button(f"🎯 Calculate {move['ticker']}", key=f"calc_{move['ticker']}"):
                        pos = calculate_small_account_position(account, risk_pct, entry_price, stop_price)
                        
                        if 'error' in pos:
                            st.error(pos['error'])
                        else:
                            st.success(f"✅ Koop {pos['shares']} aandelen van {move['ticker']}")
                            
                            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                            met_col1.metric("Aandelen", pos['shares'])
                            met_col2.metric("Investering", f"€{pos['investment']:.2f}")
                            met_col3.metric("Max Loss", f"€{pos['risk_amount']:.2f}")
                            met_col4.metric("Risk %", f"{pos['risk_pct']:.1f}%")
                            
                            if pos['warning']:
                                st.warning(pos['warning'])
                except:
                    st.error(f"Kon {move['ticker']} niet ophalen")
    
    # TAB 2: GAMMA SQUEEZE
    with tab2:
        st.subheader("🎢 Gamma Squeeze Scanner")
        st.caption("Spot short squeezes VOORDAT ze exploderen")
        
        st.info("💡 **Tip:** Short Interest > 30% + Small Float + Volume Spike = 💣")
        
        # Scanner input
        scan_ticker = st.text_input("🔍 Scan Ticker", "GME").upper()
        
        if st.button("🚀 Scan Now", type="primary"):
            with st.spinner(f"Scanning {scan_ticker}..."):
                time.sleep(1)
                setup = detect_squeeze_setup_visual(scan_ticker)
                
                if setup:
                    # Score gauge
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        fig_score = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=setup['setup_score'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Squeeze Score"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "red"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgray"},
                                    {'range': [40, 60], 'color': "yellow"},
                                    {'range': [60, 80], 'color': "orange"},
                                    {'range': [80, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "darkred", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 80
                                }
                            }
                        ))
                        fig_score.update_layout(height=300)
                        st.plotly_chart(fig_score, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"### {scan_ticker} Setup Analysis")
                        
                        # Recommendation badge
                        rec = setup['recommendation']
                        if rec == 'EXTREME BUY':
                            st.error(f"🔥 {rec} - Dit is TNT!")
                        elif rec == 'STRONG BUY':
                            st.warning(f"⚡ {rec}")
                        elif rec == 'WATCH':
                            st.info(f"👀 {rec}")
                        else:
                            st.success(f"✅ {rec}")
                        
                        # Metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Short Interest", f"{setup['short_interest']:.1f}%")
                        m2.metric("Float", f"{setup['float']/1e6:.1f}M")
                        m3.metric("Score", f"{setup['setup_score']}/100")
                        
                        # Signals
                        if setup['signals']:
                            st.markdown("**🎯 Signals:**")
                            for signal in setup['signals']:
                                st.markdown(f"- {signal}")
                        
                        # Warning
                        if setup['setup_score'] >= 60:
                            st.warning("⚠️ **HIGH RISK:** Squeeze plays kunnen 50%+ verliezen als timing mis is!")
                else:
                    st.error(f"Kon {scan_ticker} niet analyseren")
        
        # Batch scanner
        st.markdown("---")
        st.markdown("### 📡 Batch Scanner")
        
        preset = st.selectbox("Selecteer lijst", list(PRESETS.keys()))
        
        if st.button("🔍 Scan All"):
            tickers = [t.strip() for t in PRESETS[preset].split(',')]
            
            results = []
            progress = st.progress(0)
            
            for i, ticker in enumerate(tickers):
                progress.progress((i + 1) / len(tickers))
                time.sleep(0.5)
                
                setup = detect_squeeze_setup_visual(ticker)
                if setup and setup['setup_score'] >= 40:
                    results.append({
                        'Ticker': ticker,
                        'Score': setup['setup_score'],
                        'Short %': setup['short_interest'],
                        'Recommendation': setup['recommendation']
                    })
            
            progress.empty()
            
            if results:
                df_results = pd.DataFrame(results).sort_values('Score', ascending=False)
                
                st.dataframe(df_results, use_container_width=True, hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100)
                    })
            else:
                st.info("Geen squeeze candidates gevonden")
    
    # TAB 3: EVENT CALENDAR
    with tab3:
        st.subheader("📰 Event-Driven Trading")
        st.caption("FDA approvals, Earnings surprises, M&A deals")
        
        # Demo FDA calendar
        st.markdown("### 💊 Upcoming FDA Decisions")
        
        fda_events = [
            {
                'ticker': 'SAVA',
                'drug': 'Simufilam (Alzheimer)',
                'date': '2025-09-15',
                'probability': 65,
                'expected_up': 150,
                'expected_down': -60
            },
            {
                'ticker': 'AKRO',
                'drug': 'Crinecerfont',
                'date': '2025-10-20',
                'probability': 80,
                'expected_up': 80,
                'expected_down': -40
            }
        ]
        
        for event in fda_events:
            with st.expander(f"💊 {event['ticker']} - {event['drug']}", expanded=True):
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("PDUFA Date", event['date'])
                col2.metric("Success Prob", f"{event['probability']}%")
                col3.metric("If Approved", f"+{event['expected_up']}%")
                col4.metric("If Rejected", f"{event['expected_down']}%")
                
                # Expected value calculation
                ev = (event['probability'] / 100) * event['expected_up'] + \
                     ((100 - event['probability']) / 100) * event['expected_down']
                
                if ev > 0:
                    st.success(f"✅ Expected Value: **+{ev:.1f}%** (Positive edge!)")
                else:
                    st.error(f"❌ Expected Value: **{ev:.1f}%** (Negative edge)")
                
                # Kelly calculator
                st.markdown("**🎲 Kelly Criterion:**")
                
                p = event['probability'] / 100
                b = abs(event['expected_up'] / event['expected_down'])
                kelly = (p * b - (1-p)) / b
                
                safe_kelly = max(kelly * 0.25, 0)  # 25% fractional Kelly
                safe_kelly = min(safe_kelly, 0.10)  # Max 10% for binary events
                
                if safe_kelly > 0:
                    recommended_position = account * safe_kelly
                    
                    st.info(f"💰 Recommended Position: **€{recommended_position:.2f}** ({safe_kelly*100:.1f}% of account)")
                    st.caption("⚠️ Dit is een BINARY bet - kan 100% winnen of verliezen!")
                else:
                    st.warning("⚠️ Kelly zegt: Skip deze trade (negatieve edge)")
    
    # TAB 4: MACRO REGIME
    with tab4:
        st.subheader("🌍 Macro Regime Analysis")
        st.caption("Verschillende strategie per markt conditie")
        
        # Detect current regime
        try:
            sp500 = yf.Ticker("^GSPC")
            df_sp = sp500.history(period="1y")
            vix = yf.Ticker("^VIX")
            vix_level = vix.history(period="1d")['Close'].iloc[-1]
            
            sma200 = df_sp['Close'].rolling(200).mean().iloc[-1]
            current = df_sp['Close'].iloc[-1]
            
            # Determine regime
            if current > sma200 and vix_level < 20:
                regime = "🐂 BULL MARKET"
                color = "green"
                strategy = {
                    'assets': ['NVDA', 'TSLA', 'COIN', 'MSTR'],
                    'style': 'High Beta Growth',
                    'expected': '50-100%/year',
                    'risk': 'High'
                }
            elif current < sma200 and vix_level > 30:
                regime = "🐻 BEAR MARKET"
                color = "red"
                strategy = {
                    'assets': ['SH', 'PSQ', 'SQQQ'],
                    'style': 'Inverse/Short',
                    'expected': '30-60%/year',
                    'risk': 'High'
                }
            elif vix_level > 40:
                regime = "💥 CRISIS"
                color = "orange"
                strategy = {
                    'assets': ['VXX', 'UVXY', 'GLD'],
                    'style': 'Volatility + Gold',
                    'expected': '100-300%/year',
                    'risk': 'EXTREME'
                }
            else:
                regime = "😐 NEUTRAL"
                color = "gray"
                strategy = {
                    'assets': ['SPY', 'QQQ', 'IWM'],
                    'style': 'Broad Index',
                    'expected': '10-20%/year',
                    'risk': 'Medium'
                }
            
            # Display regime
            st.markdown(f"### Current Regime: :{color}[{regime}]")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("S&P 500", f"€{current:.2f}", 
                         f"{((current - sma200)/sma200*100):.1f}% vs 200MA")
                st.metric("VIX (Fear Index)", f"{vix_level:.1f}",
                         "😱 High" if vix_level > 30 else "😌 Low")
            
            with col2:
                st.markdown("**📊 Recommended Strategy:**")
                st.info(f"**Style:** {strategy['style']}")
                st.info(f"**Expected:** {strategy['expected']}")
                st.info(f"**Risk:** {strategy['risk']}")
            
            # Assets to trade
            st.markdown("**🎯 Assets for this Regime:**")
            
            cols = st.columns(len(strategy['assets']))
            for i, asset in enumerate(strategy['assets']):
                with cols[i]:
                    if st.button(f"📊 {asset}", key=f"regime_{asset}"):
                        start_analysis_for(asset)
        
        except:
            st.error("Kon macro data niet ophalen")
    
    # TAB 5: MASTER DASHBOARD
    with tab5:
        st.subheader("📊 Master Dashboard")
        st.caption("Alle strategieën in één overzicht")
        
        # Portfolio allocation visualization
        st.markdown("### 💼 Recommended Portfolio Allocation")
        
        if risk_level == "🐢 Conservative (20-30%)":
            allocation = {
                'Whale Picks': 30,
                'Growth Momentum': 30,
                'Events': 20,
                'Cash': 20
            }
            expected_return = "20-30%"
            max_dd = "-15%"
        elif risk_level == "⚡ Aggressive (40-80%)":
            allocation = {
                'Whale Picks': 40,
                'Squeeze Candidates': 25,
                'Events': 25,
                'Cash': 10
            }
            expected_return = "40-80%"
            max_dd = "-30%"
        else:  # EXTREME
            allocation = {
                'Squeeze Candidates': 40,
                'Whale Picks': 30,
                'Events': 25,
                'Volatility': 5,
                'Cash': 0
            }
            expected_return = "100-300%"
            max_dd = "-50%"
        
        # Pie chart
        fig_pie = px.pie(
            values=list(allocation.values()),
            names=list(allocation.keys()),
            title=f"Portfolio Allocation ({risk_level})",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Expected metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Account", f"€{account:.0f}")
        col2.metric("📈 Expected Return", expected_return)
        col3.metric("📉 Max Drawdown", max_dd)
        
        # Position calculator
        st.markdown("---")
        st.markdown("### 🎯 Position Sizes per Strategy")
        
        for category, pct in allocation.items():
            if category != 'Cash':
                position_size = account * (pct / 100)
                st.progress(pct / 100, text=f"{category}: €{position_size:.2f} ({pct}%)")
        
        # Warnings for small accounts
        if account < 500:
            st.warning("""
            ⚠️ **Klein Account Strategie:**
            - Focus op 1-2 beste setups
            - Gebruik ALLEEN 8/10+ scores
            - Wacht op PERFECTE timing
            - Kan niet diversificeren (accepteer dit!)
            """)
        
        # Track performance
        if st.session_state['trade_journal']:
            st.markdown("---")
            st.markdown("### 📈 Jouw Performance")
            
            df_journal = pd.DataFrame(st.session_state['trade_journal'])
            
            total_trades = len(df_journal)
            winning = (df_journal['profit'] > 0).sum()
            win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
            total_pl = df_journal['profit'].sum()
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            perf_col1.metric("Trades", total_trades)
            perf_col2.metric("Win Rate", f"{win_rate:.1f}%")
            perf_col3.metric("Total P/L", f"€{total_pl:.2f}")
            perf_col4.metric("Return", f"{(total_pl/account*100):.1f}%")

# Add to main navigation
def main():
    st.sidebar.title("💎 Zenith Terminal")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("", [
        "🔎 Markt Analyse",
        "🚀 Revolutionary Strategies",  # ← NIEUW!
        "💼 Portfolio",
        "📡 Scanner",
        "📓 Trade Journal"
    ], key="nav_page")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v29 Revolutionary Edition")
    
    if page == "🚀 Revolutionary Strategies":
        render_revolutionary_page()
    else:
        st.info(f"Page {page} - Zie zenith_terminal_v28_final.py voor andere pagina's")

if __name__ == "__main__":
    main()
