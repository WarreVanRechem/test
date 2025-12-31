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
from scipy.optimize import minimize # NODIG VOOR OPTIMALISATIE

# --- CONFIGURATIE ---
st.set_page_config(page_title="Zenith Terminal v20.0 Architect", layout="wide", page_icon="💎")
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

# --- QUANT FUNCTIONS (NIEUW IN V20) ---

def run_backtest(ticker, period="5y"):
    """
    Simuleert de Zenith Strategie over het verleden.
    Strategie: KOPEN als Prijs > 200MA EN RSI < 35. VERKOPEN als RSI > 70.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty: return None
        
        # Indicatoren
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Backtest Loop
        balance = 10000
        shares = 0
        trades = []
        in_position = False
        
        for i in range(200, len(df)):
            price = df['Close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            sma = df['SMA200'].iloc[i]
            date = df.index[i]
            
            # BUY SIGNAL (Trend + Oversold)
            if not in_position and price > sma and rsi < 35:
                shares = balance / price
                balance = 0
                in_position = True
                trades.append({"Date": date, "Type": "BUY", "Price": price, "Value": shares * price})
                
            # SELL SIGNAL (Overbought of Trendbreuk)
            elif in_position and (rsi > 75 or price < sma * 0.95):
                balance = shares * price
                shares = 0
                in_position = False
                trades.append({"Date": date, "Type": "SELL", "Price": price, "Value": balance})
                
        # Eindwaarde
        final_val = balance if not in_position else shares * df['Close'].iloc[-1]
        ret = ((final_val - 10000) / 10000) * 100
        
        # Benchmark (Buy & Hold)
        start_price = df['Close'].iloc[200]
        end_price = df['Close'].iloc[-1]
        bh_ret = ((end_price - start_price) / start_price) * 100
        
        return {
            "return": ret,
            "bh_return": bh_ret,
            "trades": len(trades),
            "final_value": final_val,
            "history": df
        }
    except: return None

def optimize_portfolio(portfolio_tickers):
    """Markowitz Efficient Frontier Optimalisatie."""
    try:
        if len(portfolio_tickers) < 2: return None
        data = yf.download(portfolio_tickers, period="1y")['Close']
        returns = data.pct_change()
        
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(portfolio_tickers)
        
        def portfolio_performance(weights, mean_returns, cov_matrix):
            returns = np.sum(mean_returns * weights) * 252
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return std, returns
        
        def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
            p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
            return -(p_ret - risk_free_rate) / p_var
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]
        
        opt_results = minimize(neg_sharpe, init_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
        
        return dict(zip(portfolio_tickers, opt_results.x))
    except: return None

# --- BASIS FUNCTIES (BEHOUDEN VAN V19) ---
# ... (Deze blijven hetzelfde voor stabiliteit)
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
    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            price = t.fast_info.last_price
            prev = t.fast_info.previous_close
            if price and prev:
                change = ((price - prev) / prev) * 100
                data[name] = (price, change)
            else: data[name] = (0, 0)
        except: data[name] = (0, 0)
    return data

@st.cache_data(ttl=3600)
def get_zenith_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="7y")
        if df.empty: return None, None, None, None, None, None, "Geen data."
        info = stock.info
        current_p = df['Close'].iloc[-1]
        
        div_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate')
        dividend_pct = (div_rate / current_p) * 100 if (div_rate and current_p > 0) else (info.get('dividendYield', 0) * 100 if info.get('dividendYield', 0) < 0.5 else info.get('dividendYield', 0))
        
        fundamentals = {"pe": info.get('trailingPE', 0), "dividend": dividend_pct, "sector": info.get('sector', "-"), "profit_margin": (info.get('profitMargins') or 0) * 100}
        wall_street = {"target": info.get('targetMeanPrice', 0) or 0, "recommendation": info.get('recommendationKey', 'none').upper(), "upside": ((info.get('targetMeanPrice', 0) - current_p) / current_p) * 100 if info.get('targetMeanPrice') else 0}

        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['StdDev'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['SMA20'] + (df['StdDev'] * 2)
        df['Lower'] = df['SMA20'] - (df['StdDev'] * 2)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        optimal_entry = df['Lower'].iloc[-1]
        recent_low = df['Low'].tail(50).min()
        recent_high = df['High'].tail(50).max()
        sniper_metrics = {"entry_price": optimal_entry, "current_diff": ((current_p - optimal_entry)/current_p) * 100, "upper_band": df['Upper'].iloc[-1], "stop_loss": min(recent_low, optimal_entry) * 0.98, "take_profit": recent_high, "rr_ratio": (recent_high - optimal_entry) / (optimal_entry - (min(recent_low, optimal_entry) * 0.98)) if (optimal_entry - (min(recent_low, optimal_entry) * 0.98)) > 0 else 0}

        try:
            market = yf.Ticker("^GSPC").history(period="7y")
            if not market.empty:
                market_aligned = market['Close'].reindex(df.index, method='nearest')
                df['Market_Perf'] = (market_aligned / market_aligned.iloc[0]) * df['Close'].iloc[0]
                market_bull = market['Close'].iloc[-1] > market['Close'].rolling(200).mean().iloc[-1]
            else: df['Market_Perf'] = df['Close']; market_bull = True
        except: df['Market_Perf'] = df['Close']; market_bull = True

        metrics = {"name": info.get('longName', ticker), "price": current_p, "sma200": df['SMA200'].iloc[-1], "rsi": df['RSI'].iloc[-1], "market_bull": market_bull}
        return df, metrics, fundamentals, wall_street, None, sniper_metrics, None
    except Exception as e: return None, None, None, None, None, None, str(e)

def get_external_info(ticker):
    buys, news_results = 0, []
    try:
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is not None and not insider.empty: buys = insider.head(10)[insider.head(10)['Text'].str.contains("Purchase", case=False, na=False)].shape[0]
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+finance&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).content)
        for entry in feed.entries[:5]:
            sentiment = "NEUTRAL"
            if ai_pipe: 
                try: sentiment = ai_pipe(entry.title[:512])[0]['label'].upper()
                except: pass
            news_results.append({"title": entry.title, "sentiment": sentiment, "link": entry.link})
    except: pass
    return buys, news_results

def generate_thesis(ticker, metrics, buys, pos_news, fundamentals, wall_street, sniper):
    thesis = []
    signal_strength = "NEUTRAAL"
    is_uptrend = metrics['price'] > metrics['sma200']
    trend_text = "Trend is Bullish 🟢" if is_uptrend else "Trend is Bearish 🔴"
    dist_to_entry = ((metrics['price'] - sniper['entry_price']) / metrics['price']) * 100
    is_sniper_buy = dist_to_entry < 1.5
    sniper_text = "🎯 **TIMING:** Prijs op Support." if is_sniper_buy else f"⏳ **TIMING:** Wacht (-{dist_to_entry:.1f}%)."
    
    if is_uptrend and is_sniper_buy: thesis.append(f"🔥 **PERFECT:** {trend_text}. {sniper_text} 'Dip Buy'."); signal_strength = "STERK KOPEN"
    elif not is_uptrend and is_sniper_buy: thesis.append(f"⚠️ **RISICO:** {trend_text}, maar {sniper_text}. Speculatief."); signal_strength = "SPECULATIEF KOPEN"
    elif is_uptrend and not is_sniper_buy: thesis.append(f"✅ **HOUDEN:** {trend_text}. Wacht op entry."); signal_strength = "HOUDEN / WACHTEN"
    else: thesis.append(f"🛑 **AFBLIJVEN:** {trend_text}."); signal_strength = "AFBLIJVEN"
    if wall_street['upside'] > 15: thesis.append(f"Analisten zien {wall_street['upside']:.0f}% upside.")
    return " ".join(thesis), signal_strength

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

st.title("💎 Zenith Institutional Terminal") 
macro = get_macro_data()
cols = st.columns(5)
metrics = ["S&P 500", "Nasdaq", "Goud", "Olie", "10Y Rente"]
for i, m in enumerate(metrics):
    val, chg = macro.get(m, (0,0))
    cols[i].metric(m, f"{val:.2f}", f"{chg:.2f}%")
st.markdown("---")

if page == "🔎 Markt Analyse":
    col_input, col_cap = st.columns(2)
    with col_input: ticker_input = st.text_input("Ticker", value=st.session_state['selected_ticker']).upper()
    with col_cap: capital = st.number_input(f"Virtueel Kapitaal ({curr_symbol})", value=10000)
    auto_run = st.session_state.get('auto_run', False)
    if st.button("Start Deep Analysis") or auto_run:
        if auto_run: st.session_state['auto_run'] = False
        df, metrics, fund, wall_street, market_data, sniper, error_msg = get_zenith_data(ticker_input)
        
        if error_msg: st.error(f"⚠️ {error_msg}")
        elif df is not None:
            with st.spinner('Analyseren...'): buys, news = get_external_info(ticker_input)
            
            score = 0
            if metrics['price'] > metrics['sma200']: score += 25
            if metrics['rsi'] < 35: score += 20 
            elif metrics['rsi'] > 70: score -= 10
            if buys > 0: score += 15
            pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE')
            if pos_news >= 2: score += 10
            if 0 < fund['pe'] < 25: score += 10
            if wall_street['upside'] > 10: score += 15
            
            thesis_text, signal = generate_thesis(ticker_input, metrics, buys, pos_news, fund, wall_street, sniper)
            st.markdown(f"## 🏢 {metrics['name']} ({ticker_input})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Zenith Score", f"{score}/100")
            sig_color = "green" if "KOPEN" in signal else "orange" if "SPECULATIEF" in signal else "red" if "AFBLIJVEN" in signal else "blue"
            c2.markdown(f"**Advies:** :{sig_color}[{signal}]")
            c3.metric("Huidige Prijs", f"{curr_symbol}{metrics['price']:.2f}")
            c4.metric("Analisten Doel", f"{curr_symbol}{wall_street['target']:.2f}", f"{wall_street['upside']:.1f}% Upside")
            st.markdown("---")
            
            # --- TABS ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Sniper & Charts", "🔙 Backtest (Verification)", "🏢 Fundamenteel", "📰 Nieuws"])
            
            with tab1:
                st.subheader("🎯 Sniper Entry Setup")
                s1, s2, s3, s4 = st.columns(4)
                entry_msg = "✅ NU KOPEN!" if sniper['current_diff'] < 1.0 else f"Wacht (-{sniper['current_diff']:.1f}%)"
                s1.metric("1. Entry", f"{curr_symbol}{sniper['entry_price']:.2f}", entry_msg)
                s2.metric("2. Stop Loss", f"{curr_symbol}{sniper['stop_loss']:.2f}")
                s3.metric("3. Take Profit", f"{curr_symbol}{sniper['take_profit']:.2f}")
                rr_color = "green" if sniper['rr_ratio'] >= 2 else "orange"
                s4.markdown(f"**R/R Ratio:** :{rr_color}[1 : {sniper['rr_ratio']:.1f}]")
                st.info(f"**Zenith Thesis:** {thesis_text}")
                
                # Chart
                end_date = df.index[-1]; start_date = end_date - pd.DateOffset(years=1); plot_df = df.loc[start_date:end_date]
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Lower'], line=dict(color='rgba(0,255,0,0.3)'), name="Lower Band"), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Upper'], line=dict(color='rgba(255,0,0,0.3)'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name="Upper Band"), row=1, col=1)
                fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Prijs"), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#9370DB'), name="RSI"), row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
                fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("🔙 Strategy Verification (5 Jaar)")
                st.write("Wat als je de **Zenith Strategie** (Trend + RSI Dip) de afgelopen jaren had gebruikt?")
                if st.button("🚀 Draai Backtest"):
                    with st.spinner("Het verleden simuleren..."):
                        bt_res = run_backtest(ticker_input)
                        if bt_res:
                            c1, c2, c3 = st.columns(3)
                            # Kleuren voor winst/verlies
                            res_color = "green" if bt_res['return'] > bt_res['bh_return'] else "red"
                            c1.metric("Jouw Strategie", f"{curr_symbol}{bt_res['final_value']:.0f}", f"{bt_res['return']:.1f}%")
                            c2.metric("Buy & Hold (Markt)", f"{bt_res['bh_return']:.1f}%")
                            c3.metric("Aantal Trades", bt_res['trades'])
                            
                            if bt_res['return'] > bt_res['bh_return']:
                                st.success("✅ **Conclusie:** Deze strategie verslaat 'Buy & Hold' voor dit aandeel!")
                            else:
                                st.warning("⚠️ **Conclusie:** Voor dit aandeel werkt 'Gewoon vasthouden' beter dan traden.")
                            
                            # Equity Curve
                            st.line_chart(bt_res['history']['Close'])
                        else: st.error("Niet genoeg data voor backtest.")

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("P/E Ratio", f"{fund['pe']:.2f}")
                    st.metric("Dividend Yield", f"{fund['dividend']:.2f}%")
                with c2:
                    st.metric("Sector", fund['sector'])
                    st.metric("Winstmarge", f"{fund['profit_margin']:.1f}%")

            with tab4:
                for n in news:
                    color = "green" if n['sentiment'] == 'POSITIVE' else "red" if n['sentiment'] == 'NEGATIVE' else "gray"
                    st.markdown(f":{color}[**{n['sentiment']}**] | [{n['title']}]({n['link']})")
        else: st.error("Onbekende fout.")

elif page == "💼 Mijn Portfolio":
    st.title("💼 Portfolio Manager")
    with st.expander("➕ Aandeel Toevoegen", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1: p_ticker = st.text_input("Ticker", key="p_ticker").upper()
        with c2: p_amount = st.number_input("Aantal", min_value=0.0, step=1.0, key="p_amount")
        with c3: p_avg = st.number_input(f"Koopprijs ({curr_symbol})", min_value=0.0, step=0.1, key="p_avg")
        with c4:
            if st.button("Toevoegen"):
                if p_ticker and p_amount > 0:
                    st.session_state['portfolio'].append({"Ticker": p_ticker, "Aantal": p_amount, "Koopprijs": p_avg})
                    st.success("Toegevoegd!"); st.rerun()

    if len(st.session_state['portfolio']) > 0:
        st.markdown("---")
        portfolio_data = []
        total_value = 0; total_cost = 0
        tickers_in_port = []
        
        for item in st.session_state['portfolio']:
            current_price = get_current_price(item['Ticker'])
            cur_val = current_price * item['Aantal']
            cost_val = item['Koopprijs'] * item['Aantal']
            total_value += cur_val; total_cost += cost_val
            profit = cur_val - cost_val
            profit_pct = ((current_price - item['Koopprijs']) / item['Koopprijs']) * 100 if item['Koopprijs'] > 0 else 0
            color = "green" if profit >= 0 else "red"
            tickers_in_port.append(item['Ticker'])
            
            portfolio_data.append({
                "Ticker": item['Ticker'], "Aantal": item['Aantal'], "Waarde": f"{curr_symbol}{cur_val:.2f}",
                "Winst": f":{color}[{curr_symbol}{profit:.2f} ({profit_pct:.1f}%)]"
            })
        
        st.write(pd.DataFrame(portfolio_data).to_markdown(index=False), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Totale Waarde", f"{curr_symbol}{total_value:.2f}")
        m2.metric("Totale Inleg", f"{curr_symbol}{total_cost:.2f}")
        tot_pct = ((total_value - total_cost)/total_cost)*100 if total_cost > 0 else 0
        m3.metric("Totaal Winst", f"{curr_symbol}{total_value - total_cost:.2f}", f"{tot_pct:.1f}%")
        
        if st.button("🗑️ Wissen"): st.session_state['portfolio'] = []; st.rerun()
        
        # --- PORTFOLIO OPTIMALISATIE (NIEUW) ---
        st.markdown("---")
        st.subheader("⚖️ AI Portfolio Optimalisatie")
        st.write("De wiskunde (Efficient Frontier) zegt dat dit je ideale verdeling is:")
        
        if len(tickers_in_port) >= 2:
            if st.button("🧮 Bereken Optimale Mix"):
                with st.spinner("Markowitz model aan het draaien..."):
                    opt_weights = optimize_portfolio(tickers_in_port)
                    if opt_weights:
                        opt_df = pd.DataFrame(list(opt_weights.items()), columns=['Ticker', 'Ideaal %'])
                        opt_df['Ideaal %'] = (opt_df['Ideaal %'] * 100).map('{:.1f}%'.format)
                        
                        # Huidige verdeling berekenen
                        current_weights = {}
                        for t in tickers_in_port:
                            val = next((item['Aantal'] * get_current_price(t) for item in st.session_state['portfolio'] if item['Ticker'] == t), 0)
                            current_weights[t] = (val / total_value) * 100
                        
                        # Plot
                        fig = go.Figure(data=[
                            go.Bar(name='Huidig', x=list(current_weights.keys()), y=list(current_weights.values())),
                            go.Bar(name='Optimaal (AI)', x=list(opt_weights.keys()), y=[v*100 for v in opt_weights.values()])
                        ])
                        fig.update_layout(barmode='group', template='plotly_dark', title="Huidig vs Optimaal")
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.error("Optimalisatie mislukt.")
        else: st.info("Voeg minimaal 2 aandelen toe om te optimaliseren.")

    else: st.write("Leeg.")

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
            try:
                df, metrics, fund, ws, _, sniper, _ = get_zenith_data(ticker)
                if df is not None:
                    buys, news = get_external_info(ticker)
                    score = 0
                    if metrics['market_bull']: score += 15
                    if metrics['price'] > metrics['sma200']: score += 20
                    if metrics['rsi'] < 30: score += 15
                    if buys > 0: score += 15
                    pos_news = sum(1 for n in news if n['sentiment'] == 'POSITIVE'); 
                    if pos_news >= 2: score += 10
                    if 0 < fund['pe'] < 25: score += 10
                    if ws['upside'] > 10: score += 15
                    if sniper and sniper['rr_ratio'] > 2: score += 10
                    advies = "NEUTRAAL"
                    if score >= 70: advies = "🟢 STERK KOPEN"
                    elif score >= 50: advies = "🟡 KOPEN / HOUDEN"
                    else: advies = "🔴 AFBLIJVEN"
                    results.append({"Ticker": ticker, "Prijs": metrics['price'], "Score": score, "Advies": advies})
            except: continue
        my_bar.progress(1.0, text="Klaar!")
        if results: st.session_state['scan_results'] = results 
        else: st.error("Geen data.")

    if 'scan_results' in st.session_state and st.session_state['scan_results']:
        results = st.session_state['scan_results']
        df_scan = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        st.dataframe(df_scan, use_container_width=True, hide_index=True)
        st.download_button("📥 Download (CSV)", df_scan.to_csv(index=False), "results.csv", "text/csv")
        st.markdown("---")
        c1, c2 = st.columns([3, 1])
        options = [r['Ticker'] for r in results]
        selected_scan = c1.selectbox("Kies:", options)
        c2.button("🚀 Analyseer Nu", on_click=start_analysis_for, args=(selected_scan,))

st.markdown("---")
st.markdown("© 2025 Zenith Terminal | Built by [Warre Van Rechem](https://www.linkedin.com/in/warre-van-rechem-928723298/)")
