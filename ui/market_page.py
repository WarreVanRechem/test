import streamlit as st
from data.market import get_zenith_data
from analysis.fundamentals import get_financial_trends
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_market_page(curr_sym):
    c1, c2 = st.columns(2)
    with c1:
        tick = st.text_input("Ticker", value=st.session_state['selected_ticker']).upper()
    with c2:
        st.number_input(f"Kapitaal ({curr_sym})", 10000)

    if st.button("Start Deep Analysis"):
        st.session_state['analysis_active'] = True
        st.session_state['selected_ticker'] = tick

    if not st.session_state['analysis_active']:
        return

    df, met, fund, ws, _, snip, _, err, _ = get_zenith_data(tick)
    if err:
        st.error(err)
        return

    st.markdown(f"## 🏢 {met['name']} ({tick})")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prijs", f"{curr_sym}{met['price']:.2f}")
    k2.metric("RSI", f"{met['rsi']:.1f}")
    if fund['fair_value']:
        k3.metric("Fair Value", f"{curr_sym}{fund['fair_value']:.2f}")
    k4.metric("Upside", f"{ws['upside']:.1f}%")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close']
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI']), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)
import streamlit as st
from data.market import get_zenith_data
from analysis.fundamentals import get_financial_trends
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_market_page(curr_sym):
    c1, c2 = st.columns(2)
    with c1:
        tick = st.text_input("Ticker", value=st.session_state['selected_ticker']).upper()
    with c2:
        st.number_input(f"Kapitaal ({curr_sym})", 10000)

    if st.button("Start Deep Analysis"):
        st.session_state['analysis_active'] = True
        st.session_state['selected_ticker'] = tick

    if not st.session_state['analysis_active']:
        return

    df, met, fund, ws, _, snip, _, err, _ = get_zenith_data(tick)
    if err:
        st.error(err)
        return

    st.markdown(f"## 🏢 {met['name']} ({tick})")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prijs", f"{curr_sym}{met['price']:.2f}")
    k2.metric("RSI", f"{met['rsi']:.1f}")
    if fund['fair_value']:
        k3.metric("Fair Value", f"{curr_sym}{fund['fair_value']:.2f}")
    k4.metric("Upside", f"{ws['upside']:.1f}%")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close']
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI']), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)
