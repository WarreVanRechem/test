import streamlit as st

from ui.layout import render_sidebar
from ui.pages import (
    market_page,
    portfolio_page,
    risk_page,
    education_page,
)

from data.market import get_price_history
from data.fundamentals import get_fundamentals
from analysis.valuation import graham_number
from analysis.technicals import add_indicators, market_regime
from analysis.risk import max_drawdown
from analysis.scoring import investment_score


st.set_page_config(
    page_title="Zenith Institutional Terminal",
    layout="wide",
    page_icon="💎",
)

page, ticker, capital = render_sidebar()

df = get_price_history(ticker)

if df is None:
    st.error("Geen marktdata beschikbaar")
    st.stop()

df = add_indicators(df)
fundamentals = get_fundamentals(ticker)

price = df["Close"].iloc[-1]
rsi = df["RSI"].iloc[-1]
regime = market_regime(price, df["SMA200"].iloc[-1], rsi)

fair = graham_number(
    fundamentals.get("eps"),
    fundamentals.get("book"),
)

score = investment_score(
    price=price,
    fair_value=fair,
    rsi=rsi,
    drawdown=max_drawdown(df),
)

if page == "🔎 Markt Analyse":
    market_page(df, price, rsi, score, regime)
elif page == "💼 Portfolio":
    portfolio_page()
elif page == "⚠️ Risk Monitor":
    risk_page()
elif page == "🎓 Educatie":
    education_page()
