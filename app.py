import streamlit as st

from ui.layout import sidebar
from ui.pages import market_page

from data.market import get_price_history
from data.fundamentals import get_fundamentals
from analysis.valuation import graham_number
from analysis.technicals import add_indicators, market_regime
from analysis.risk import max_drawdown
from analysis.scoring import investment_score

st.set_page_config(layout="wide")

page, ticker, capital = sidebar()

df = get_price_history(ticker)

if df is not None:
    df = add_indicators(df)

    fundamentals = get_fundamentals(ticker)
    fair = graham_number(
        fundamentals.get("eps"),
        fundamentals.get("book")
    )

    price = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    regime = market_regime(price, df['SMA200'].iloc[-1], rsi)

    score = investment_score(price, fair, rsi, max_drawdown(df))

    if page == "Market Analysis":
        market_page(df, price, rsi, score, regime)
