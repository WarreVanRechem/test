import streamlit as st
from data.market import get_price_history
from data.fundamentals import get_fundamentals
from analysis.valuation import graham_number, dcf_value
from analysis.technicals import indicators, regime
from analysis.risk import max_drawdown
from analysis.scoring import investment_score


st.set_page_config(layout="wide")
st.title("Zenith Institutional Terminal")


ticker = st.text_input("Ticker", "AAPL")


if ticker:
    df = get_price_history(ticker)
    if df is not None:
        df = indicators(df)
        f = get_fundamentals(ticker)
        fair = graham_number(f['eps'], f['book'])
        score = investment_score(df['Close'].iloc[-1], fair or df['Close'].iloc[-1], df['RSI'].iloc[-1], max_drawdown(df))
        st.metric("Investment Score", score)
        st.line_chart(df['Close'])