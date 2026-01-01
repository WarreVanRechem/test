import streamlit as st

def sidebar():
    st.sidebar.title("💎 Zenith Terminal")

    page = st.sidebar.radio(
        "Navigatie",
        ["Market Analysis", "Portfolio", "Risk Monitor", "Education"]
    )

    ticker = st.sidebar.text_input("Ticker", "AAPL")
    capital = st.sidebar.number_input("Kapitaal (€)", 10_000)

    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ Geen financieel advies")

    return page, ticker.upper(), capital
