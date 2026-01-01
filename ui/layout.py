import streamlit as st


def render_sidebar():
    st.sidebar.title("💎 Zenith Terminal")

    page = st.sidebar.radio(
        "Navigatie",
        ["🔎 Markt Analyse", "💼 Portfolio", "⚠️ Risk Monitor", "🎓 Educatie"]
    )

    ticker = st.sidebar.text_input("Ticker", "AAPL")
    capital = st.sidebar.number_input("Kapitaal (€)", 10_000)

    st.sidebar.markdown("---")
    st.sidebar.error("⚠️ Geen financieel advies")

    return page, ticker.upper(), capital
