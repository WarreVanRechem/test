import streamlit as st


def market_kpis(price, rsi, score, regime):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Prijs", f"€{price:.2f}")
    c2.metric("RSI (14)", f"{rsi:.1f}")
    c3.metric("Investment Score", f"{score}/100")
    c4.metric("Marktregime", regime)
