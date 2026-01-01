import streamlit as st

def render_sidebar():
    st.sidebar.error("⚠️ Geen financieel advies.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2026 Zenith Terminal")

    page = st.sidebar.radio(
        "Ga naar:",
        ["🔎 Markt Analyse", "💼 Mijn Portfolio", "🎓 Leer de Basics"],
        key="nav_page"
    )

    currency = st.sidebar.radio("Valuta", ["USD", "EUR"])
    return page, "€" if currency == "EUR" else "$"
