import streamlit as st
from ui.metrics import market_kpis
from ui.charts import technical_chart


def market_page(df, price, rsi, score, regime):
    st.markdown("## 🔎 Markt Analyse")
    market_kpis(price, rsi, score, regime)
    st.plotly_chart(technical_chart(df), use_container_width=True)


def portfolio_page():
    st.markdown("## 💼 Portfolio")
    st.info("Portfolio-optimalisatie volgt")


def risk_page():
    st.markdown("## ⚠️ Risk Monitor")
    st.info("VaR, Drawdown & Monte Carlo volgen")


def education_page():
    st.markdown("## 🎓 Educatie")
    st.write("Uitlegmodules volgen")
