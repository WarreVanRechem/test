import streamlit as st
from ui.metrics import kpi_row
from ui.charts import price_chart

def market_page(df, price, rsi, score, regime):
    st.subheader("📊 Market Analysis")

    kpi_row(price, rsi, score, regime)

    st.plotly_chart(price_chart(df), use_container_width=True)
