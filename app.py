import config
import streamlit as st
from session import init_session
from ui.sidebar import render_sidebar
from ui.market_page import render_market_page
from data.macro import get_macro_data

init_session()

page, curr_sym = render_sidebar()

st.title("💎 Zenith Institutional Terminal")

mac = get_macro_data()
cols = st.columns(5)
for i, k in enumerate(mac):
    v, ch = mac[k]
    cols[i].metric(k, f"{v:.2f}", f"{ch:.2f}%")

st.markdown("---")

if page == "🔎 Markt Analyse":
    render_market_page(curr_sym)
