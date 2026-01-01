import streamlit as st

def init_session():
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = []
    if 'nav_page' not in st.session_state:
        st.session_state['nav_page'] = "🔎 Markt Analyse"
    if 'selected_ticker' not in st.session_state:
        st.session_state['selected_ticker'] = "RDW"
    if 'analysis_active' not in st.session_state:
        st.session_state['analysis_active'] = False

def start_analysis_for(ticker):
    st.session_state['selected_ticker'] = ticker
    st.session_state['nav_page'] = "🔎 Markt Analyse"
    st.session_state['analysis_active'] = True

def reset_analysis():
    st.session_state['analysis_active'] = False
