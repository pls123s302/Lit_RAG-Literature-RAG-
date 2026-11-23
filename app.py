"""Streamlit 入口：科研文献管理助手（多模态 RAG）。"""
import streamlit as st

st.set_page_config(page_title="科研文献管理助手", page_icon="📚", layout="wide")

pg = st.navigation([
    st.Page("ui/page_chat.py", title="💬 文献问答", icon="💬", default=True),
    st.Page("ui/page_kb.py", title="📁 文献库管理", icon="📁"),
    st.Page("ui/page_eval.py", title="📊 效果测评", icon="📊"),
])
pg.run()
