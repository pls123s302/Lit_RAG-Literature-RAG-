"""Streamlit 入口：科研文献管理助手（多模态 RAG）。"""
import streamlit as st

st.set_page_config(page_title="LitRAG · 科研文献管理助手", page_icon="📚",
                   layout="wide", initial_sidebar_state="expanded")

from ui.style import inject_css

inject_css()

pg = st.navigation([
    st.Page("ui/page_chat.py", title="文献问答", icon="💬", default=True),
    st.Page("ui/page_idea.py", title="Idea 实验室", icon="💡"),
    st.Page("ui/page_kb.py", title="文献库管理", icon="📁"),
    st.Page("ui/page_eval.py", title="效果测评", icon="📊"),
])
pg.run()
