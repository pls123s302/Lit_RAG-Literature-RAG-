"""全局视觉样式：学术靛蓝主题，一次注入全页生效。"""
import streamlit as st

_CSS = """
<style>
/* ===== 全局 ===== */
html, body, [class*="css"] { font-family: "PingFang SC", "Microsoft YaHei", "Source Han Sans SC", system-ui, sans-serif; }
.stApp { background: linear-gradient(180deg, #f6f7fb 0%, #eef1f8 100%); }
.block-container { padding-top: 1.2rem; max-width: 1200px; }
h1 { font-weight: 800; letter-spacing: .5px; }
h1::after { content: ""; display: block; width: 56px; height: 4px; margin-top: 6px;
            border-radius: 2px; background: linear-gradient(90deg, #6366f1, #8b5cf6); }

/* ===== 侧边栏 ===== */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8ebf4; }
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
section[data-testid="stSidebar"] h2 { font-size: 1rem; color: #475569; letter-spacing: 1px; }

/* ===== 聊天 ===== */
[data-testid="stChatMessage"] { background: #ffffff; border: 1px solid #e8ebf4;
    border-radius: 16px; padding: 10px 16px; margin-bottom: 10px;
    box-shadow: 0 2px 12px rgba(30, 41, 59, .05); }
[data-testid="stChatInput"] > div { border-radius: 14px !important; border: 1.5px solid #d8ddf0 !important; }
[data-testid="stChatInput"] > div:focus-within { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,.15); }

/* ===== 按钮 ===== */
button[kind="primary"] { background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none;
    border-radius: 12px; font-weight: 600; box-shadow: 0 4px 14px rgba(99,102,241,.35); }
button[kind="primary"]:hover { filter: brightness(1.08); }
button[kind="primaryFormSubmit"] { background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none; border-radius: 12px; }
button[kind="secondary"], button[kind="secondaryFormSubmit"] { border-radius: 12px;
    border: 1px solid #d8ddf0; background: #fff; }
div[data-testid="stButton"] > button { transition: all .15s ease; }
div[data-testid="stButton"] > button:hover { transform: translateY(-1px); }

/* ===== Metric 卡片 ===== */
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e8ebf4; border-radius: 14px;
    padding: 14px 18px; box-shadow: 0 2px 12px rgba(30,41,59,.05); }

/* ===== Expander ===== */
details[data-testid="stExpander"] { background: #ffffff; border: 1px solid #e8ebf4;
    border-radius: 14px; box-shadow: 0 1px 8px rgba(30,41,59,.04); }

/* ===== 上传区 ===== */
[data-testid="stFileUploaderDropzone"] { border-radius: 14px; border: 2px dashed #c9cff2;
    background: #fbfbfe; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #6366f1; background: #f4f4ff; }

/* ===== 状态/提示 ===== */
[data-testid="stStatus"] { border-radius: 12px; border: 1px solid #e8ebf4; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stProgress"] { height: 8px; border-radius: 4px; }

/* ===== 欢迎页 ===== */
.hero-title { font-size: 1.9rem; font-weight: 800; text-align: center; margin: 2.2rem 0 .3rem;
    background: linear-gradient(90deg, #4f46e5, #8b5cf6, #6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { text-align: center; color: #64748b; font-size: .95rem; margin-bottom: 1.6rem; }
.chip-btn > button { width: 100%; text-align: left; background: #fff; border: 1px solid #e4e8f5;
    border-radius: 12px; padding: 12px 16px; color: #334155; font-size: .92rem;
    box-shadow: 0 1px 6px rgba(30,41,59,.04); height: auto; white-space: normal; }
.chip-btn > button:hover { border-color: #6366f1; color: #4f46e5; background: #f7f7ff; }

/* ===== 引用卡片 ===== */
.cite-card { background: #f8f9ff; border-left: 3px solid #6366f1; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 8px; font-size: .9rem; color: #334155; }
.cite-card b { color: #4f46e5; }
.cite-card .cite-loc { color: #94a3b8; font-size: .82rem; margin-left: 6px; }
.cite-card blockquote { margin: 6px 0 0; padding-left: 10px; border-left: 2px solid #dbe0f5;
    color: #64748b; }

/* ===== 滚动条 ===== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #cdd3e8; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #aeb6d8; }
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)
