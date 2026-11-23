"""文献问答页：欢迎页 + 流式生成 + 引用卡片 + LaTeX 公式渲染。"""
import html
import re

import streamlit as st

from src import config

if not config.API_KEY:
    st.error("未检测到 DASHSCOPE_API_KEY，请在项目根目录 .env 中配置：DASHSCOPE_API_KEY=sk-xxx")
    st.stop()

st.title("📚 文献问答")

if "history" not in st.session_state:
    st.session_state.history = []

EXAMPLES = [
    "AKT 的核心创新点有哪些？",
    "注意力权重的衰减公式具体是什么？",
    "AKT 相比 DKT baseline 提升了多少？",
    "各篇论文分别解决了知识追踪的什么问题？",
    "csKT 是怎么解决冷启动问题的？",
    "DiffuQKT 的实验结果在哪些数据集上验证？",
]


def render_rich(text: str):
    """渲染含 LaTeX 公式的回答：$$...$$ 独立公式块走 st.latex，其余走 markdown。"""
    for part in re.split(r"(\$\$.+?\$\$)", text, flags=re.S):
        part = part.strip()
        if not part:
            continue
        if part.startswith("$$") and part.endswith("$$"):
            st.latex(part[2:-2].strip())
        else:
            st.markdown(part)


def render_citations(sources: list):
    """引用来源渲染为卡片。"""
    if not sources:
        return
    cards = []
    for s in sources:
        loc = html.escape(str(s.get("location", "")))
        src = html.escape(str(s.get("source", "")))
        snippet = html.escape(str(s.get("snippet", "")))
        cards.append(
            f'<div class="cite-card"><b>[{s["id"]}]</b> {src}'
            f'<span class="cite-loc">· {loc}</span>'
            f"<blockquote>{snippet}…</blockquote></div>")
    with st.expander(f"📎 引用来源（{len(sources)}）"):
        st.markdown("".join(cards), unsafe_allow_html=True)


def render_attachments(msg: dict):
    if msg.get("sources"):
        render_citations(msg["sources"])
    if msg.get("trace"):
        with st.expander("🔍 执行轨迹"):
            st.markdown(" → ".join(msg["trace"]) + f" ⏱ {msg.get('latency_ms', 0)}ms")


# 侧边栏
with st.sidebar:
    st.subheader("检索设置")
    config.RERANK_TOP_K = st.slider("送入生成的证据块数", 1, 10, config.RERANK_TOP_K)
    use_cache = st.toggle("启用分层缓存", True)
    if st.button("🗑️ 清空缓存"):
        from src.cache.layered_cache import clear_cache
        clear_cache()
        st.toast("缓存已清空")
    st.caption(f"build {config.UI_BUILD}")

# 欢迎页（无对话历史时展示示例问题，点击即问）
if not st.session_state.history:
    st.markdown('<div class="hero-title">📚 LitRAG 科研文献管理助手</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">基于文献库的 5 篇知识追踪论文 · 支持公式 LaTeX 渲染 · 答案全程标注引用来源'
                '</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="small")
    for i, ex in enumerate(EXAMPLES):
        with (c1 if i % 2 == 0 else c2):
            if st.button(f"💬 {ex}", key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_q = ex
                st.rerun()

question = st.chat_input("向文献库提问，或点击上方示例问题…")
if not question and st.session_state.get("pending_q"):
    question = st.session_state.pop("pending_q")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("检索与生成中…", expanded=True)
        ph = st.empty()  # 流式占位（单元素容器，clear 语义可靠）
        if use_cache:
            from src.graph.rag_graph import ask_stream
            buf = []
            for token in ask_stream(question):
                buf.append(token)
                ph.markdown("".join(buf))
            result = st.session_state.pop("_rag_final", {})
        else:
            from src.evaluation.runner import run_without_cache
            result = run_without_cache(question)
            streamed = ""
        status.update(label=f"完成 ⏱ {result.get('latency_ms', 0)}ms", state="complete", expanded=False)

        # 清掉流式占位后统一用 render_rich 重渲染（公式排版正确，且避免内容重复）
        answer = result.get("answer") or streamed
        ph.empty()
        render_rich(answer)
        if result.get("cache_hit"):
            st.info(f"⚡ 缓存命中（{result['cache_hit']}层），{result['latency_ms']}ms")
        render_attachments({"sources": result.get("sources"), "trace": result.get("trace"),
                            "latency_ms": result.get("latency_ms")})

    st.session_state.history.append({
        "role": "assistant", "content": answer,
        "sources": result.get("sources"), "trace": result.get("trace"),
        "latency_ms": result.get("latency_ms"),
    })
