"""问答页：对话 + 引用来源 + 执行轨迹 + 流式生成 + LaTeX 公式渲染。"""
import re

import streamlit as st

from src import config

if not config.API_KEY:
    st.error("未检测到 DASHSCOPE_API_KEY，请在项目根目录 .env 中配置：DASHSCOPE_API_KEY=sk-xxx")
    st.stop()

st.title("📚 文献问答")

if "history" not in st.session_state:
    st.session_state.history = []


def render_rich(text: str):
    """渲染含 LaTeX 公式的回答：$$...$$ 独立公式块走 st.latex，其余走 markdown（KaTeX 行内公式）。"""
    for part in re.split(r"(\$\$.+?\$\$)", text, flags=re.S):
        part = part.strip()
        if not part:
            continue
        if part.startswith("$$") and part.endswith("$$"):
            st.latex(part[2:-2].strip())
        else:
            st.markdown(part)


def render_attachments(msg: dict):
    if msg.get("sources"):
        with st.expander(f"📎 引用来源（{len(msg['sources'])}）"):
            for s in msg["sources"]:
                st.markdown(f"**[来源{s['id']}]** 《{s['source']}》 {s['location']}\n\n> {s['snippet']}")
    if msg.get("trace"):
        with st.expander("🔍 执行轨迹"):
            st.markdown(" → ".join(msg["trace"]) + f" ⏱ {msg.get('latency_ms', 0)}ms")


# 检索参数侧边栏
with st.sidebar:
    st.subheader("检索设置")
    config.RERANK_TOP_K = st.slider("送入生成的证据块数", 1, 10, config.RERANK_TOP_K)
    use_cache = st.toggle("启用分层缓存", True)
    if st.button("🗑️ 清空缓存"):
        from src.cache.layered_cache import clear_cache
        clear_cache()
        st.toast("缓存已清空")
    st.caption(f"build {config.UI_BUILD}")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        render_rich(msg["content"])
        render_attachments(msg)

question = st.chat_input("向文献库提问，如：AKT 的注意力机制是怎么设计的？…")
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
                ph.markdown("".join(buf))  # 实时流式渲染
            result = st.session_state.pop("_rag_final", {})
            streamed = "".join(buf)
        else:
            from src.evaluation.runner import run_without_cache
            result = run_without_cache(question)
            streamed = ""
        status.update(label=f"完成 ⏱ {result.get('latency_ms', 0)}ms", state="complete", expanded=False)

        # 清掉流式占位，只保留 render_rich 的最终排版（公式块居中渲染，且不会重复显示）
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
