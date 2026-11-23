"""💡 Idea 实验室：跨论文 Idea 融合，全自动产出研究提案。"""
import html
import json
import streamlit as st

from src import config
from src.graph.idea_graph import run_idea_fusion
from src.ingestion.indexer import list_indexed_docs

st.title("💡 Idea 实验室")

st.markdown("自动提取每篇文献的核心创新与**局限**，通过机制组合 · 问题迁移 · 矛盾驱动 · 局限互补"
            "四种策略融合生成新研究 idea，经文献库查新与三维度评审后输出研究提案。")

docs = list_indexed_docs()
if not docs:
    st.info("文献库为空，请先在「文献库管理」上传论文。")
    st.stop()

# ---- 论文选择 ----
with st.expander("🧪 参与融合的论文（默认全选）", expanded=True):
    selected = []
    for name in sorted(docs):
        if st.checkbox(name.replace(".pdf", ""), value=True, key=f"pp_{name}"):
            selected.append(name)

c_run, c_hint = st.columns([1, 3])
run_clicked = c_run.button("🚀 生成研究提案", type="primary",
                           disabled=len(selected) < 2 or not config.API_KEY)
c_hint.caption("至少选择 2 篇论文；全流程约 2-4 分钟（提取 → 融合 → 查新 → 评审）。"
               "新颖性判断仅基于库内文献，正式科研请另行外部查新。")

if run_clicked:
    status = st.status("Idea 融合引擎运行中…", expanded=True)
    stage_text = st.empty()

    def cb(msg):
        stage_text.markdown(f"**当前阶段：** {msg}")

    with st.spinner("融合引擎运行中（提取 → 融合 → 查新 → 评审）"):
        result = run_idea_fusion(selected, progress_cb=cb)
    status.update(label="完成", state="complete", expanded=False)
    st.session_state.proposals = result["proposals"]
    st.session_state.fusion_trace = result["trace"]

# ---- 展示提案 ----
proposals = st.session_state.get("proposals", [])
if proposals:
    st.divider()
    st.subheader(f"🏆 研究提案（{len(proposals)} 个，按综合评分排序）")

    _VERDICT_COLOR = {"库内未见相似工作": "#16a34a", "存在部分相似": "#d97706",
                      "高度重合": "#dc2626", "查新失败": "#94a3b8"}

    for i, p in enumerate(proposals, 1):
        sc = p.get("scores", {})
        nv = p.get("novelty", {})
        color = _VERDICT_COLOR.get(nv.get("verdict", ""), "#64748b")
        with st.container(border=True):
            top = st.columns([5, 2, 2, 2])
            top[0].markdown(f"### {i}. {p.get('name', '未命名 idea')}")
            top[1].metric("新颖性", f"{sc.get('novelty', 0)}/5")
            top[2].metric("可行性", f"{sc.get('feasibility', 0)}/5")
            top[3].metric("影响力", f"{sc.get('impact', 0)}/5")

            st.markdown(
                f'<span style="background:{color}1a;color:{color};padding:3px 10px;'
                f'border-radius:10px;font-size:.85rem;">🔍 {nv.get("verdict", "")}</span> '
                f'<span style="background:#eef2ff;color:#4f46e5;padding:3px 10px;'
                f'border-radius:10px;font-size:.85rem;margin-left:6px;">'
                f'🧬 {p.get("strategy", "")}</span>',
                unsafe_allow_html=True)
            if nv.get("analysis"):
                st.caption(f"查新分析：{nv['analysis']}（基于: {', '.join(nv.get('similar_chunks', []))}）")

            st.markdown(f"**💡 动机**\n\n{p.get('motivation', '')}")
            st.markdown(f"**🔬 技术路线**\n\n{p.get('method_sketch', '')}")
            st.markdown(f"**🧪 实验方案**\n\n{p.get('experiment_plan', '')}")
            if p.get("risks"):
                st.markdown(f"**⚠️ 风险**\n\n{p['risks']}")
            st.caption("来源论文：" + "、".join(p.get("source_papers", [])))
            if sc.get("comment"):
                st.caption(f"评审意见：{sc['comment']}")

    # 导出实施级文档（逐条 LLM 详化为编码 Agent 可执行的 technical spec）
    if st.button("📄 生成实施级提案文档（.md）"):
        from src.graph.idea_graph import export_proposals_md
        bar = st.progress(0.0, text="准备详化…")

        def cb(i, total, name):
            bar.progress(i / total, text=f"详化中 [{i}/{total}] {name[:36]}…")

        with st.spinner("把提案扩写为编码 Agent 可实施的详细技术方案…"):
            md = export_proposals_md(proposals, detail=True, progress_cb=cb)
        bar.empty()
        st.session_state.proposal_md = md
        st.toast("文档已生成")

    if st.session_state.get("proposal_md"):
        st.download_button("⬇️ 下载提案文档", data=st.session_state.proposal_md.encode("utf-8"),
                           file_name="research_proposals.md", mime="text/markdown")
        with st.expander("预览文档"):
            st.markdown(st.session_state.proposal_md)

    with st.expander("🔍 引擎执行轨迹"):
        st.write(" → ".join(st.session_state.get("fusion_trace", [])))
