"""效果测评页：benchmark 生成 + 分类别测评报告 + 策略对比。"""
import json
import streamlit as st

from src import config
from src.evaluation.runner import load_testset, run_evaluation, save_testset

st.title("📊 效果测评")

st.markdown("基于文献库自动生成分类 benchmark（事实/公式/拒答/闲聊/多跳等 10 类场景），"
            "运行后输出 RAGAS 四指标 + 自动断言（路由/拒答/LaTeX/章节覆盖），按类别定位系统短板。")

# ---- Benchmark 生成 ----
with st.expander("🧰 基准测试集", expanded=not load_testset()):
    st.caption("从文献库已索引内容自动出题（每篇论文出事实/公式/口语化/对比/多跳/综合题，"
               "全局出拒答/闲聊/综述题）。生成约需 1-2 分钟。")
    c1, c2 = st.columns([1, 3])
    if c1.button("生成 benchmark", disabled=not config.API_KEY):
        from src.evaluation.benchmark import generate_benchmark
        bar = st.progress(0.0, text="扫描文献库…")
        def cb(done, total, msg):
            bar.progress(min(done / max(total, 1), 1.0), text=f"[{done}/{total}] {msg}")
        with st.spinner("基于文献内容出题中…"):
            ts = generate_benchmark(progress_cb=cb)
        bar.empty()
        if ts:
            st.success(f"生成 {len(ts)} 条测试集，已保存。")
        else:
            st.error("生成失败：请确认文献库中已索引至少一篇论文。")
        st.rerun()
    if c2.button("🗑️ 清空测试集"):
        save_testset([])
        st.rerun()

testset = load_testset()
cats = {}
for it in testset:
    cats[it.get("category", "通用")] = cats.get(it.get("category", "通用"), 0) + 1
st.info(f"当前测试集：**{len(testset)} 条**" +
        ("" if not cats else " ｜ " + " ｜ ".join(f"{k}×{v}" for k, v in sorted(cats.items()))))

with st.expander("查看/编辑测试集明细", expanded=False):
    for i, item in enumerate(testset):
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"**[{item.get('category', '通用')}]** {item['question']}")
        if c2.button("删", key=f"del_{i}"):
            testset.pop(i)
            save_testset(testset)
            st.rerun()

# ---- 策略选择（可插拔对比实验） ----
from src.pipeline import available, PIPELINE_DEFAULTS
plugins = available()
with st.expander("🧪 链路策略（可插拔对比实验）"):
    st.caption("同一测试集在不同策略下跑分对比。切分策略变更后需重新索引（勾选强制重建）。")
    sel = {}
    for stage, options in plugins.items():
        default = PIPELINE_DEFAULTS.get(stage, options[0] if options else None)
        sel[stage] = st.selectbox(stage, options,
                                  index=options.index(default) if default in options else 0,
                                  key=f"pl_{stage}")
    force_reindex = st.checkbox("切分策略变更后强制重建索引", key="pl_force")

override = {k: v for k, v in sel.items() if v != PIPELINE_DEFAULTS.get(k)}

# ---- 运行测评 ----
if st.button("🚀 运行测评", disabled=not testset, type="primary"):
    if force_reindex and override.get("chunker"):
        from pathlib import Path
        from src.ingestion.indexer import index_file
        from src.retrieval.hybrid import HybridRetriever
        with st.spinner("切分策略变更，强制重建索引…"):
            for f in Path(config.KB_DIR).glob("*.pdf"):
                index_file(f, force=True)
            HybridRetriever().rebuild()

    progress = st.progress(0.0, text="准备运行…")

    def cb(done, total, detail):
        progress.progress(done / total, text=f"[{done}/{total}] {detail['question'][:30]}…")

    with st.spinner("测评运行中（每条包含完整检索链路 + LLM 裁判打分，需要一些时间）"):
        report = run_evaluation(testset, progress_cb=cb, pipeline=override or None)
    progress.empty()
    st.session_state.report = report

# ---- 历史报告 ----
st.divider()
st.subheader("历史报告")
reports = sorted(config.EVAL_DIR.glob("report_*.json"), reverse=True)
if not reports:
    st.info("暂无测评报告。")
for rp in reports[:5]:
    data = json.loads(rp.read_text(encoding="utf-8"))
    with st.expander(f"{data['timestamp']}（{data['count']} 条）"):
        s = data["summary"]
        cols = st.columns(4)
        for col, (k, v) in zip(cols, s.items()):
            col.metric(k, f"{v:.1%}" if isinstance(v, float) else "—")
        by_cat = data.get("by_category")
        if by_cat:
            st.markdown("**分类别结果**")
            rows = []
            for cat, c in by_cat.items():
                rows.append({
                    "类别": cat, "题数": c["n"],
                    "faithfulness": c["metrics"].get("faithfulness"),
                    "relevancy": c["metrics"].get("answer_relevancy"),
                    "precision": c["metrics"].get("context_precision"),
                    "recall": c["metrics"].get("context_recall"),
                    "路由✓": f"{c['route_ok']}/{c['n']}",
                    "拒答✓": f"{c['answer_ok']}/{c['n']}",
                    "LaTeX✓": f"{c['latex_ok']}/{c['n']}",
                    "章节✓": f"{c['section_ok']}/{c['n']}",
                    "均延迟": f"{c['avg_latency_ms']}ms",
                })
            st.dataframe(rows, use_container_width=True)
        if data.get("pipeline"):
            st.caption(f"策略覆盖: {data['pipeline']}")
