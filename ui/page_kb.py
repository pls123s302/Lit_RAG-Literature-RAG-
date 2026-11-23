"""知识库管理页：上传文档、增量索引、删除、统计。"""
import shutil
import streamlit as st

from src import config
from src.ingestion.indexer import get_vectorstore, index_file, list_indexed_docs, remove_file
from src.retrieval.hybrid import HybridRetriever

st.title("📁 文献库管理")

st.markdown(f"文献库目录：`{config.KB_DIR}`（支持论文 PDF / Markdown 笔记 / TXT / 公式与图表截图）")

uploaded = st.file_uploader(
    "上传文献", type=["pdf", "md", "markdown", "txt", "png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=True,
)

if uploaded:
    vs = get_vectorstore()
    retriever = HybridRetriever()
    for f in uploaded:
        path = config.KB_DIR / f.name
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        with st.spinner(f"索引 {f.name} 中…（含解析、清洗、切分、增量 embedding）"):
            try:
                stats = index_file(path, vs)
            except Exception as e:
                st.error(f"{f.name} 索引失败：{e}")
                continue
        retriever.rebuild()
        kind = "全新索引" if stats["unchanged"] == 0 and stats["removed"] == 0 else "增量更新"
        st.success(f"✅ {f.name}（{kind}）：新增 {stats['added']} 块，"
                   f"未变 {stats['unchanged']} 块，移除 {stats['removed']} 块")
    st.rerun()

st.divider()
st.subheader("已索引文档")

if st.button("🔄 全量重建索引（切换切分策略后使用）"):
    from pathlib import Path
    with st.spinner("重建中…"):
        for f in Path(config.KB_DIR).glob("*"):
            if f.is_file():
                index_file(f, force=True)
        HybridRetriever().rebuild()
    st.success("重建完成")
    st.rerun()

docs = list_indexed_docs()
if not docs:
    st.info("知识库为空，请先上传文档。")
else:
    total = sum(docs.values())
    c1, c2 = st.columns(2)
    c1.metric("文档数", len(docs))
    c2.metric("知识块总数", total)
    for name, n in docs.items():
        col1, col2 = st.columns([4, 1])
        col1.write(f"📄 {name} — {n} 块")
        if col2.button("删除", key=f"del_{name}"):
            remove_file(name)
            path = config.KB_DIR / name
            if path.exists():
                path.unlink()
            HybridRetriever().rebuild()
            st.rerun()
