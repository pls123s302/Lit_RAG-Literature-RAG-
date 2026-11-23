"""链路策略配置与工厂：通过 PIPELINE 环境变量 / config 切换各环节实现。

PIPELINE 示例（JSON，可只写要覆盖的环节）：
  {"chunker": "recursive", "reranker": "rrf_only"}

用于效果对比实验：同一份测试集在不同策略下跑分。
"""
import json
import os

from src import config
from src.registry import get_plugin, list_plugins, register

# 环节名 -> 默认实现
PIPELINE_DEFAULTS = {
    "chunker": "academic_section",
    "query_rewriter": "llm_rewrite",
    "reranker": "llm_listwise",
    "retriever": "hybrid_rrf",
}


def _active_pipeline() -> dict:
    """合并默认策略与用户覆盖（环境变量 PIPELINE 为 JSON）。"""
    active = dict(PIPELINE_DEFAULTS)
    raw = os.getenv("PIPELINE", "")
    if raw:
        try:
            active.update(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return active


def get_stage(stage: str):
    """获取某环节当前激活的插件实例。"""
    return get_plugin(stage, _active_pipeline()[stage])


def available() -> dict:
    return list_plugins()


# ================= 各环节的可插拔实现 =================

# ---- 切分器 ----
@register("chunker", "academic_section")
def _chunker_academic():
    """学术论文结构化切分：跨页按章节组装，[摘要][方法][实验] 等标签前缀。"""
    from src.ingestion.section_splitter import split_documents_academic
    return split_documents_academic


@register("chunker", "heading_aware")
def _chunker_heading():
    from src.ingestion.splitter import split_documents
    return split_documents


@register("chunker", "recursive")
def _chunker_recursive():
    """无结构感知的纯递归切分（基线，用于对比实验）。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from src.ingestion.splitter import clean_text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""])
    def _split(docs):
        return [d for doc in docs
                for d in splitter.create_documents([clean_text(doc.page_content)],
                                                   metadatas=[doc.metadata])]
    return _split


# ---- 查询改写 ----
@register("query_rewriter", "llm_rewrite")
def _rewriter_llm():
    from src.retrieval.rerank import rewrite_query
    return rewrite_query


@register("query_rewriter", "noop")
def _rewriter_noop():
    """不改写直接检索（基线）。"""
    return lambda q: {"rewritten": q, "keywords": []}


# ---- 重排 ----
@register("reranker", "llm_listwise")
def _reranker_llm():
    from src.retrieval.rerank import rerank
    return rerank


@register("reranker", "rrf_only")
def _reranker_rrf_only():
    """不做重排，按 RRF 分数截断（基线）。"""
    from src import config as cfg

    def _rerank(question, docs):
        return docs[: cfg.RERANK_TOP_K]
    return _rerank
