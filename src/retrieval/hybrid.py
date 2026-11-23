"""混合检索：BM25（jieba 分词）+ Dense（Chroma 向量），RRF 融合。

search_many 多路查询共用一次批量 embedding + 一次 Chroma 批量查询（在线链路提速）。"""
import functools
import re
import threading

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src import config
from src.ingestion.indexer import get_vectorstore

_lock = threading.Lock()

# 查询在问公式时，给公式转录块确定性加权（纯符号文本的 embedding 语义对齐弱，不赌它）
_FORMULA_QUERY_RE = re.compile(r"公式|方程|表达式|推导|formula|equation|latex|derive", re.I)
_FORMULA_MARKERS = ("[页面公式-LaTeX转录]", "[含公式]")
_FORMULA_BOOST_QUERY = 3.0   # 查询明确问公式时
_FORMULA_BOOST_BASE = 1.3    # 任何查询下的小幅基础加权（公式块信息密度高，但别压过正文的语义匹配）


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


@functools.lru_cache(maxsize=1)
def _load_corpus() -> tuple[list[str], list[str], tuple[dict, ...], BM25Okapi]:
    """从向量库全量取块构建 BM25 语料；索引更新后调用 rebuild() 失效缓存。"""
    data = get_vectorstore().get(include=["documents", "metadatas"])
    ids: list[str] = data["ids"]
    docs: list[str] = data["documents"]
    metas: tuple[dict, ...] = tuple(dict(m) for m in data["metadatas"])
    corpus = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus) if corpus else None
    return ids, docs, metas, bm25


class HybridRetriever:
    def rebuild(self) -> None:
        with _lock:
            _load_corpus.cache_clear()

    def search(self, query: str) -> list[Document]:
        return self.search_many([query])

    def search_many(self, queries: list[str]) -> list[Document]:
        """多路查询混合检索：1 次批量 embedding + 1 次批量向量查询 + 各路 BM25，全部 RRF 融合。

        返回融合排序后的候选块（metadata 带 chunk_id / rrf_score）。"""
        if not queries:
            return []
        ids, docs, metas, bm25 = _load_corpus()
        if not ids:
            return []
        pos = {cid: i for i, cid in enumerate(ids)}

        # Dense：一次 API 拿到所有查询的向量，一次 Chroma 调用批量检索
        from src.llm import get_embeddings
        embs = get_embeddings().embed_documents(list(queries))
        n_res = min(config.DENSE_TOP_K, len(ids))
        hits = get_vectorstore()._collection.query(query_embeddings=embs, n_results=n_res)

        # Sparse：BM25 本地逐路打分
        sparse_per_q: list[list[str]] = []
        if bm25 is not None:
            for q in queries:
                scores = bm25.get_scores(_tokenize(q))
                ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
                sparse_per_q.append([ids[i] for i in ranked[: config.SPARSE_TOP_K] if scores[i] > 0])
        else:
            sparse_per_q = [[] for _ in queries]

        # RRF: score = sum 1/(k + rank)，跨所有查询、双路融合
        rrf: dict[str, float] = {}
        for dense_ids in hits["ids"]:
            for rank, cid in enumerate(dense_ids, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (config.RRF_K + rank)
        for sparse_ids in sparse_per_q:
            for rank, cid in enumerate(sparse_ids, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (config.RRF_K + rank)

        # 公式块加权：基础小幅加权任何查询都生效；查询明确问公式时大幅加权
        is_formula_query = _FORMULA_QUERY_RE.search(" ".join(queries)) is not None
        for cid in rrf:
            i = pos.get(cid)
            if i is None:
                continue
            if any(m in docs[i] for m in _FORMULA_MARKERS):
                rrf[cid] *= _FORMULA_BOOST_QUERY if is_formula_query else _FORMULA_BOOST_BASE

        out = []
        for cid in sorted(rrf, key=rrf.get, reverse=True):
            i = pos.get(cid)
            if i is None:
                continue
            out.append(Document(page_content=docs[i],
                                metadata={**metas[i], "chunk_id": cid,
                                          "rrf_score": round(rrf[cid], 6)}))
        return out
