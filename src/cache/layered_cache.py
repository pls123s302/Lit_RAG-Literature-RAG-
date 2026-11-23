"""分层缓存：L1 精确缓存（问题哈希 → 答案，JSON 文件）+ L2 语义缓存（embedding 余弦相似度，Chroma 独立 collection）。"""
import hashlib
import json

from langchain_chroma import Chroma

from src import config
from src.llm import get_embeddings

EXACT_PATH = config.CACHE_DIR / "exact_cache.json"


# ---------- L1 精确缓存 ----------
def _load_exact() -> dict:
    if EXACT_PATH.exists():
        return json.loads(EXACT_PATH.read_text(encoding="utf-8"))
    return {}


def _save_exact(cache: dict) -> None:
    EXACT_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def exact_get(question: str) -> dict | None:
    key = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    return _load_exact().get(key)


def exact_put(question: str, answer: dict) -> None:
    cache = _load_exact()
    key = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    cache[key] = answer
    _save_exact(cache)


# ---------- L2 语义缓存 ----------
def _sem_store() -> Chroma:
    return Chroma(
        collection_name="semantic_cache",
        embedding_function=get_embeddings(),
        persist_directory=str(config.CACHE_DIR / "semantic"),
    )


def semantic_get(question: str) -> dict | None:
    """问题 embedding 相似度超过阈值则命中，返回缓存的答案。"""
    store = _sem_store()
    emb = get_embeddings().embed_query(question)
    hits = store._collection.query(query_embeddings=[emb], n_results=1,
                                   include=["documents", "metadatas", "distances"])
    if not hits["ids"][0] or hits["distances"] is None or hits["distances"][0] is None:
        return None
    # Chroma 返回的是距离（1 - 余弦），距离 < 1 - 阈值 即命中
    if hits["distances"][0][0] <= 1 - config.SEMANTIC_CACHE_THRESHOLD:
        try:
            sources = json.loads(hits["metadatas"][0][0].get("sources", "[]"))
        except (json.JSONDecodeError, TypeError):
            sources = []
        return {"answer": hits["documents"][0][0], "sources": sources,
                "similarity": round(1 - hits["distances"][0][0], 4)}
    return None


def semantic_put(question: str, answer: str, sources: list) -> None:
    store = _sem_store()
    store.add_texts(
        texts=[answer],
        metadatas=[{"question": question, "sources": json.dumps(sources, ensure_ascii=False)}],
        ids=[hashlib.sha256((question + answer).encode()).hexdigest()[:32]],
    )
    # 简单上限：最多 1000 条
    count = store._collection.count()
    if count > 1000:
        oldest = store._collection.get(include=[], limit=count - 1000)
        store._collection.delete(ids=oldest["ids"])


# ---------- 统一入口 ----------
def cache_lookup(question: str) -> tuple[dict | None, str]:
    """返回 (缓存答案或 None, 命中层级 'exact'/'semantic'/None)。"""
    hit = exact_get(question)
    if hit:
        return hit, "exact"
    hit = semantic_get(question)
    if hit:
        return hit, "semantic"
    return None, None


def cache_store(question: str, answer: str, sources: list) -> None:
    entry = {"answer": answer, "sources": sources}
    exact_put(question, entry)
    semantic_put(question, answer, sources)


def clear_cache() -> None:
    if EXACT_PATH.exists():
        EXACT_PATH.unlink()
    import shutil
    shutil.rmtree(config.CACHE_DIR / "semantic", ignore_errors=True)
