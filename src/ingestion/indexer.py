"""增量索引：文档级 Hash 热更新 + 块级 SHA-256 指纹增量 embedding，Chroma 持久化。

- 文档级：文件字节 Hash 未变 → 秒跳过（热更新判断）；
- 块级：只对内容变更的块重新 embedding，新增/修改/删除快速同步。
"""
import hashlib
import json
import logging

from langchain_chroma import Chroma

from src import config
from src.ingestion.parsers import parse_file
from src.llm import get_embeddings
from src.pipeline import get_stage

logger = logging.getLogger(__name__)

FINGERPRINT_PATH = config.INDEX_DIR / "fingerprints.json"


def _fingerprint(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_fingerprints() -> dict:
    if FINGERPRINT_PATH.exists():
        return json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
    return {}


def _save_fingerprints(fp: dict) -> None:
    FINGERPRINT_PATH.write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="knowledge_base",
        embedding_function=get_embeddings(),
        persist_directory=str(config.INDEX_DIR),
    )


def index_file(path, vectorstore: Chroma | None = None, force: bool = False) -> dict:
    """对单个文件做增量索引。force=True 跳过文档级 Hash（如切分策略变更后需全量重建）。"""
    vectorstore = vectorstore or get_vectorstore()
    doc_name = path.name

    # 0. 文档级 Hash：文件字节未变直接跳过（热更新快速路径）
    file_hash = _fingerprint(path.read_bytes())
    all_fp = _load_fingerprints()
    old_entry = all_fp.get(doc_name, {})
    if not force and old_entry.get("file_hash") == file_hash:
        return {"added": 0, "unchanged": len(old_entry.get("chunks", {})),
                "removed": 0, "skipped": True}

    # 1. 解析 + 切分（切分器来自插件工厂，可切换策略）
    docs = parse_file(path)
    chunks = get_stage("chunker")(docs)
    if not chunks:
        return {"added": 0, "unchanged": 0, "removed": 0}

    # 2. 与旧块级指纹比对，只 embed 变更块
    old_chunks = old_entry.get("chunks", {})   # chunk_id -> fingerprint
    new_fp, to_add, unchanged = {}, [], 0
    for i, chunk in enumerate(chunks):
        cid = f"{doc_name}::{i}"
        fp = _fingerprint(chunk.page_content)
        new_fp[cid] = fp
        if old_chunks.get(cid) == fp:
            unchanged += 1
        else:
            to_add.append((cid, chunk))

    # 3. 变更块先删旧再写入
    changed_ids = [cid for cid, _ in to_add]
    if changed_ids:
        vectorstore.delete(ids=changed_ids)
    if to_add:
        vectorstore.add_documents(
            [c for _, c in to_add],
            ids=[cid for cid, _ in to_add],
        )

    # 4. 文件变短时，删除多余的旧块
    removed_ids = [cid for cid in old_chunks if cid not in new_fp]
    if removed_ids:
        vectorstore.delete(ids=removed_ids)

    # 5. 落盘（含文档级 Hash + 块级指纹 + 切分策略，策略变更可检测）
    all_fp = _load_fingerprints()
    all_fp[doc_name] = {"file_hash": file_hash, "chunks": new_fp,
                        "chunker": _active_chunker_name()}
    _save_fingerprints(all_fp)

    stats = {"added": len(to_add), "unchanged": unchanged, "removed": len(removed_ids)}
    logger.info("增量索引 %s: %s", doc_name, stats)
    return stats


def _active_chunker_name() -> str:
    from src.pipeline import _active_pipeline
    return _active_pipeline()["chunker"]


def remove_file(doc_name: str, vectorstore: Chroma | None = None) -> None:
    vectorstore = vectorstore or get_vectorstore()
    all_fp = _load_fingerprints()
    entry = all_fp.pop(doc_name, {})
    ids = list(entry.get("chunks", {})) if "chunks" in entry else list(entry)
    if ids:
        vectorstore.delete(ids=ids)
    _save_fingerprints(all_fp)


def list_indexed_docs() -> dict[str, int]:
    """返回 {文件名: 块数}。"""
    out = {}
    for name, entry in _load_fingerprints().items():
        chunks = entry.get("chunks", {}) if isinstance(entry, dict) and "chunks" in entry else entry
        out[name] = len(chunks)
    return out
