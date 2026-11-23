import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

# 1. 增量索引
from src.ingestion.indexer import index_file, list_indexed_docs
from src.retrieval.hybrid import HybridRetriever

p = Path("data/knowledge/sample.md")
if not p.exists():
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# RAG系统\n\nRAG（检索增强生成）是结合检索与生成的技术。\n\n"
        "## 混合检索\n\n混合检索结合BM25稀疏检索与稠密向量检索，通过RRF融合排序，可以显著提升召回质量。\n\n"
        "## 缓存\n\n分层缓存包括L1精确缓存和L2语义缓存两层。L1用问题哈希做精确匹配，L2用embedding余弦相似度做语义匹配，阈值0.95。\n",
        encoding="utf-8")
print("== 索引 ==", index_file(p))
HybridRetriever().rebuild()
print("已索引:", list_indexed_docs())

# 2. 问答
from src.graph.rag_graph import ask
t0 = time.time()
r = ask("系统有哪几层缓存？各自怎么工作？")
print(f"\n== 首次问答 ({int((time.time()-t0)*1000)}ms, cache_hit={r['cache_hit']}) ==")
print(r["answer"])
print("sources:", [(s["source"], s["location"]) for s in r["sources"]])
print("trace:", " -> ".join(r["trace"]))

# 3. 缓存命中
t0 = time.time()
r2 = ask("系统有哪几层缓存？各自怎么工作？")
print(f"\n== 二次提问 (cache_hit={r2['cache_hit']}, {r2['latency_ms']}ms) ==")
t0 = time.time()
r3 = ask("缓存分为哪两层？怎么匹配的？")
print(f"== 语义相似问题 (cache_hit={r3['cache_hit']}, {r3['latency_ms']}ms) ==")

# 4. 知识库外问题（拒答路径）
r4 = ask("法国的首都是哪里？")
print("\n== 库外问题 ==", r4["trace"])
print(r4["answer"][:60])

# 5. 测评（单条）
from src.evaluation.runner import run_evaluation
print("\n== 测评 ==")
report = run_evaluation([{"question": "系统有哪几层缓存？", "ground_truth": "分层缓存包括L1精确缓存和L2语义缓存两层。L1用问题哈希做精确匹配，L2用embedding余弦相似度做语义匹配。"}])
print("summary:", report["summary"])
