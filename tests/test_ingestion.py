import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from src.ingestion.parsers import parse_markdown
from src.ingestion.splitter import split_documents, clean_text

p = Path("data/knowledge/sample.md")
p.parent.mkdir(parents=True, exist_ok=True)
long_para = "混合检索结合BM25稀疏检索与稠密向量检索，通过RRF融合排序，可以显著提升召回质量。" * 20
p.write_text(
    "# RAG系统\n\nRAG（检索增强生成）是结合检索与生成的技术。\n\n"
    f"## 混合检索\n\n{long_para}\n\n## 缓存\n\n分层缓存包括精确缓存和语义缓存。\n",
    encoding="utf-8",
)

docs = parse_markdown(p)
print("markdown sections:", len(docs), [d.metadata["heading"] for d in docs])
chunks = split_documents(docs)
for c in chunks:
    print(len(c.page_content), "|", c.metadata["heading"], "|", c.page_content[:36].replace("\n", " "))

print("clean:", repr(clean_text("你好   \n\n\n\n世界 (1)[2]  end")))
