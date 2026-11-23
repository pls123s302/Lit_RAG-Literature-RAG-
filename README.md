# 科研文献管理助手（多模态 RAG）

面向科研文献管理场景的多模态 RAG 问答系统，支持**论文 PDF / Markdown 笔记 / 公式与图表截图（OCR+VLM）**等知识源，
具备公式 LaTeX 转录入库、低置信度二次检索、引用核查等文献问答专属能力，
覆盖离线索引、在线检索问答、效果测评、增量索引与分层缓存的全链路工程实现。

📊 **[Benchmark 评测报告](docs/BENCHMARK.md)**：44 题 / 9 类场景的领域 benchmark 设计与实测结果
（Faithfulness 95% · 库外拒答 4/4 · LaTeX 合法 44/44），含短板分析与 A/B 对比方法。

## ✨ 核心功能

| 页面 | 功能 |
|---|---|
| 💬 文献问答 | 流式回答 + 公式 LaTeX 渲染 + 引用来源卡片 + 执行轨迹可视化 |
| 💡 **Idea 实验室** | **全自动科研 Idea 融合 Agent**：提取每篇文献的创新与局限 → 四策略融合（机制组合/问题迁移/矛盾驱动/局限互补）→ 文献库查新 → 三维评审排序 → 输出实施级研究提案文档（含模块规格/损失函数/实验计划/验收 checklist，可直接交付编码 Agent） |
| 📁 文献库管理 | 多格式上传、增量索引统计、全量重建 |
| 📊 效果测评 | 自建 9 类 44 题领域 benchmark + RAGAS 四指标 + 自动断言 + 策略 A/B 对比 |

## 技术栈

| 模块 | 技术 |
|---|---|
| 编排 | LangGraph（StateGraph 条件分支检索问答图） |
| 生成 / 向量化 | 阿里通义 DashScope（OpenAI 兼容接口）：qwen-plus + text-embedding-v3 |
| 向量库 | ChromaDB（持久化 + 增量） |
| 稀疏检索 | BM25（rank-bm25 + jieba 分词） |
| 融合 | RRF（Reciprocal Rank Fusion） |
| 重排 | LLM listwise rerank（1-5 相关性打分过滤） |
| PDF / 图片 | PyMuPDF + RapidOCR(OCR) + qwen3.8-omni-flash(VLM 语义描述) |
| 切分 | 标题层级感知切分 + 递归切分兜底 |
| 缓存 | L1 精确缓存（问题哈希）+ L2 语义缓存（embedding 相似度 ≥ 0.95） |
| 测评 | 自研简化版 RAGAS 四指标（LLM 裁判） |
| UI | Streamlit 三页：知识问答 / 知识库管理 / 效果测评 |

## 快速开始

```bash
conda activate Langgraph
pip install -r requirements.txt

# 配置 API Key
copy .env .env   # 填入 DASHSCOPE_API_KEY

streamlit run app.py
```

## 在线链路（LangGraph）

```
START → check_cache ─(命中)──────────────────────────────→ cache_write → END
          │未命中
          ▼
        route_question（问题路由：知识问答 / 闲聊）
          ├─ 闲聊 → casual_answer（直接回复）→ cache_write → END
          ▼
        rewrite（LLM 查询改写 + 关键词扩展）
          ▼
        hybrid_retrieve（改写问题 ∥ 扩展关键词 × [BM25 + Dense] → RRF 融合）
          ▼
        rerank（LLM listwise 重排，过滤无关块）
          ├─ 证据不足(首轮) → second_retrieve（低置信度二次检索，换查询再召回）→ rerank
          ├─ 证据不足(已重试) → no_answer（拒答）→ cache_write → END
          ▼
        generate（RAG prompt + [来源n] 引用标注）
          ▼
        verify_citations（引用核查：编号合法性 + LLM 内容支持度校验）
          ├─ 未通过(首轮) → second_retrieve → …（重试一轮）
          └─ 通过 → cache_write → END
```

## 目录结构

```
app.py                  # Streamlit 入口
ui/                     # 三个页面
src/
├── config.py           # 全局配置
├── llm.py              # DashScope 封装
├── ingestion/          # 解析 / 清洗 / 切分 / 增量索引（SHA-256 指纹）
├── retrieval/          # 混合检索(BM25+Dense+RRF) / 查询改写 / LLM 重排
├── graph/              # LangGraph 问答图
├── cache/              # 精确缓存 + 语义缓存
└── evaluation/         # 简化版 RAGAS 四指标 + 批量测评 runner
data/                   # 知识库文件、索引、缓存、测评报告
```

## 测评指标

- **Faithfulness** 忠实度：答案事实陈述被上下文支持的比例（幻觉度量）
- **Answer Relevancy** 答案相关性：LLM 0-10 打分归一
- **Context Precision** 上下文精确率：检索块中对回答有用的比例
- **Context Recall** 上下文召回率：标准答案要点被检索块覆盖的比例
