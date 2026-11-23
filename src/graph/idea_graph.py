"""全自动科研 Idea 融合 Agent（LangGraph 管线）。

每篇论文提取 Idea 卡片（创新/问题/技术/局限）→ 四种策略融合生成新 idea
→ 检索文献库做库内查新 → LLM 评审打分排序 → 输出研究提案卡。

管线：extract → fuse → novelty_check → score_rank → END
"""
import json
import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from src.ingestion.indexer import get_vectorstore
from src.llm import get_chat_llm
from src.retrieval.hybrid import HybridRetriever

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.S)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _parse_llm_json(raw: str, regex, default):
    m = regex.search(raw)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return default


class IdeaState(TypedDict):
    papers: list[str]
    idea_cards: Annotated[list[dict], operator.add]   # 每篇论文一张卡片
    fusions: list[dict]                                # 融合生成的原始候选
    ranked: list[dict]                                 # 查新+评分后的最终提案
    trace: Annotated[list[str], operator.add]


# ---------- 节点 1：逐篇提取 Idea 卡片 ----------
_EXTRACT_PROMPT = """基于以下论文片段，提取该论文的「Idea 卡片」，用于后续跨论文创新融合。

只输出 JSON（不要其他内容）：
{{"title": "论文标题",
  "problem": "解决的核心问题（一句话）",
  "core_ideas": ["核心创新点1", "核心创新点2"],
  "techniques": ["关键技术手段"],
  "limitations": ["论文明确提到或隐含的局限/未解决问题（重要！融合的主要原料）"],
  "datasets": ["使用的数据集"]}}

片段：
{chunks}"""


def extract_ideas(state: IdeaState) -> dict:
    from src.ingestion.indexer import get_vectorstore
    store = get_vectorstore()
    data = store.get(include=["documents", "metadatas"])
    by_doc: dict[str, list] = {}
    for i in range(len(data["ids"])):
        meta = data["metadatas"][i]
        if meta.get("source", "").endswith(".md"):
            continue
        by_doc.setdefault(meta["source"], []).append((data["documents"][i], meta))

    cards = []
    trace = []
    for paper in state["papers"]:
        chunks = by_doc.get(paper, [])
        if not chunks:
            continue
        # 按章节优先级选块，拼到 5k 字符
        priority = ["[题录]", "[摘要]", "[结论]", "[方法]", "[实验]", "[引言]", "[正文]"]
        sel, budget = [], 0
        for sec in priority:
            for t, m in chunks:
                if m.get("section") == sec and (t, m) not in sel and budget < 5000:
                    sel.append((t, m))
                    budget += len(t)
            if budget >= 5000:
                break
        if not sel:
            sel = chunks[:4]
        body = "\n\n".join(f"[{m.get('section','?')}] {t[:900]}" for t, m in sel)

        try:
            raw = get_chat_llm(temperature=0.3).invoke(
                _EXTRACT_PROMPT.format(chunks=body)).content
            card = _parse_llm_json(raw, _JSON_OBJ_RE, {})
        except Exception:
            card = {}
        if card and card.get("title"):
            card["paper"] = paper
            cards.append(card)
            trace.append(f"提取卡片: {paper[:28]}… (创新{len(card.get('core_ideas', []))}项 "
                         f"局限{len(card.get('limitations', []))}项)")
        else:
            trace.append(f"提取失败（跳过）: {paper[:28]}…")

    return {"idea_cards": cards, "trace": trace}


# ---------- 节点 2：四策略融合生成 ----------
_FUSE_PROMPT = """你是科研创意引擎。以下是 {n} 篇知识追踪领域论文的 Idea 卡片。

请用以下四种策略各生成至少 1 个【新的研究 idea】（共约 {m} 个）：
1. 机制组合：把论文 A 的核心机制嫁接到论文 B 的问题/框架上；
2. 问题迁移：把某篇的技术手段迁移到它没有覆盖的场景；
3. 矛盾驱动：找出卡片之间相互矛盾/竞争的结论，设计验证或统一框架；
4. 局限互补：用论文 B 的机制解决论文 A 声明的局限。

要求：
- idea 必须真的"新"——不能是对任何单篇论文的复述；
- motivation 要指明用了哪几篇的什么组件（引用论文名）；
- method_sketch 写清技术路线（3-5 步）；
- experiment_plan 给出数据集与对比基线（从卡片里的数据集中选）。

只输出 JSON 数组：
[{{"name": "idea 命名", "strategy": "机制组合/问题迁移/矛盾驱动/局限互补",
   "source_papers": ["论文标题"],
   "motivation": "...", "method_sketch": "...",
   "experiment_plan": "...", "risks": "主要风险与失败模式"}}]

Idea 卡片：
{cards}"""


def fuse(state: IdeaState) -> dict:
    cards_compact = json.dumps(state["idea_cards"], ensure_ascii=False, indent=1)
    n_papers = len(state["idea_cards"])
    raw = (get_chat_llm(temperature=0.6)
           .invoke(_FUSE_PROMPT.format(n=n_papers, m=n_papers + 2,
                                       cards=cards_compact[:12000])).content)
    fusions = _parse_llm_json(raw, _JSON_ARRAY_RE, [])
    fusions = [f for f in fusions if f.get("name") and f.get("method_sketch")]
    return {"fusions": fusions,
            "trace": [f"融合生成 {len(fusions)} 个候选 idea"]}


# ---------- 节点 3：库内查新 ----------
_NOVELTY_PROMPT = """判断研究 idea 相对下列已有文献内容的新颖程度。

已有文献片段：
{context}

研究 idea：{idea}

考虑：核心机制是否已被覆盖？只是主题相同还是连方法都相似？
只输出 JSON：{{"verdict": "库内未见相似工作 / 存在部分相似 / 高度重合",
"analysis": "50字以内说明"}}"""


def novelty_check(state: IdeaState) -> dict:
    retriever = HybridRetriever()
    checked = []
    trace = []
    for idea in state["fusions"]:
        query = f"{idea.get('name', '')} {idea.get('method_sketch', '')[:200]}"
        try:
            docs = retriever.search_many([query])[:3]
            context = "\n\n".join(f"[{d.metadata.get('section','?')}] {d.page_content[:400]}"
                                  for d in docs)
            raw = get_chat_llm(temperature=0.0).invoke(
                _NOVELTY_PROMPT.format(context=context,
                                       idea=json.dumps(idea, ensure_ascii=False)[:800])).content
            verdict = _parse_llm_json(raw, _JSON_OBJ_RE,
                                      {"verdict": "库内未见相似工作", "analysis": ""})
        except Exception:
            docs, verdict = [], {"verdict": "查新失败", "analysis": ""}
        idea["novelty"] = {"verdict": verdict.get("verdict", ""),
                           "analysis": verdict.get("analysis", ""),
                           "similar_chunks": [d.metadata.get("source") for d in docs]}
        checked.append(idea)
        trace.append(f"查新: {idea.get('name','')[:24]}… → {idea['novelty']['verdict']}")

    return {"ranked": checked, "trace": trace}


# ---------- 节点 4：评审打分排序 ----------
_SCORE_PROMPT = """作为领域评审专家，给以下研究 idea 打分（1-5 整数）。
novelty=相对现有工作的原创性；feasibility=用现有公开资源实现的可操作性；impact=若成功的影响力。
结合它的库内查新结论"{verdict}"评分：高度重合则 novelty 不得高于 2。

只输出 JSON：{{"novelty": n, "feasibility": n, "impact": n, "comment": "30字内评语"}}

Idea：{idea}"""


def score_rank(state: IdeaState) -> dict:
    ranked = []
    trace = []
    for idea in state["ranked"]:
        idea_compact = json.dumps({k: idea.get(k) for k in
                                   ("name", "motivation", "method_sketch", "experiment_plan")},
                                  ensure_ascii=False)[:800]
        try:
            raw = get_chat_llm(temperature=0.0).invoke(
                _SCORE_PROMPT.format(verdict=idea["novelty"].get("verdict", ""),
                                     idea=idea_compact)).content
            scores = _parse_llm_json(raw, _JSON_OBJ_RE, {})
        except Exception:
            scores = {}
        s = {k: int(scores.get(k, 0)) for k in ("novelty", "feasibility", "impact")}
        s["comment"] = scores.get("comment", "")
        s["total"] = sum(s[k] for k in ("novelty", "feasibility", "impact"))
        idea["scores"] = s
        ranked.append(idea)
        trace.append(f"评分: {idea.get('name','')[:24]}… → 总分 {s['total']}/15")

    ranked.sort(key=lambda x: x["scores"]["total"], reverse=True)
    return {"ranked": ranked, "trace": [f"完成评审排序，Top1: {ranked[0]['name'] if ranked else '—'}"]}


# ---------- 提案详化导出：把 idea 扩写成编码 Agent 可实施的 technical spec ----------
_DETAIL_PROMPT = """你是资深机器学习研究工程师。把以下研究 idea 扩写成【可直接交付编码 Agent 实现】的详细技术方案。

硬性要求：
1. 方法设计必须到模块级：每个模块写清输入/输出（维度或字段）、网络组件选型、前向流程顺序；
2. 损失函数用 LaTeX 公式写出（含各项含义）；
3. 数据集只用素材中出现的真实公开数据集，写明获取方式、关键字段、划分比例与预处理步骤；
4. 基线方法必须来自素材中提到的真实方法名；指标写明具体名称与计算口径；
5. 给出推荐技术栈与项目目录结构（目录结构用代码块）；
6. 无法从素材确定的设计决策，必须标注 `[假设]` 并给出默认取值——编码 Agent 需要知道哪些可以自行决定；
7. 验收标准写成可量化 checklist（如 "冷启动桶(交互≤5次) AUC 相比 csKT 提升 ≥1%"）；
8. 用 Markdown 输出以下章节：总体架构 / 模块规格(表格) / 损失函数 / 训练流程 / 数据与预处理 / 基线与指标 / 实验计划(主实验+消融) / 实现指南 / 风险与缓解 / 验收标准。

研究 idea：
{idea}

文献库相关片段（用于校准术语、数据集与基线名称，禁止编造素材外的数据集名）：
{context}"""


def detail_proposal(idea: dict, retriever: HybridRetriever | None = None) -> str:
    """把单个 idea 扩写为实施级技术方案（Markdown 章节）。"""
    retriever = retriever or HybridRetriever()
    try:
        docs = retriever.search_many(
            [f"{idea.get('name','')} {idea.get('method_sketch','')[:160]}"])[:4]
        context = "\n\n".join(f"[{d.metadata.get('source','?')}|{d.metadata.get('section','?')}] "
                              f"{d.page_content[:500]}" for d in docs)
    except Exception:
        context = "（检索失败，仅凭 idea 素材撰写，数据集名称需自行核实）"

    idea_compact = json.dumps({k: idea.get(k) for k in
                               ("name", "strategy", "source_papers", "motivation",
                                "method_sketch", "experiment_plan", "risks")},
                              ensure_ascii=False)
    return (get_chat_llm(temperature=0.2)
            .invoke(_DETAIL_PROMPT.format(idea=idea_compact, context=context[:6000])).content)


def export_proposals_md(proposals: list[dict], detail: bool = True,
                        progress_cb=None) -> str:
    """把提案列表导出为面向编码 Agent 的 Markdown 文档。

    detail=True 时逐条调用 LLM 扩写为实施级技术方案；False 时仅导出概要。"""
    import time as _time
    lines = [
        "# LitRAG 研究提案文档（编码 Agent 实施版）",
        "",
        f"> 生成时间：{_time.strftime('%Y-%m-%d %H:%M')} ｜ 提案数：{len(proposals)}",
        "> 来源：LitRAG 科研文献管理助手 Idea 融合引擎（机制组合/问题迁移/矛盾驱动/局限互补四策略）",
        "> 使用说明：每份提案自包含——模块规格、损失函数、数据集、基线、验收标准齐备；"
        "标注 `[假设]` 的设计决策可由实现方自行确定默认值；文档末尾附来源论文，"
        "实现前建议通读「参考资料」中的论文章节。",
        "",
        "---",
        "",
    ]
    retriever = HybridRetriever() if detail else None
    for i, p in enumerate(proposals, 1):
        sc, nv = p.get("scores", {}), p.get("novelty", {})
        lines += [
            f"# 提案 {i}：{p.get('name', '未命名')}",
            "",
            "| 元信息 | 值 |",
            "|---|---|",
            f"| 融合策略 | {p.get('strategy', '')} |",
            f"| 来源论文 | {'、'.join(p.get('source_papers', []))} |",
            f"| 评分 | 新颖 {sc.get('novelty', 0)}/5 · 可行 {sc.get('feasibility', 0)}/5 · "
            f"影响 {sc.get('impact', 0)}/5（总分 {sc.get('total', 0)}/15） |",
            f"| 库内查新 | {nv.get('verdict', '')} — {nv.get('analysis', '')} |",
            "",
            "## 1. 背景与动机",
            "",
            p.get("motivation", ""),
            "",
            "## 2. 核心创新点",
            "",
            p.get("method_sketch", ""),
            "",
        ]
        if detail:
            if progress_cb:
                progress_cb(i, len(proposals), p.get("name", ""))
            try:
                spec = detail_proposal(p, retriever)
            except Exception as e:
                spec = f"（详化失败：{str(e)[:100]}，请参考上方核心创新点手写实现方案）"
            lines += [spec, ""]
        else:
            lines += ["## 实验计划", "", p.get("experiment_plan", ""),
                      "", "## 风险", "", p.get("risks", ""), ""]
        if p.get("risks") and detail:
            lines += ["---", ""]
    return "\n".join(lines)


# ---------- 组图 ----------
def build_idea_graph():
    g = StateGraph(IdeaState)
    g.add_node("extract_ideas", extract_ideas)
    g.add_node("fuse", fuse)
    g.add_node("novelty_check", novelty_check)
    g.add_node("score_rank", score_rank)
    g.add_edge(START, "extract_ideas")
    g.add_edge("extract_ideas", "fuse")
    g.add_edge("fuse", "novelty_check")
    g.add_edge("novelty_check", "score_rank")
    g.add_edge("score_rank", END)
    return g.compile()


def run_idea_fusion(papers: list[str], progress_cb=None) -> dict:
    """对外入口。progress_cb(stage, msg) 用于 UI 进度展示。"""
    graph = build_idea_graph()
    final = None
    for mode, payload in graph.stream(
            {"papers": papers, "idea_cards": [], "fusions": [], "ranked": [], "trace": []},
            stream_mode=["messages", "values"]):
        if mode == "values":
            final = payload
            if progress_cb:
                n = len(final.get("trace", []))
                progress_cb(final.get("trace", [])[n - 1] if n else "")
    final = final or {}
    return {"proposals": final.get("ranked", []), "cards": final.get("idea_cards", []),
            "trace": final.get("trace", [])}
