"""LangGraph 在线检索问答图（优化版：减少串行 LLM 调用、流式生成）。

START → check_cache ─(命中)→ END
      └─(未命中)→ route_and_rewrite（一次 LLM 调用同时完成路由+改写）
                      ├─(chitchat)→ casual_answer → cache_write → END
                      └─(knowledge)→ hybrid_retrieve → rerank → judge_sufficiency
                          ├─(证据不足且未重试)→ second_retrieve → rerank → …
                          ├─(证据不足且已重试)→ no_answer → cache_write → END
                          └─(证据充足)→ generate（流式）→ verify_citations
                                           ├─(未通过且未重试)→ second_retrieve → …
                                           └─(通过)→ cache_write → END
"""
import functools
import json
import operator
import re
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src import config
from src.cache import layered_cache
from src.llm import get_chat_llm
from src.pipeline import get_stage
from src.retrieval.hybrid import HybridRetriever

RAG_PROMPT = """你是科研文献管理与问答助手，基于用户上传的科研文献（论文 PDF、Markdown 笔记、图表截图）回答问题，并标注引用来源。

要求：
1. 答案中的每个事实都要能在参考文档中找到依据，禁止编造；
2. 引用时在句末标注 [来源n]，n 必须是下面真实存在的来源编号，禁止引用未提供的来源；
3. 参考文档不足以回答时，明确说明文献库中没有相关内容；
4. 涉及数学公式时用 LaTeX 输出：行内公式用 $...$，独立公式块用 $$...$$，符号与参考文档保持一致；
   LaTeX 必须语法合法：上下标必须用花括号分组（如 $x_{{t}}$、$\\hat{{y}}_{{t-1}}$、$e^{{-\\theta d}}$），
   禁止裸 ^ 或 _ 后直接跟逗号/括号；参考文档含 [页面公式-LaTeX转录] 时优先原样引用其中的 LaTeX；
5. 专业术语、方法名、模型名保留英文原文（如 monotonic attention、AKT），首次出现时可附中文解释；
6. 涉及方法对比时，按"方法-关键机制-实验结论"的结构组织答案。

参考文档：
{context}

用户问题：{question}"""

ROUTE_REWRITE_PROMPT = """你是检索问答系统的前置处理器，对用户输入同时完成两件事：

1. 类型判断：knowledge（需要查用户上传的文献/笔记才能回答的问题）/ chitchat（闲聊、问候、对助手自身的元问题如"你是谁"）
2. 若为 knowledge：将问题改写为独立、无指代、适合检索的问法（补全指代），并提炼 2 个关键词查询；
   关键词必须从不同角度覆盖问题（相关子概念、英文术语、组件名），不要机械截取原问短语。
   例：问"XX的公式有哪些"时，关键词应指向具体组件（如 "XX monotonic attention formula"、"XX model components"）。
   若为 chitchat，rewritten 原样返回输入

用户输入：{question}

只输出 JSON：
{{"route": "knowledge或chitchat", "rewritten": "改写后的问题", "keywords": ["关键词查询1", "关键词查询2"]}}"""

VERIFY_PROMPT = """核查下面的回答与其引用来源是否一致。

参考文档：
{context}

回答：
{answer}

判断每处 [来源n] 引用：该来源文档是否确实支持回答中被标注的句子。
只输出 JSON：{{"supported": true/false, "has_citation": true/false, "reason": "一句话"}}"""


class RAGState(TypedDict):
    question: str
    route: str                    # knowledge / chitchat
    rewritten: str
    keywords: list[str]
    candidates: list[Document]    # 各路查询 RRF 融合后的候选（每轮全量覆盖）
    contexts: list[Document]
    answer: str
    sources: list[dict]
    cache_hit: str | None
    attempt: int                  # 已执行的检索轮数（0=首轮）
    confidence: float             # 生成后引用核查结果：1 通过 / 0 未通过
    trace: Annotated[list[str], operator.add]
    start_time: float
    latency_ms: int


_retriever = HybridRetriever()


def _mark(state: RAGState, text: str) -> str:
    """trace 条目前缀：距问题开始的累计毫秒数，用于定位慢点。"""
    return f"[+{(time.time() - state['start_time']) * 1000:.0f}ms] {text}"


def check_cache(state: RAGState) -> dict:
    hit, level = layered_cache.cache_lookup(state["question"])
    if hit:
        return {
            "answer": hit["answer"],
            "sources": hit.get("sources", []),
            "cache_hit": level,
            "trace": [f"缓存命中（{level}）"],
            "latency_ms": int((time.time() - state["start_time"]) * 1000),
        }
    return {"cache_hit": None, "trace": ["缓存未命中，进入检索链路"]}


# ---------- 1. 路由 + 查询改写（合并为一次 LLM 调用） ----------
def route_and_rewrite(state: RAGState) -> dict:
    """一次 LLM 调用同时完成问题路由与查询改写（省一次串行调用）。"""
    try:
        raw = (get_chat_llm(temperature=0.0)
               .invoke(ROUTE_REWRITE_PROMPT.format(question=state["question"])).content)
        m = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(m.group(0)) if m else {}
        route = parsed.get("route", "knowledge")
        if route not in ("knowledge", "chitchat"):
            route = "knowledge"
        rewritten = parsed.get("rewritten") or state["question"]
        keywords = [k for k in parsed.get("keywords", []) if k][:2]
    except Exception:
        route, rewritten, keywords = "knowledge", state["question"], []
    return {
        "route": route,
        "rewritten": rewritten,
        "keywords": keywords,
        "trace": [_mark(state, f"路由: {route} | 查询改写: {rewritten} | 关键词: {keywords}")],
    }


def casual_answer(state: RAGState) -> dict:
    answer = (get_chat_llm()
              .invoke(f"你是科研文献管理助手，当前问题是闲聊/元问题，无需检索文献，"
                      f"直接友好简短地回复，并可顺带提示用户可以询问文献内容。\n"
                      f"用户：{state['question']}").content)
    return {"answer": answer, "sources": [], "trace": ["闲聊直接回复"]}


# ---------- 2. 混合检索（多路共用一次批量 embedding） ----------
def hybrid_retrieve(state: RAGState) -> dict:
    queries = [state["rewritten"], *state.get("keywords", [])][:3]
    candidates = _retriever.search_many(queries)
    return {
        "candidates": candidates,
        "trace": [_mark(state, f"混合检索 {len(queries)} 路查询（批量embedding），融合后 {len(candidates)} 个候选块")],
    }


def rerank_step(state: RAGState) -> dict:
    contexts = get_stage("reranker")(state["rewritten"], state["candidates"])
    return {
        "contexts": contexts,
        "trace": [_mark(state, f"LLM 重排后保留 {len(contexts)} 个证据块")],
    }


def judge_sufficiency(state: RAGState) -> Literal["generate", "no_answer", "second_retrieve"]:
    """证据不足时：首轮先尝试二次检索换查询再召回，仍无证据才拒答。"""
    if state.get("contexts"):
        return "generate"
    if state.get("attempt", 0) == 0:
        return "second_retrieve"
    return "no_answer"


# ---------- 3. 低置信度二次检索 ----------
def second_retrieve(state: RAGState) -> dict:
    """换检索策略：改用原始问题 + 未用过的关键词组合，扩大候选。"""
    attempt = state.get("attempt", 0) + 1
    base = state["question"].strip("？?。.！! ")
    queries = [state["question"], *state.get("keywords", []), base][:3]
    candidates = _retriever.search_many(queries)
    return {
        "attempt": attempt,
        "candidates": candidates,
        "contexts": [],
        "trace": [_mark(state, f"低置信度二次检索（第{attempt}轮），查询: {queries}，召回 {len(candidates)} 个候选块")],
    }


# ---------- 4. 生成（流式） ----------
def generate(state: RAGState) -> dict:
    contexts = state["contexts"]
    blocks, sources = [], []
    for i, d in enumerate(contexts, start=1):
        meta = d.metadata
        sec, head = meta.get("section", ""), meta.get("heading", "")
        if sec == "[公式]":
            loc = f"第{meta.get('page', '?')}页公式"
        elif head:
            loc = f"{sec}·{head}" if sec else head
        else:
            loc = f"第{meta.get('page', '?')}页"
        blocks.append(f"[来源{i}] 《{meta.get('source', '?')}》{loc}\n{d.page_content}")
        sources.append({
            "id": i, "source": meta.get("source"), "location": loc,
            "snippet": d.page_content[:120].replace("\n", " "),
        })
    answer = (get_chat_llm(thinking=config.GENERATE_THINKING)
              .invoke(RAG_PROMPT.format(context="\n\n".join(blocks), question=state["question"]))
              .content)
    return {
        "answer": answer,
        "sources": sources,
        "trace": [_mark(state, f"生成回答，引用 {len(sources)} 个来源")],
    }


# ---------- 5. 引用核查（抽查制：正则常开，LLM 内容核查按开关） ----------
def _citation_ids(answer: str) -> set[int]:
    return {int(m) for m in re.findall(r"\[来源(\d+)\]", answer)}


def _strip_bogus_citations(answer: str, valid_ids: set[int]) -> str:
    def repl(m):
        return "" if int(m.group(1)) not in valid_ids else m.group(0)
    return re.sub(r"\s*\[来源(\d+)\]", repl, answer)


def verify_citations(state: RAGState) -> dict:
    answer, sources = state["answer"], state.get("sources", [])
    cited = _citation_ids(answer)
    valid_ids = {s["id"] for s in sources}

    # 第一道：正则校验（毫秒级）——编号必须真实存在
    if not cited:
        ok, reason = False, "回答未包含引用标注"
    elif not cited <= valid_ids:
        ok, reason = False, "引用编号越界"
        answer = _strip_bogus_citations(answer, valid_ids)
    else:
        ok, reason = True, "正则校验通过"

    # 第二道：LLM 内容支持度核查（仅当开关打开；默认关闭以保证速度）
    if ok and config.LLM_CITATION_VERIFY:
        try:
            blocks = "\n\n".join(f"[来源{s['id']}] {s['snippet']}" for s in sources)
            raw = (get_chat_llm(temperature=0.0)
                   .invoke(VERIFY_PROMPT.format(context=blocks, answer=answer)).content)
            m = re.search(r"\{.*\}", raw, re.S)
            verdict = json.loads(m.group(0)) if m else {"supported": True}
            ok = bool(verdict.get("supported", True))
            reason = f"LLM核查{'通过' if ok else '未通过'}"
        except Exception:
            pass

    return {
        "confidence": 1.0 if ok else 0.0,
        "answer": answer,
        "trace": [_mark(state, f"引用核查: {'通过' if ok else '未通过（' + reason + '）'}")],
    }


def route_verify(state: RAGState) -> Literal["cache_write", "second_retrieve"]:
    if state.get("confidence", 1.0) >= 1.0 or state.get("attempt", 0) >= 1:
        return "cache_write"
    return "second_retrieve"


def no_answer(state: RAGState) -> dict:
    return {
        "answer": "文献库中没有找到与该问题相关的内容，请尝试换一种问法，或先上传相关文献。",
        "sources": [],
        "trace": [_mark(state, "证据不足，拒答")],
    }


def cache_write(state: RAGState) -> dict:
    if state.get("cache_hit") is None and state.get("sources"):
        layered_cache.cache_store(state["question"], state["answer"], state["sources"])
    return {"latency_ms": int((time.time() - state["start_time"]) * 1000)}


@functools.lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("check_cache", check_cache)
    g.add_node("route_and_rewrite", route_and_rewrite)
    g.add_node("casual_answer", casual_answer)
    g.add_node("hybrid_retrieve", hybrid_retrieve)
    g.add_node("rerank", rerank_step)
    g.add_node("generate", generate)
    g.add_node("verify_citations", verify_citations)
    g.add_node("second_retrieve", second_retrieve)
    g.add_node("no_answer", no_answer)
    g.add_node("cache_write", cache_write)

    g.add_edge(START, "check_cache")
    g.add_conditional_edges("check_cache", lambda s: "cache_write" if s.get("cache_hit") else "route_and_rewrite",
                            {"cache_write": "cache_write", "route_and_rewrite": "route_and_rewrite"})
    g.add_conditional_edges("route_and_rewrite",
                            lambda s: "casual_answer" if s.get("route") == "chitchat" else "hybrid_retrieve",
                            {"casual_answer": "casual_answer", "hybrid_retrieve": "hybrid_retrieve"})
    g.add_edge("casual_answer", "cache_write")
    g.add_edge("hybrid_retrieve", "rerank")
    g.add_conditional_edges("rerank", judge_sufficiency,
                            {"generate": "generate", "no_answer": "no_answer",
                             "second_retrieve": "second_retrieve"})
    g.add_edge("second_retrieve", "rerank")
    g.add_edge("generate", "verify_citations")
    g.add_conditional_edges("verify_citations", route_verify,
                            {"cache_write": "cache_write", "second_retrieve": "second_retrieve"})
    g.add_edge("no_answer", "cache_write")
    g.add_edge("cache_write", END)
    return g.compile()


def initial_state(question: str) -> dict:
    return {
        "question": question,
        "keywords": [],
        "candidates": [],
        "contexts": [],
        "sources": [],
        "cache_hit": None,
        "attempt": 0,
        "confidence": 0.0,
        "trace": [],
        "start_time": time.time(),
    }


def ask(question: str) -> dict:
    """对外入口：执行完整问答图。"""
    return build_graph().invoke(initial_state(question))


def ask_stream(question: str):
    """流式执行：generator 逐段产出 generate 节点的 LLM token，
    结束时把完整最终状态写入 st.session_state 供调用方读取（仅供 Streamlit UI 使用）。"""
    import streamlit as st

    graph = build_graph()
    inp = initial_state(question)

    def gen():
        final = None
        for mode, payload in graph.stream(inp, stream_mode=["messages", "values"]):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") != "generate":
                    continue
                c = chunk.content
                if isinstance(c, list):
                    c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                if c:
                    yield c
            else:
                final = payload
        st.session_state["_rag_final"] = final or {}

    return gen()
