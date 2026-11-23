"""从文献库自动生成分类基准测试集（benchmark）。

按 10 类场景从真实索引内容出题：每篇论文出事实/公式/口语化/概念对比/多跳/跨章节题，
全局出库外拒答/闲聊/综述题。题目带 category + expect 字段供 runner 做自动断言。"""
import json
import random
import re
from collections import defaultdict

from src import config
from src.llm import get_chat_llm
from src.ingestion.indexer import get_vectorstore

# ---- 每篇论文的出题类别：(类别, 每篇题数, 出题要求) ----
PER_PAPER_CATEGORIES = [
    ("事实细节", 2, "问具体数值、参数设置、组件名称等单点事实（如某数据集上的AUC、某超参数取值）。"),
    ("公式类", 1, "问论文中某个具体公式的形式（如损失函数、注意力权重、某个模块的定义式）。"
                 "标准答案里必须用 LaTeX 写出公式。"),
    ("口语化指代", 1, "模拟用户口语：问题中绝对不能出现论文标题或方法名，"
                    "用'它/该模型/这篇文献'指代论文主体（如'它的主要贡献是什么？'）。"),
    ("相似概念对比", 1, "问论文内部两个容易混淆的概念/变体/设置的区别（如某方法的两个变体、两种实验设置）。"),
    ("多跳推理", 1, "答案必须综合两个不同章节的信息才能完整（如'方法里引入的某参数在实验中如何设置/有何影响'）。"
                  "sections_used 必须包含两个不同标签。"),
    ("跨章节综合", 1, "问需要概括全文的问题（核心创新、整体框架、主要结论），答案需覆盖摘要+方法+实验。"),
]

# ---- 全局类别（不区分论文） ----
GLOBAL_CATEGORIES = [
    ("库外拒答", 4, None),      # 专用生成逻辑
    ("闲聊元问题", 3, None),    # 静态题
    ("文献综述扩展", 2, None),  # 跨论文对比
]

_STATIC_CHITCHAT = [
    {"question": "你是谁？", "ground_truth": "我是科研文献管理助手，帮助用户管理并问答上传的科研文献。",
     "category": "闲聊元问题", "expect_route": "chitchat"},
    {"question": "你能帮我做什么？", "ground_truth": "我可以回答你上传文献中的内容，包括方法细节、实验结果、公式推导，并标注引用来源。",
     "category": "闲聊元问题", "expect_route": "chitchat"},
    {"question": "你们支持上传什么格式的文献？", "ground_truth": "支持论文 PDF、Markdown 笔记、TXT 以及公式图表截图。",
     "category": "闲聊元问题", "expect_route": "chitchat"},
]

_JSON_RE = re.compile(r"\[.*\]", re.S)


def _library_chunks() -> dict[str, list]:
    """按论文分组索引块（排除自建样例 md）。"""
    data = get_vectorstore().get(include=["documents", "metadatas"])
    by_doc = defaultdict(list)
    for i in range(len(data["ids"])):
        meta = data["metadatas"][i]
        text, src = data["documents"][i], meta.get("source", "")
        if src.endswith(".md"):
            continue
        by_doc[src].append((text, meta))
    return dict(by_doc)


def _pick(chunks: list, pred=None, k: int = 3) -> list:
    pool = [(t, m) for t, m in chunks if (pred is None or pred(t, m))] or chunks
    random.shuffle(pool)
    return pool[:k]


def _fmt(chunks: list) -> str:
    parts = [f"[片段{j}｜标签:{m.get('section', '未知')}]\n{t[:700]}"
             for j, (t, m) in enumerate(chunks, 1)]
    return "\n\n".join(parts)


def _gen_category(cat: str, n: int, req: str, chunks: list, source: str) -> list[dict]:
    sel = _pick(chunks, k=3)
    prompt = (f"你是科研文献问答系统的测试集构造器。基于以下论文片段，生成{n}道「{cat}」类型的问答题。\n\n"
              f"出题要求：\n{req}\n"
              f"- 每个问题必须能从片段内容中找到答案；标准答案忠实片段，保留关键数字与英文术语；\n"
              f"- sections_used 必须从片段标注的真实标签中选（如 [摘要] [方法] [实验] [公式]），禁止写 [片段N]；\n\n"
              f"只输出 JSON 数组（不要其他内容）：\n"
              f'[{{"question": "...", "ground_truth": "...", "sections_used": ["[实验]"]}}]\n\n'
              f"论文片段：\n{_fmt(sel)}")
    try:
        raw = get_chat_llm(temperature=0.3).invoke(prompt).content
        m = _JSON_RE.search(raw)
        items = json.loads(m.group(0)) if m else []
    except Exception:
        items = []
    out = []
    for it in items[:n]:
        if not it.get("question") or not it.get("ground_truth"):
            continue
        out.append({
            "question": it["question"].strip(),
            "ground_truth": it["ground_truth"].strip(),
            "category": cat,
            "expect_route": "knowledge",
            "expect_answer": True,
            "sections": it.get("sections_used", []),
            "source_doc": source,
        })
    return out


def _gen_refusal(all_chunks_meta: list) -> list[dict]:
    """库外拒答题：问与文献主题相邻但肯定未覆盖的问题。"""
    titles = sorted({m.get("source", "") for m in all_chunks_meta})
    prompt = (f"文献库中只包含以下知识追踪（Knowledge Tracing）领域的论文：\n"
              f"{chr(10).join(titles)}\n\n"
              f"请生成4个用户可能问、但该文献库肯定回答不了的问题。"
              f"要求：问题本身合理自然，但主题必须明确超出这5篇论文的覆盖范围"
              f"（例如：LLM微调技术、RAG系统架构、推荐系统、图神经网络基础理论等相邻但不同的领域）。"
              f"不要问知识追踪领域内部的问题（这些论文可能覆盖）。\n\n"
              f'只输出 JSON 数组：[{{"question": "..."}}]')
    try:
        raw = get_chat_llm(temperature=0.5).invoke(prompt).content
        m = _JSON_RE.search(raw)
        items = json.loads(m.group(0)) if m else []
    except Exception:
        items = []
    return [{"question": it["question"].strip(),
             "ground_truth": "应拒答：文献库中没有相关内容",
             "category": "库外拒答",
             "expect_route": "knowledge",
             "expect_answer": False,
             "sections": [], "source_doc": ""}
            for it in items[:4] if it.get("question")]


def _gen_survey(by_doc: dict) -> list[dict]:
    """跨论文综述题：基于各论文摘要做全局对比。"""
    abstracts = []
    for src, chunks in by_doc.items():
        abs_chunk = next((t for t, m in chunks if m.get("section") == "[摘要]"), chunks[0][0] if chunks else "")
        abstracts.append(f"《{src}》: {abs_chunk[:500]}")
    prompt = (f"文献库包含以下5篇知识追踪论文的摘要：\n\n" + "\n\n".join(abstracts) +
              f"\n\n请生成2道跨论文的综述型问答题，例如："
              f"'文献库中哪些模型使用了注意力机制？各自的特点是什么？'、"
              f"'这些论文分别解决知识追踪的什么问题？'。"
              f"标准答案必须基于上述摘要内容，并点名具体论文。\n\n"
              f'只输出 JSON 数组：[{{"question": "...", "ground_truth": "...", "sections_used": ["[摘要]"]}}]')
    try:
        raw = get_chat_llm(temperature=0.3).invoke(prompt).content
        m = _JSON_RE.search(raw)
        items = json.loads(m.group(0)) if m else []
    except Exception:
        items = []
    return [{"question": it["question"].strip(),
             "ground_truth": it["ground_truth"].strip(),
             "category": "文献综述扩展",
             "expect_route": "knowledge",
             "expect_answer": True,
             "sections": it.get("sections_used", []),
             "source_doc": "全部"}
            for it in items[:2] if it.get("question") and it.get("ground_truth")]


# 每类别的块筛选器（提高出题命中率）
_CATEGORY_PICKERS = {
    "事实细节": lambda t, m: m.get("section") in ("[实验]", "[方法]") and sum(c.isdigit() for c in t) > 8,
    "公式类": lambda t, m: m.get("section") in ("[公式]", "[方法]") and sum(c in t for c in "=\\frac∑∫^_") >= 2,
    "口语化指代": lambda t, m: m.get("section") in ("[题录]", "[摘要]", "[引言]"),
    "相似概念对比": lambda t, m: re.search(r"(?i)variant|compare|vs\.?|baseline|difference", t) is not None,
    "多跳推理": lambda t, m: m.get("section") in ("[方法]", "[实验]"),
    "跨章节综合": lambda t, m: m.get("section") in ("[摘要]", "[结论]", "[题录]"),
}


def generate_benchmark(progress_cb=None) -> list[dict]:
    """主入口：扫描文献库 → 按类别出题 → 返回测试集（并落盘）。"""
    by_doc = _library_chunks()
    if not by_doc:
        return []
    testset: list[dict] = []

    # 每篇论文 × 每类
    papers = sorted(by_doc.keys())
    total = len(papers) * len(PER_PAPER_CATEGORIES)
    done = 0
    for src in papers:
        chunks = by_doc[src]
        for cat, n, req in PER_PAPER_CATEGORIES:
            picker = _CATEGORY_PICKERS.get(cat)
            items = _gen_category(cat, n, req, _pick(chunks, pred=picker, k=10), src)
            testset.extend(items)
            done += 1
            if progress_cb:
                progress_cb(done, total, f"{src[:24]}… × {cat}")

    # 全局类别
    all_meta = [m for chunks in by_doc.values() for _, m in chunks]
    testset.extend(_gen_refusal(all_meta))
    if progress_cb:
        progress_cb(total, total, "库外拒答题")
    testset.extend(_STATIC_CHITCHAT)
    testset.extend(_gen_survey(by_doc))
    if progress_cb:
        progress_cb(total + 2, total + 2, "综述题")

    # 去重 + 过滤出题残留 + 落盘
    seen, dedup = set(), []
    for it in testset:
        key = it["question"].strip().lower()
        if key in seen or re.search(r"片段\s*\d", it["question"]):
            continue
        if any("[片段" in s for s in it.get("sections", [])):
            it["sections"] = []  # 期望值残留片段编号时清空（跳过章节断言），不丢题
        seen.add(key)
        dedup.append(it)
    random.shuffle(dedup)
    from src.evaluation.runner import save_testset
    save_testset(dedup)
    return dedup
