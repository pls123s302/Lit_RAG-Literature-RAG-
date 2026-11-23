"""简化版 RAGAS 指标：用 LLM（qwen）当裁判，复刻四个核心指标。

- faithfulness        忠实度：答案是否全部有上下文依据（0-1）
- answer_relevancy    答案相关性：答案是否切题（0-1）
- context_precision   上下文精确率：检索块中相关块的比例（0-1）
- context_recall      上下文召回率：标准答案要点被检索块覆盖的比例（0-1）
"""
import json
import re

from src.llm import get_chat_llm


def _ask_llm(prompt: str) -> str:
    return get_chat_llm(temperature=0.0).invoke(prompt).content


def _extract_json(raw: str, default):
    m = re.search(r"\{.*\}|\[.*\]", raw, re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return default


def faithfulness(answer: str, contexts: list[str]) -> float:
    """把答案拆成事实陈述，逐条判断是否被上下文支持，返回支持比例。"""
    raw = _ask_llm(
        "把下面的回答拆成独立的事实陈述，然后逐条判断是否能从参考文档中找到依据。\n"
        f"参考文档：\n{''.join(contexts)}\n\n回答：\n{answer}\n\n"
        '只输出 JSON：{"statements": [{"s": "陈述", "supported": true/false}]}'
    )
    data = _extract_json(raw, {})
    stmts = data.get("statements", [])
    if not stmts:
        return 1.0 if not answer.strip() else 0.0
    return sum(1 for s in stmts if s.get("supported")) / len(stmts)


def answer_relevancy(question: str, answer: str) -> float:
    raw = _ask_llm(
        f"问题：{question}\n\n回答：{answer}\n\n"
        "请评估该回答对问题的相关性和完整度，输出 0-10 的整数分数"
        '（10=完全切题且完整，0=完全跑题）。只输出 JSON：{"score": 8}'
    )
    score = _extract_json(raw, {}).get("score", 0)
    return max(0.0, min(1.0, float(score) / 10))


def context_precision(question: str, contexts: list[str]) -> float:
    raw = _ask_llm(
        f"问题：{question}\n\n候选文档：\n"
        + "\n".join(f"[文档{i}] {c[:400]}" for i, c in enumerate(contexts, 1))
        + '\n\n判断每个文档对回答该问题是否有用。只输出 JSON：{"useful": [true, false, ...]}'
    )
    flags = _extract_json(raw, {}).get("useful", [])
    if not flags or not contexts:
        return 0.0
    n = min(len(flags), len(contexts))
    return sum(1 for f in flags[:n] if f) / n


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    """把标准答案拆成要点，判断各要点是否被检索上下文覆盖。"""
    raw = _ask_llm(
        "把标准答案拆成若干要点，逐条判断参考文档是否覆盖该要点。\n"
        f"参考文档：\n{''.join(contexts)}\n\n标准答案：\n{ground_truth}\n\n"
        '只输出 JSON：{"points": [{"p": "要点", "covered": true/false}]}'
    )
    points = _extract_json(raw, {}).get("points", [])
    if not points:
        return 0.0
    return sum(1 for p in points if p.get("covered")) / len(points)


def evaluate_single(question: str, answer: str, ground_truth: str,
                    contexts: list[str]) -> dict:
    """对单条问答计算全部四个指标。"""
    ctxs = [c if isinstance(c, str) else c.page_content for c in contexts]
    return {
        "faithfulness": faithfulness(answer, ctxs),
        "answer_relevancy": answer_relevancy(question, answer),
        "context_precision": context_precision(question, ctxs),
        "context_recall": context_recall(ground_truth, ctxs),
    }
