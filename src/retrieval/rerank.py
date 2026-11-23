"""查询改写 + LLM listwise 重排。"""
import json
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src import config
from src.llm import get_chat_llm

REWRITE_PROMPT = """你是一个知识库检索查询优化器。请把用户问题改写为更适合检索的形式：
1. 补全指代，使问题独立可理解；
2. 提炼出 2 个检索用的关键词查询。

用户问题：{question}

严格按以下 JSON 输出，不要输出其他内容：
{{"rewritten": "改写后的问题", "keywords": ["关键词查询1", "关键词查询2"]}}"""


def rewrite_query(question: str) -> dict:
    """LLM 查询改写与扩展。失败时回退到原始问题。"""
    try:
        raw = (get_chat_llm(temperature=0.0)
               .invoke(REWRITE_PROMPT.format(question=question))
               .content)
        m = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(m.group(0)) if m else {}
        return {
            "rewritten": parsed.get("rewritten", question),
            "keywords": parsed.get("keywords", []),
        }
    except Exception:
        return {"rewritten": question, "keywords": []}


RERANK_PROMPT = """你是重排模型。给定用户问题和候选文档，对每个文档的相关性打 1-5 分：
5=文档直接包含回答问题所需的信息；4=文档高度相关、包含大部分答案；3=文档仅部分相关、只含少量有用信息；
2=文档与问题仅主题相近但没有答案；1=文档与问题完全无关。

注意：
1. 与问题主题不同领域的文档必须打 1 或 2 分，禁止打 3 分以上；
2. 若文档含有大量数学符号、公式片段（如希腊字母、上下标堆叠、=、∑、exp 等）而问题在询问公式/定义/方法细节，
   即使符号文本被线性化打乱也应视为高度相关（4-5 分）。

问题：{question}

候选文档：
{docs}

只输出 JSON 数组，按文档顺序给出分数，例如 [5,3,1,2,4]，不要输出其他内容。"""


def rerank(question: str, docs: list[Document]) -> list[Document]:
    """LLM listwise 重排，保留相关性 >= MIN_EVIDENCE_SCORE 的块。失败时按原序截断。"""
    if not docs:
        return []
    # 截取 RRF 前 N 个候选（N 越小 prompt 越短，重排越快）
    docs = docs[: config.RERANK_MAX_CANDIDATES]
    numbered = "\n\n".join(
        f"[文档{i}] {d.page_content[:400]}" for i, d in enumerate(docs, start=1)
    )
    try:
        raw = (get_chat_llm(temperature=0.0)
               .invoke(RERANK_PROMPT.format(question=question, docs=numbered))
               .content)
        m = re.search(r"\[[\d\s,\.]+\]", raw)
        scores = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(0))] if m else []
        if len(scores) == len(docs):
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            # 全部低于阈值是合法结果（返回空，触发拒答），不是解析失败
            kept = [(d, s) for d, s in ranked if s >= config.MIN_EVIDENCE_SCORE][: config.RERANK_TOP_K]
            for d, s in kept:
                d.metadata["rerank_score"] = s
            return [d for d, _ in kept]
    except Exception:
        pass
    # 兜底（仅解析失败/调用异常）：保留 RRF 顺序，截断到 RERANK_TOP_K
    for d in docs[: config.RERANK_TOP_K]:
        d.metadata["rerank_score"] = None
    return docs[: config.RERANK_TOP_K]
