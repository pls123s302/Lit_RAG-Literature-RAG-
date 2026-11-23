"""批量测评 runner：分类别统计 RAGAS 四指标 + 自动断言（路由/拒答/LaTeX/section 覆盖）。

支持 pipeline 注入策略覆盖（插件化 A/B 对比）。测评绕过缓存保证真实链路。"""
import json
import os
import re
import time

from src import config
from src.evaluation.metrics import evaluate_single
from src.graph.rag_graph import build_graph
from src.pipeline import get_stage
from src.retrieval.hybrid import HybridRetriever

_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def run_without_cache(question: str) -> dict:
    """直接执行检索+生成（跳过缓存读写），用于测评。"""
    graph = build_graph()
    import src.graph.rag_graph as rg
    orig_cache_lookup = rg.layered_cache.cache_lookup
    orig_cache_store = rg.layered_cache.cache_store
    rg.layered_cache.cache_lookup = lambda q: (None, None)
    rg.layered_cache.cache_store = lambda *a, **k: None
    try:
        return graph.invoke({
            "question": question, "keywords": [], "candidates": [], "contexts": [],
            "sources": [], "cache_hit": None, "attempt": 0, "confidence": 0.0,
            "trace": [], "start_time": time.time(),
        })
    finally:
        rg.layered_cache.cache_lookup = orig_cache_lookup
        rg.layered_cache.cache_store = orig_cache_store


def retrieve_contexts(question: str) -> list[str]:
    """复现链路中的检索（供 context 指标使用），与 ask 保持一致。"""
    result = get_stage("query_rewriter")(question)
    retriever = HybridRetriever()
    candidates = retriever.search_many([result["rewritten"], *result.get("keywords", [])][:3])
    return [d.page_content for d in get_stage("reranker")(question, candidates)]


# ---- 自动断言 ----
_RE_REFUSAL = re.compile(r"没有找到|没有相关|未包含|不包含|无法回答|没有提供|未提供|不足以回答|没有包含")
_RE_BAD_LATEX = re.compile(r"\^\s*[,()\s]|_\s*[,()]\s")


def _assertions(item: dict, result: dict) -> dict:
    answer = result.get("answer", "")
    route = result.get("route", "knowledge")
    sections_of_ctx = [c.metadata.get("section", "") for c in result.get("contexts", [])]

    route_ok = (route == item.get("expect_route", "knowledge"))

    answer_ok = True
    if item.get("expect_answer") is False:  # 库外拒答题
        answer_ok = bool(_RE_REFUSAL.search(answer)) or any("拒答" in t for t in result.get("trace", []))

    latex_ok = not _RE_BAD_LATEX.search(answer)

    section_ok = True
    required = item.get("sections") or []
    if required:
        section_ok = any(s in sections_of_ctx for s in required)

    return {"route_ok": route_ok, "answer_ok": answer_ok,
            "latex_ok": latex_ok, "section_ok": section_ok}


def run_evaluation(testset: list[dict], progress_cb=None, pipeline: dict | None = None) -> dict:
    """testset: [{"question", "ground_truth", "category", "expect_route", "expect_answer", "sections"}]。
    pipeline 可注入策略覆盖做对比实验。返回总分 + 分类别统计 + 自动断言 + 失败明细。"""
    old = os.environ.get("PIPELINE")
    if pipeline:
        os.environ["PIPELINE"] = json.dumps(pipeline)
    try:
        return _run_eval(testset, progress_cb)
    finally:
        if old is None:
            os.environ.pop("PIPELINE", None)
        else:
            os.environ["PIPELINE"] = old


def _run_eval(testset: list[dict], progress_cb) -> dict:
    details = []
    agg = {k: 0.0 for k in _METRICS}
    metric_n = {k: 0 for k in _METRICS}
    # 边跑边落盘：单条失败/崩溃不丢已完成进度
    ckpt = config.EVAL_DIR / "checkpoint.jsonl"
    ckpt.write_text("", encoding="utf-8")

    for i, item in enumerate(testset):
        q, gt = item["question"], item["ground_truth"]
        cat = item.get("category", "通用")
        try:
            result = run_without_cache(q)
        except Exception as e:
            detail = {"question": q, "ground_truth": gt, "category": cat,
                      "answer": f"[链路执行失败: {str(e)[:120]}]", "latency_ms": 0,
                      "route_ok": False, "answer_ok": not item.get("expect_answer", True) is False,
                      "latex_ok": False, "section_ok": False, "error": "chain"}
            details.append(detail)
            with open(ckpt, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            if progress_cb:
                progress_cb(i + 1, len(testset), detail)
            continue

        answer = result["answer"]
        contexts = [d.page_content for d in result.get("contexts", [])]

        # 断言（所有题都做）
        asserts = _assertions(item, result)

        # RAGAS 指标：拒答/闲聊类不计算（空上下文会污染均值）；裁判超时等异常不炸全局
        scores = {}
        skip_metrics = item.get("expect_answer") is False or cat == "闲聊元问题"
        if not skip_metrics and contexts:
            try:
                scores = evaluate_single(q, answer, gt, contexts)
            except Exception:
                scores = {}
            for k in _METRICS:
                if k in scores:
                    agg[k] += scores[k]
                    metric_n[k] += 1

        detail = {"question": q, "ground_truth": gt, "category": cat,
                  "answer": answer, "latency_ms": result.get("latency_ms", 0),
                  **asserts, **scores}
        details.append(detail)
        with open(ckpt, "a", encoding="utf-8") as f:
            f.write(json.dumps(detail, ensure_ascii=False) + "\n")
        if progress_cb:
            progress_cb(i + 1, len(testset), detail)

    n = max(1, len(testset))
    summary = {k: round(agg[k] / metric_n[k], 4) if metric_n[k] else None for k in _METRICS}

    # 分类别汇总
    by_category: dict[str, dict] = {}
    for d in details:
        c = by_category.setdefault(d["category"], {"n": 0, "metrics": {k: [] for k in _METRICS},
                                                   "route_ok": 0, "answer_ok": 0,
                                                   "latex_ok": 0, "section_ok": 0, "latency": []})
        c["n"] += 1
        c["latency"].append(d.get("latency_ms", 0))
        for a in ("route_ok", "answer_ok", "latex_ok", "section_ok"):
            c[a] += 1 if d[a] else 0
        for k in _METRICS:
            if k in d:
                c["metrics"][k].append(d[k])
    for cat, c in by_category.items():
        c["metrics"] = {k: round(sum(v) / len(v), 4) if v else None for k, v in c["metrics"].items()}
        c["avg_latency_ms"] = round(sum(c["latency"]) / len(c["latency"]))
        del c["latency"]

    report = {"summary": summary, "by_category": by_category, "details": details,
              "count": len(testset), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "pipeline": os.environ.get("PIPELINE", "")}

    out = config.EVAL_DIR / f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def save_testset(testset: list[dict]) -> None:
    config.TESTSET_PATH.write_text(
        json.dumps(testset, ensure_ascii=False, indent=1), encoding="utf-8")


def load_testset() -> list[dict]:
    if config.TESTSET_PATH.exists():
        return json.loads(config.TESTSET_PATH.read_text(encoding="utf-8"))
    return []
