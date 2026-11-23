"""全量测评执行脚本：进度实时落盘，结果汇总输出。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
import src.llm  # noqa: F401  (IPv4优先补丁)
from src.evaluation.runner import load_testset, run_evaluation

ts = load_testset()
print(f"测试集: {len(ts)} 条，开始…", flush=True)


def cb(done, total, d):
    status = "PASS" if (d["route_ok"] and d["answer_ok"] and d["latex_ok"]) else "FAIL"
    print(f"[{done}/{total}] {status} [{d['category']}] {d['question'][:36]}", flush=True)


report = run_evaluation(ts, progress_cb=cb)

with open("data/eval/last_run_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"=== 总体指标 ({report['timestamp']}, {report['count']} 条) ===\n")
    for k, v in report["summary"].items():
        f.write(f"  {k}: {v:.1%}\n" if isinstance(v, float) else f"  {k}: —\n")
    f.write("\n=== 分类别 ===\n")
    for cat, c in sorted(report["by_category"].items()):
        m = c["metrics"]
        f_ = lambda x: f"{x:.0%}" if isinstance(x, float) else "—"
        f.write(f"  [{cat}] n={c['n']} 忠实度{f_(m['faithfulness'])} 相关{f_(m['answer_relevancy'])} "
                f"精确{f_(m['context_precision'])} 召回{f_(m['context_recall'])} | "
                f"路由{c['route_ok']}/{c['n']} 拒答{c['answer_ok']}/{c['n']} "
                f"LaTeX{c['latex_ok']}/{c['n']} 章节{c['section_ok']}/{c['n']} | "
                f"均延迟{c['avg_latency_ms']}ms\n")
    fails = [d for d in report["details"]
             if not (d["route_ok"] and d["answer_ok"] and d["latex_ok"] and d["section_ok"])]
    f.write(f"\n=== 断言失败 ({len(fails)} 条) ===\n")
    for d in fails:
        bad = [k for k in ("route_ok", "answer_ok", "latex_ok", "section_ok") if not d[k]]
        f.write(f"  [{d['category']}] {bad} | {d['question'][:60]}\n")

print("DONE", flush=True)
