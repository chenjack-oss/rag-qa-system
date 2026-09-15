"""A/B 实验分组分析：从 jsonl（MongoDB 导出）或直连库聚合各分组指标

用法（离线分析，推荐）：
    # 1) 从 MongoDB 导出
    python -c "from app.clients.mongo_history_utils import get_mongo_client; \
               from app.experiments.metrics_recorder import export_to_jsonl; \
               print(export_to_jsonl(get_mongo_client(), 'rerank_latency_tradeoff', 'ab_out.jsonl'))"
    # 2) 分组对比
    python -m app.experiments.analyze --input ab_out.jsonl --out ab_report.md

用法（直连库）：
    python -m app.experiments.analyze --mongo --experiment rerank_latency_tradeoff
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.experiments.metrics_recorder import build_outcome_doc  # noqa: F401,E401  (re-export 便于脚本内使用)


def aggregate(records) -> dict:
    """按 variant 聚合指标（纯函数）。

    指标口径：
    - n / p50 / p95 延迟（first_token_ms / total_ms）
    - hit_rate：hit 非 null 的子集上的命中率（自动判定的样本才有贡献）
    - feedback：点赞/点踩计数（预留）
    """
    by_variant = defaultdict(list)
    for rec in records:
        by_variant[rec.get("variant", "unknown")].append(rec)

    def _percentile(sorted_vals, p):
        if not sorted_vals:
            return 0.0
        idx = min(len(sorted_vals) - 1, max(0, round(p / 100 * (len(sorted_vals) - 1))))
        return sorted_vals[idx]

    out = {}
    for variant, recs in by_variant.items():
        ft = sorted(r["first_token_ms"] for r in recs if r.get("first_token_ms") is not None)
        tt = sorted(r["total_ms"] for r in recs if r.get("total_ms") is not None)
        hit_vals = [r["hit"] for r in recs if r.get("hit") is not None]
        up = sum(1 for r in recs if r.get("user_feedback") == 1)
        down = sum(1 for r in recs if r.get("user_feedback") == -1)
        out[variant] = {
            "n": len(recs),
            "first_token_p50_ms": _percentile(ft, 50),
            "first_token_p95_ms": _percentile(ft, 95),
            "total_p50_ms": _percentile(tt, 50),
            "total_p95_ms": _percentile(tt, 95),
            "hit_rate": (sum(hit_vals) / len(hit_vals)) if hit_vals else None,
            "n_hit_labeled": len(hit_vals),
            "feedback_up": up,
            "feedback_down": down,
        }
    return out


def render(ab: dict, experiment: str) -> str:
    keys = ["n", "first_token_p50_ms", "first_token_p95_ms", "total_p50_ms", "total_p95_ms", "hit_rate", "feedback_up", "feedback_down"]
    lines = [
        f"# A/B 实验报告：{experiment}",
        "",
        "| 分组 | " + " | ".join(keys) + " |",
        "|---" * (len(keys) + 1) + "|",
    ]
    for variant, m in sorted(ab.items()):
        row = [variant]
        for key in keys:
            val = m.get(key)
            row.append("-" if val is None else (f"{val:.4f}" if isinstance(val, float) else str(val)))
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "> 解读建议：先看 p95 延迟差是否达到业务预期，再看 hit_rate 的差值区间；",
        "> 样本量不足以收敛前（n_hit_labeled < 50）不建议据此做全量决策。",
    ]
    return "\n".join(lines)


def _load_from_mongo(experiment: str):
    from app.clients.mongo_history_utils import get_mongo_client  # 延迟导入，离线模式不依赖

    client = get_mongo_client()
    import tempfile

    from app.experiments.metrics_recorder import export_to_jsonl

    tmp = tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False)
    export_to_jsonl(client, experiment, tmp.name)
    tmp.close()
    return Path(tmp.name).read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="A/B 实验分组分析")
    parser.add_argument("--input", help="指标 jsonl 路径（MongoDB 导出或手写）")
    parser.add_argument("--mongo", action="store_true", help="直连 MongoDB 读取")
    parser.add_argument("--experiment", default="", help="实验名（mongo 模式必填）")
    parser.add_argument("--out", default="", help="报告输出路径（markdown）")
    args = parser.parse_args()

    if args.mongo:
        if not args.experiment:
            parser.error("--mongo 模式需要 --experiment")
        raw = _load_from_mongo(args.experiment)
    elif args.input:
        raw = Path(args.input).read_text(encoding="utf-8")
    else:
        parser.error("需要 --input 或 --mongo")

    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not records:
        print("无指标记录")
        sys.exit(1)

    experiment = args.experiment or records[0].get("experiment", "unknown")
    report = render(aggregate(records), experiment)
    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告已写入: {args.out}")


if __name__ == "__main__":
    main()
