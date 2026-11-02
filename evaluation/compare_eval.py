"""评测回归门禁：对比两次评测结果，指标跌破阈值就让 CI 失败

解决的问题：评测报告是"看一眼就过"的静态产物——没人拦得住指标回退。
本脚本把两次评测结果做逐指标 Δ 对比，超过阈值直接 `exit 1`，可挂进 CI。

支持两种输入（按扩展名自动识别）：
- `*.json`：`run_retrieval_eval.py --metrics-json` / `run_gen_eval.py --metrics-json` 的产物
- `*.jsonl`：检索 dump，需同时给 `--dataset`，现场重算指标

用法：

    # 1) 存基线
    python evaluation/run_retrieval_eval.py --dataset ... --dump base.jsonl \
        --metrics-json results/base.json
    # 2) 改动后重跑，存候选
    python evaluation/run_retrieval_eval.py --dataset ... --dump new.jsonl \
        --metrics-json results/new.json
    # 3) 门禁：任一比例型指标跌幅 > 0.02 即失败
    python evaluation/compare_eval.py --baseline results/base.json \
        --candidate results/new.json --fail-if-drop 0.02

门禁只覆盖 0~1 的比例型指标；延迟单独展示、不参与门禁（毫秒量纲与绝对阈值
不可比，硬卡会出假警报）。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import evaluate_config, load_jsonl  # noqa: E402

# 延迟类指标：越低越好，且不参与比例门禁
LATENCY_KEYS = {"avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "latency_ms"}
# 计数类指标：不是比例，不参与门禁（样本数变化本身不代表质量回退）
COUNT_KEYS = {"n", "n_missing", "n_claims"}
PRIVATE_PREFIX = "_"


def _is_latency(key: str) -> bool:
    return key in LATENCY_KEYS or "latency" in key


def _flatten(payload, k: int) -> Dict[str, dict]:
    """把各种 metrics json 形态统一成 {config: {metric: value}}。"""
    if "per_config" in payload:
        return payload["per_config"]
    if all(isinstance(v, dict) for v in payload.values()):
        return payload
    raise ValueError("无法识别的 metrics json 结构：缺少 per_config")


def load_metrics_file(path: str | Path, dataset: str = "", k: int = 5) -> Dict[str, dict]:
    """按扩展名载入指标：json 直接读，jsonl 视为 dump 现场重算。"""
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _flatten(payload, k)

    if not dataset:
        raise ValueError(f"{path.name} 是 dump，必须同时提供 --dataset 才能重算指标")

    dataset_by_qid = {d["qid"]: d for d in load_jsonl(dataset) if d.get("qid")}
    grouped: Dict[str, List[dict]] = {}
    for rec in load_jsonl(path):
        grouped.setdefault(rec.get("config", "default"), []).append(rec)
    return {config: evaluate_config(rows, dataset_by_qid, k=k) for config, rows in grouped.items()}


def _public_metrics(metrics: dict) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if not key.startswith(PRIVATE_PREFIX) and isinstance(value, (int, float))
    }


def compare_configs(baseline: Dict[str, dict], candidate: Dict[str, dict]) -> List[dict]:
    """逐配置、逐指标对比；只比较双方都有的配置与指标。"""
    rows: List[dict] = []
    for config in sorted(set(baseline) & set(candidate)):
        base_metrics = _public_metrics(baseline[config])
        cand_metrics = _public_metrics(candidate[config])
        for key in sorted(set(base_metrics) & set(cand_metrics)):
            base_value = base_metrics[key]
            cand_value = cand_metrics[key]
            delta = round(cand_value - base_value, 4)
            rows.append(
                {
                    "config": config,
                    "metric": key,
                    "baseline": round(base_value, 4),
                    "candidate": round(cand_value, 4),
                    "delta": delta,
                    "lower_is_better": _is_latency(key),
                    "regressed": (delta > 0) if _is_latency(key) else (delta < 0),
                }
            )
    return rows


def check_regressions(rows: List[dict], max_drop: float) -> List[dict]:
    """返回触发门禁的行（比例型指标跌幅 > max_drop）。延迟不参与门禁。"""
    if max_drop <= 0:
        return []
    hits = []
    for row in rows:
        if row["lower_is_better"] or row["metric"] in COUNT_KEYS:
            continue
        if row["delta"] < -max_drop:
            hits.append(row)
    return hits


def render_comparison(
    rows: List[dict],
    baseline_name: str,
    candidate_name: str,
    max_drop: float = 0.0,
    regressions: List[dict] | None = None,
) -> str:
    """渲染 markdown 对比表；有门禁时给 PASS/FAIL 结论。"""
    changed = [r for r in rows if abs(r["delta"]) > 1e-9]
    lines = [
        "# 评测对比报告",
        "",
        f"- 基线：`{baseline_name}`；候选：`{candidate_name}`",
        f"- 变化指标 {len(changed)}/{len(rows)}（Δ = 候选 - 基线；延迟类越低越好，不参与门禁）",
    ]
    if max_drop > 0:
        verdict = "FAIL" if regressions else "PASS"
        lines.append(f"- 门禁阈值 max_drop={max_drop} → **{verdict}**（{len(regressions or [])} 项超标）")

    lines += [
        "",
        "| 配置 | 指标 | 基线 | 候选 | Δ | 方向 |",
        "|---|---|---|---|---|---|",
    ]
    if len(changed) < len(rows):
        lines.append(f"| — | 其余 {len(rows) - len(changed)} 项指标无变化，已省略 | | | | |")
    for row in changed:
        arrow = "↑" if row["delta"] > 0 else ("↓" if row["delta"] < 0 else "=")
        flag = ""
        if not row["lower_is_better"] and row["delta"] < -1e-9:
            flag = " ⚠️"
        elif row["lower_is_better"] and row["delta"] > 1e-9:
            flag = " ⚠️"
        lines.append(
            f"| {row['config']} | {row['metric']} | {row['baseline']:.4f} | "
            f"{row['candidate']:.4f} | {row['delta']:+.4f} | {arrow}{flag} |"
        )

    if regressions:
        lines += ["", "## 门禁失败项", "", "| 配置 | 指标 | 跌幅 |", "|---|---|---|"]
        for row in regressions:
            lines.append(f"| {row['config']} | {row['metric']} | {row['delta']:+.4f} |")

    lines += [
        "",
        "> 复现：`python evaluation/compare_eval.py --baseline <base> --candidate <new> --fail-if-drop 0.02`",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="评测回归门禁")
    parser.add_argument("--baseline", required=True, help="基线 metrics json 或 dump jsonl")
    parser.add_argument("--candidate", required=True, help="候选 metrics json 或 dump jsonl")
    parser.add_argument("--dataset", default="", help="标注集（输入为 dump 时必填）")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fail-if-drop", type=float, default=0.0, help="比例型指标允许的最大跌幅，>0 才启用门禁")
    parser.add_argument("--out", default="", help="markdown 输出路径")
    parser.add_argument("--json", default="", help="对比结果输出路径（json）")
    args = parser.parse_args()

    try:
        baseline = load_metrics_file(args.baseline, dataset=args.dataset, k=args.k)
        candidate = load_metrics_file(args.candidate, dataset=args.dataset, k=args.k)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"载入评测结果失败：{exc}")
        sys.exit(2)

    rows = compare_configs(baseline, candidate)
    if not rows:
        print("两次结果没有可比对的公共配置/指标，退出")
        sys.exit(2)

    regressions = check_regressions(rows, args.fail_if_drop)
    report = render_comparison(rows, args.baseline, args.candidate, args.fail_if_drop, regressions)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"对比报告已写入: {out_path}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rows": rows, "regressions": regressions, "fail_if_drop": args.fail_if_drop}
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"对比结果已写入: {json_path}")

    if regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
