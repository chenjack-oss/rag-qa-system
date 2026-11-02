"""badcase 自动归因：把"指标掉了"翻译成"下一刀改哪一层"

检索/生成评测跑完只得到一张表，看完没人知道从哪下手。本模块把失败样本自动
分四类，每条给出**建议排查层**与**判定依据**，输出可直接贴进 issue 的 markdown。

分类（按严重度从高到低）：

| 类型 | 判定 | 建议排查层 |
|---|---|---|
| `miss` | Top-K 内没有任何相关块 | 召回层（切片质量 / 稀疏权重 / 过滤表达式） |
| `untraceable` | 答案引用了检索结果之外的 chunk | 生成层（引用注入 / prompt 约束） |
| `low_rank` | 相关块命中，但首个相关块排名 > rank_threshold | 精排层（reranker 分数分布 / RRF 权重） |
| `no_citation` | 答案没有任何 `[c_xxx]` 引用标记 | 生成层（Prompt 引用约束） |

输入：
- `dataset`：标注集，`{qid: {relevant_chunk_ids: [...]}}`
- `dump`：检索 dump，一行 `{qid, config, ranked_chunk_ids}`
- `answers`（可选）：answers.jsonl，提供后才会判 `untraceable` / `no_citation`
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.gen_metrics import extract_citations  # noqa: E402
from evaluation.metrics import load_jsonl  # noqa: E402

SEVERITY = {"miss": 0, "untraceable": 1, "low_rank": 2, "no_citation": 3}
TYPE_LABEL = {
    "miss": "未召回",
    "untraceable": "引用越界",
    "low_rank": "排名靠后",
    "no_citation": "无引用",
}
LAYER = {
    "miss": "召回层（切片质量 / 稀疏权重 / 过滤表达式）",
    "untraceable": "生成层（引用注入与校验）",
    "low_rank": "精排层（reranker 分数分布 / RRF 权重）",
    "no_citation": "生成层（Prompt 引用约束）",
}


def _first_relevant_rank(ranked: Sequence[str], relevant: set[str]) -> int | None:
    for rank, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return rank
    return None


def diagnose(
    label: dict,
    dump_record: dict,
    answer: str = "",
    k: int = 5,
    rank_threshold: int = 3,
) -> List[dict]:
    """对单条 (标注, dump, 答案) 归因，返回命中的全部问题（可能多条）。

    :param rank_threshold: 首个相关块排名超过该值视为 low_rank（默认 3，即 Top3 之外）
    """
    relevant = set(label.get("relevant_chunk_ids") or [])
    ranked = list(dump_record.get("ranked_chunk_ids") or [])
    qid = dump_record.get("qid", label.get("qid", "?"))
    config = dump_record.get("config", "default")
    issues: List[dict] = []

    if not relevant:
        return issues

    topk = ranked[:k]
    first_rank = _first_relevant_rank(ranked, relevant)
    if not any(cid in relevant for cid in topk):
        missing = [cid for cid in relevant if cid not in topk]
        issues.append(
            {
                "qid": qid,
                "config": config,
                "type": "miss",
                "detail": f"Top{k} 无相关块",
                "layer": LAYER["miss"],
                "evidence": f"未召回 {','.join(missing[:3])}",
            }
        )
    elif first_rank is not None and first_rank > rank_threshold:
        issues.append(
            {
                "qid": qid,
                "config": config,
                "type": "low_rank",
                "detail": f"首个相关块 rank={first_rank}",
                "layer": LAYER["low_rank"],
                "evidence": f"MRR={1.0 / first_rank:.3f}，Top{rank_threshold} 内被无关块挤占",
            }
        )

    if answer:
        cited = extract_citations(answer)
        retrieved = set(ranked)
        invalid = [cid for cid in cited if cid not in retrieved]
        if cited and invalid:
            issues.append(
                {
                    "qid": qid,
                    "config": config,
                    "type": "untraceable",
                    "detail": f"{len(invalid)}/{len(cited)} 条引用不在检索结果内",
                    "layer": LAYER["untraceable"],
                    "evidence": "越界引用 " + ",".join(invalid[:3]),
                }
            )
        elif not cited:
            issues.append(
                {
                    "qid": qid,
                    "config": config,
                    "type": "no_citation",
                    "detail": "答案无引用标记",
                    "layer": LAYER["no_citation"],
                    "evidence": f"答案长度 {len(answer)} 字，无法回溯到 chunk",
                }
            )
    return issues


def mine_badcases(
    dump_records: Iterable[dict],
    dataset_by_qid: Dict[str, dict],
    answers_by_qid: Dict[str, str] | None = None,
    k: int = 5,
    rank_threshold: int = 3,
) -> dict:
    """批量归因。

    :return: {"items": [...], "n_total": 样本数, "n_labeled": 有效标注数, "by_type": {...}}
    """
    answers_by_qid = answers_by_qid or {}
    items: List[dict] = []
    seen = 0
    labeled = 0

    for rec in dump_records:
        qid = rec.get("qid")
        label = dataset_by_qid.get(qid)
        if label is None:
            continue
        seen += 1
        if label.get("relevant_chunk_ids"):
            labeled += 1
        items.extend(
            diagnose(
                label,
                rec,
                answer=answers_by_qid.get(qid, ""),
                k=k,
                rank_threshold=rank_threshold,
            )
        )

    items.sort(key=lambda it: (SEVERITY.get(it["type"], 9), it["qid"]))
    by_type: Dict[str, int] = defaultdict(int)
    for item in items:
        by_type[item["type"]] += 1
    return {"items": items, "n_total": seen, "n_labeled": labeled, "by_type": dict(by_type)}


def render_badcases(result: dict, k: int = 5) -> str:
    """渲染 markdown：先给分类计数，再给逐条清单。"""
    items = result.get("items") or []
    n_total = result.get("n_total", 0)
    n_labeled = result.get("n_labeled", 0)
    by_type = result.get("by_type") or {}

    lines = [
        "# badcase 归因清单",
        "",
        f"- 样本 {n_total} 条，其中有标注 {n_labeled} 条；命中问题 **{len(items)}** 条",
        f"- 判定口径：k={k}；miss=Top{k} 无相关块；low_rank=首个相关块排名 > 3",
        "",
        "| 类型 | 条数 | 建议排查层 |",
        "|---|---|---|",
    ]
    for key in sorted(SEVERITY, key=lambda t: SEVERITY[t]):
        if by_type.get(key):
            lines.append(f"| {TYPE_LABEL[key]}（`{key}`） | {by_type[key]} | {LAYER[key]} |")

    if not items:
        lines += ["", "> 本轮没有命中问题样本。"]
        return "\n".join(lines)

    lines += [
        "",
        "| qid | 配置 | 类型 | 现状 | 建议排查层 | 依据 |",
        "|---|---|---|---|---|---|",
    ]
    for item in items:
        lines.append(
            "| {qid} | {config} | {type_label} | {detail} | {layer} | {evidence} |".format(
                qid=item["qid"],
                config=item["config"],
                type_label=TYPE_LABEL.get(item["type"], item["type"]),
                detail=item["detail"],
                layer=item["layer"],
                evidence=item["evidence"],
            )
        )
    lines += ["", "> 用法：`python evaluation/badcase_mining.py --dataset ... --dump ... [--answers ...]`"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="badcase 自动归因")
    parser.add_argument("--dataset", required=True, help="标注集 jsonl 路径")
    parser.add_argument("--dump", required=True, help="检索 dump jsonl 路径")
    parser.add_argument("--answers", default="", help="answers.jsonl（可选，用于判定引用类问题）")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rank-threshold", type=int, default=3)
    parser.add_argument("--out", default="", help="markdown 输出路径")
    parser.add_argument("--json", default="", help="结构化结果输出路径（json）")
    args = parser.parse_args()

    dataset_by_qid = {d["qid"]: d for d in load_jsonl(args.dataset) if d.get("qid")}
    dump_records = load_jsonl(args.dump)
    if not dataset_by_qid or not dump_records:
        print("标注集或 dump 为空，退出")
        sys.exit(1)

    answers_by_qid = {}
    if args.answers:
        from evaluation.gen_metrics import load_answers

        answers_by_qid = {r["qid"]: r.get("answer", "") for r in load_answers(args.answers) if r.get("qid")}

    result = mine_badcases(
        dump_records,
        dataset_by_qid,
        answers_by_qid=answers_by_qid,
        k=args.k,
        rank_threshold=args.rank_threshold,
    )
    report = render_badcases(result, k=args.k)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"清单已写入: {out_path}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结构化结果已写入: {json_path}")


if __name__ == "__main__":
    main()
