"""检索与答案质量指标库（纯函数，可离线运行）

设计原则：
- 所有指标均为纯函数，输入是"标注数据 + 系统输出的检索结果 dump"，不依赖
  Milvus / LLM / 网络服务，可在 CI 中直接单测，也可离线批量回放评测。
- 指标口径与简历/汇报口径一致：hit@k（Top-K 命中率）、MRR、nDCG@k、
  Recall@k、答案可溯源率。
- 幻觉占比（faithfulness）需要 LLM-as-judge，见 run_gen_eval.py（可选，
  需要 API Key），本文件不承担。

数据契约：
- dump 一行 = 一次检索记录（某个实验配置下）：
    {"qid": "...", "config": "full", "ranked_chunk_ids": [...], "latency_ms": 123}
- 标注集一行：
    {"qid": "...", "relevant_chunk_ids": [...], ...}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# ---------------------------------------------------------------- jsonl 读取

def load_jsonl(path: str | Path) -> List[dict]:
    """读取 jsonl 文件；跳过空行与解析失败的行（评测数据脏不应阻塞整体评测）。"""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[load_jsonl] 跳过第 {line_no} 行：JSON 解析失败")
    return records


# ---------------------------------------------------------------- 单条指标

def hit_at_k(ranked_chunk_ids: Sequence[str], relevant: set[str], k: int) -> bool:
    """Top-K 命中：前 k 个结果中至少包含一个相关块。"""
    if not relevant:
        return False
    return any(cid in relevant for cid in ranked_chunk_ids[:k])


def recall_at_k(ranked_chunk_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Recall@K：前 k 个结果覆盖了多少比例的相关块（|命中相关| / |全部相关|）。"""
    if not relevant:
        return 0.0
    topk = set(ranked_chunk_ids[:k])
    return len(topk & relevant) / len(relevant)


def mrr(ranked_chunk_ids: Sequence[str], relevant: set[str]) -> float:
    """MRR：第一个相关块排名倒数的均值基数（单条 = 1/rank，无相关命中为 0）。"""
    for rank, cid in enumerate(ranked_chunk_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_chunk_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """二值相关增益的 nDCG@K（二元相关标注场景的标准实现）。"""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(ranked_chunk_ids[:k])
        if cid in relevant
    )
    # 理想 DCG：所有相关块排在最前
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------- 聚合指标

def evaluate_config(
    dump_records: Iterable[dict],
    dataset_by_qid: Dict[str, dict],
    k: int = 5,
) -> dict:
    """对某个实验配置（config）的 dump 记录聚合计算指标。

    :param dump_records: 同一 config 下的检索记录列表
    :param dataset_by_qid: qid -> 标注记录（须含 relevant_chunk_ids）
    :param k: 截断位置
    :return: {"n": 有效样本数, "hit@k": float, "recall@k": float,
              "mrr": float, "ndcg@k": float, "avg_latency_ms": float}
    """
    hits: List[bool] = []
    recalls: List[float] = []
    rrs: List[float] = []
    ndcgs: List[float] = []
    latencies: List[float] = []
    missing = 0

    for rec in dump_records:
        qid = rec.get("qid")
        label = dataset_by_qid.get(qid)
        if label is None:
            missing += 1
            continue
        relevant = set(label.get("relevant_chunk_ids") or [])
        ranked = rec.get("ranked_chunk_ids") or []
        hits.append(hit_at_k(ranked, relevant, k))
        recalls.append(recall_at_k(ranked, relevant, k))
        rrs.append(mrr(ranked, relevant))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        if rec.get("latency_ms") is not None:
            latencies.append(float(rec["latency_ms"]))

    n = len(hits)
    if missing:
        print(f"[evaluate_config] {missing} 条 dump 记录在标注集中找不到对应 qid，已跳过")
    return {
        "n": n,
        "n_missing": missing,
        f"hit@{k}": sum(hits) / n if n else 0.0,
        f"recall@{k}": sum(recalls) / n if n else 0.0,
        "mrr": sum(rrs) / n if n else 0.0,
        f"ndcg@{k}": sum(ndcgs) / n if n else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def answer_traceability_rate(
    answers: Iterable[dict],
) -> float:
    """答案可溯源率：答案引用（cited_chunk_ids）全部能在检索结果（ranked_chunk_ids）
    中找到的比例。

    answers 一行：
        {"qid": "...", "cited_chunk_ids": [...], "ranked_chunk_ids": [...]}
    引用了检索结果之外的块 = 不可溯源（幻觉疑似信号之一）。
    """
    checked = 0
    traceable = 0
    for ans in answers:
        cited = ans.get("cited_chunk_ids") or []
        retrieved = set(ans.get("ranked_chunk_ids") or [])
        if not cited:
            continue
        checked += 1
        if all(cid in retrieved for cid in cited):
            traceable += 1
    return traceable / checked if checked else 0.0


# ---------------------------------------------------------------- 报告渲染

def render_report(per_config: Dict[str, dict], k: int, title: str = "检索评测报告") -> str:
    """把 evaluate_config 的多配置结果渲染成 markdown 表格。"""
    metric_keys = [f"hit@{k}", f"recall@{k}", "mrr", f"ndcg@{k}", "avg_latency_ms"]
    lines = [
        f"# {title}",
        "",
        f"- 评测口径：hit/recall/ndcg 截断位置 k={k}",
        "- 生成方式：`python evaluation/run_retrieval_eval.py --dataset ... --dump ...`",
        "",
        "| 配置 | 样本数 | " + " | ".join(metric_keys) + " |",
        "|---" * (len(metric_keys) + 2) + "|",
    ]
    for config, m in per_config.items():
        row = [config, str(m.get("n", 0))]
        row += [f"{m.get(key, 0.0):.4f}" for key in metric_keys]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)
