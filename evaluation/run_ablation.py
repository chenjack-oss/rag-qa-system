"""消融实验（Ablation Study）：对比检索链路各组件的贡献

思路：固定同一份标注集，分别收集"关闭某个组件"时的检索结果 dump，
与完整链路（full）对比各指标的退化幅度，量化每个组件的真实贡献。

预置消融配置（与 query 链路中的开关一一对应，通过环境变量或实验配置注入）：

| config       | 含义                                   | 对应注入方式                              |
|--------------|----------------------------------------|-------------------------------------------|
| full         | 完整链路：RRF + 重排 + HyDE + 稠稀混合  | 默认                                      |
| no_rrf       | 关闭 RRF，直接用稠密向量单路结果         | EVAL_ABLATION_MODE=no_rrf                 |
| no_rerank    | 关闭 FlagReranker 精排，保留 RRF 融合序  | EVAL_ABLATION_MODE=no_rerank              |
| no_hyde      | 关闭 HyDE 查询增强，仅混合检索一路       | EVAL_ABLATION_MODE=no_hyde                |
| no_sparse    | 关闭稀疏向量，仅稠密向量混合检索         | EVAL_ABLATION_MODE=no_sparse              |

回放采集：python evaluation/dump_retrieval.py --dataset ... --configs full,no_rerank,...
（服务端按 EVAL_ABLATION_MODE 裁剪链路，见 app/utils/eval_dump_utils.py 说明）

本脚本只负责"对比与呈现"：读取多配置 dump，输出每个配置相对 full 的指标差值表。
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import evaluate_config, load_jsonl  # noqa: E402

# 各组件的"期望贡献方向"提示：消融后指标应下降，若不降反升则值得复盘
ABLATION_NOTES = {
    "no_rrf": "关闭 RRF 后退化为单路结果，多路互补信息丢失，hit@k 预期下降",
    "no_rerank": "关闭精排后保留 RRF 融合序，Top1 精度（MRR）预期下降最明显",
    "no_hyde": "关闭 HyDE 后口语化/模糊表述的召回预期下降",
    "no_sparse": "关闭稀疏向量后型号编码等精准匹配预期下降（稀疏权重 0.2 的作用）",
}


def render_ablation(per_config: dict, k: int) -> str:
    metrics = [f"hit@{k}", f"recall@{k}", "mrr", f"ndcg@{k}"]
    baseline = per_config.get("full")
    lines = [
        "# 消融实验报告（Ablation Study）",
        "",
        f"- 截断位置 k={k}；Δ 列为相对 full 配置的变化（负值=该组件被移除后退化）",
        "",
        "| 配置 | 样本数 | " + " | ".join(metrics) + " | avg_latency_ms | 说明 |",
        "|---" * (len(metrics) + 4) + "|",
    ]
    for config, m in per_config.items():
        row = [config, str(m.get("n", 0))]
        for key in metrics:
            val = m.get(key, 0.0)
            if config != "full" and baseline and baseline.get("n"):
                delta = val - baseline.get(key, 0.0)
                row.append(f"{val:.4f} ({delta:+.4f})")
            else:
                row.append(f"{val:.4f}")
        row.append(f"{m.get('avg_latency_ms', 0.0):.0f}")
        row.append(ABLATION_NOTES.get(config, "完整链路基线"))
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "> 复现方式：`python evaluation/run_ablation.py --dataset ... --dump ...`",
        "> 若某组件消融后指标不降反升，优先检查：标注集分布是否偏向该组件、开关实现是否真正裁剪了链路。",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="消融实验对比")
    parser.add_argument("--dataset", required=True, help="标注集 jsonl 路径")
    parser.add_argument("--dump", required=True, help="多配置检索 dump jsonl 路径")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="", help="报告输出路径（markdown）")
    args = parser.parse_args()

    dataset_by_qid = {d["qid"]: d for d in load_jsonl(args.dataset) if d.get("qid")}
    dump_records = load_jsonl(args.dump)

    grouped = defaultdict(list)
    for rec in dump_records:
        grouped[rec.get("config", "default")].append(rec)

    per_config = {
        config: evaluate_config(records, dataset_by_qid, k=args.k)
        for config, records in sorted(grouped.items())
    }
    report = render_ablation(per_config, k=args.k)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"报告已写入: {out_path}")


if __name__ == "__main__":
    main()
