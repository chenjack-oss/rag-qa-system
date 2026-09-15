"""离线检索评测入口：标注集 + 检索 dump -> 指标报告（markdown）

用法：

    python evaluation/run_retrieval_eval.py \
        --dataset evaluation/data/eval_dataset.sample.jsonl \
        --dump evaluation/dumps/retrieval_dump.sample.jsonl \
        --k 5 --out evaluation/results/eval_report.md

数据来源（两条获取路径，均真实可用）：
1. 线上收集：设置环境变量 EVAL_DUMP_PATH=/path/to/dump.jsonl 启动查询服务，
   每次问答会自动把检索 TopK（含 chunk_id 顺序）追加写入该文件（见
   app/utils/eval_dump_utils.py）。
2. 回放收集：python evaluation/dump_retrieval.py --dataset ... --out ...
   离线批量回放标注集（需本地 Milvus / 模型已就绪）。
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# 保证可直接运行（python evaluation/run_retrieval_eval.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import evaluate_config, load_jsonl, render_report  # noqa: E402


def group_by_config(dump_records):
    grouped = defaultdict(list)
    for rec in dump_records:
        grouped[rec.get("config", "default")].append(rec)
    return grouped


def main():
    parser = argparse.ArgumentParser(description="离线检索评测")
    parser.add_argument("--dataset", required=True, help="标注集 jsonl 路径")
    parser.add_argument("--dump", required=True, help="检索 dump jsonl 路径")
    parser.add_argument("--k", type=int, default=5, help="截断位置 K（默认 5）")
    parser.add_argument("--out", default="", help="报告输出路径（markdown）；为空则仅打印")
    parser.add_argument("--title", default="检索评测报告", help="报告标题")
    args = parser.parse_args()

    dataset = load_jsonl(args.dataset)
    dataset_by_qid = {d["qid"]: d for d in dataset if d.get("qid")}
    dump_records = load_jsonl(args.dump)
    if not dataset_by_qid or not dump_records:
        print("标注集或 dump 为空，退出")
        sys.exit(1)

    per_config = {}
    for config, records in sorted(group_by_config(dump_records).items()):
        per_config[config] = evaluate_config(records, dataset_by_qid, k=args.k)

    report = render_report(per_config, k=args.k, title=args.title)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"报告已写入: {out_path}")


if __name__ == "__main__":
    main()
