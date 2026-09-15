"""生成评测样例数据（仅用于演示评测流程格式，非真实业务标注）

运行后产出：
- evaluation/data/eval_dataset.sample.jsonl   标注集样例（10 条）
- evaluation/dumps/retrieval_dump.sample.jsonl 四种配置的检索 dump 样例
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 每条：问题 + 相关 chunk（人工标注的 ground truth）
DATASET = [
    {"qid": "q001", "query": "HAK 180 烫金机的最大加热温度是多少", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0001", "c_0002"]},
    {"qid": "q002", "query": "烫金机温度传感器坏了怎么更换", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0003"]},
    {"qid": "q003", "query": "设备开机后显示屏不亮是什么问题", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0004", "c_0005"]},
    {"qid": "q004", "query": "HAK 180 的额定电压和功率参数", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0002", "c_0006"]},
    {"qid": "q005", "query": "怎么清洁烫金头胶辊", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0007"]},
    {"qid": "q006", "query": "更换耗材后需要做哪些校准操作", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0008", "c_0009"]},
    {"qid": "q007", "query": "设备报错 E03 是什么故障", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0010"]},
    {"qid": "q008", "query": "烫金机的日常保养周期是多久", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0009", "c_0011"]},
    {"qid": "q009", "query": "安全操作需要注意哪些事项", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0012"]},
    {"qid": "q010", "query": "烫金压力怎么调节", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0013", "c_0001"]},
]

# 模拟四种配置下系统的 TopK 输出（k<=5）。
# 构造原则：full 最优；各消融配置按真实退化方向劣化（no_hyde 丢口语化召回、
# no_rrf 单路排序差、no_rerank top1 精度下降），用于演示消融报告的解读方式。
DUMPS = {
    "full": {
        "q001": ["c_0001", "c_0002", "c_0013", "c_0006", "c_0011"],
        "q002": ["c_0003", "c_0008", "c_0007", "c_0013", "c_0005"],
        "q003": ["c_0004", "c_0005", "c_0010", "c_0007", "c_0002"],
        "q004": ["c_0006", "c_0002", "c_0001", "c_0011", "c_0009"],
        "q005": ["c_0007", "c_0003", "c_0009", "c_0005", "c_0001"],
        "q006": ["c_0008", "c_0009", "c_0003", "c_0010", "c_0007"],
        "q007": ["c_0010", "c_0004", "c_0005", "c_0013", "c_0006"],
        "q008": ["c_0009", "c_0011", "c_0007", "c_0008", "c_0003"],
        "q009": ["c_0012", "c_0009", "c_0007", "c_0004", "c_0008"],
        "q010": ["c_0013", "c_0001", "c_0002", "c_0007", "c_0010"],
    },
    # 关闭 RRF：仅稠密向量主路 → 部分仅 HyDE 能命中的块丢失（q009、q003 top1 丢失）
    "no_rrf": {
        "q001": ["c_0002", "c_0001", "c_0013", "c_0006", "c_0011"],
        "q002": ["c_0003", "c_0008", "c_0007", "c_0013", "c_0005"],
        "q003": ["c_0005", "c_0010", "c_0007", "c_0002", "c_0012"],
        "q004": ["c_0002", "c_0006", "c_0001", "c_0011", "c_0009"],
        "q005": ["c_0007", "c_0003", "c_0009", "c_0005", "c_0001"],
        "q006": ["c_0008", "c_0003", "c_0009", "c_0010", "c_0007"],
        "q007": ["c_0010", "c_0004", "c_0005", "c_0013", "c_0006"],
        "q008": ["c_0009", "c_0007", "c_0008", "c_0003", "c_0011"],
        "q009": ["c_0009", "c_0007", "c_0004", "c_0008", "c_0012"],
        "q010": ["c_0001", "c_0013", "c_0002", "c_0007", "c_0010"],
    },
    # 关闭精排：保留 RRF 融合序 → top1 精度下降（MRR 下降），hit@5 基本不变
    "no_rerank": {
        "q001": ["c_0013", "c_0001", "c_0002", "c_0006", "c_0011"],
        "q002": ["c_0008", "c_0003", "c_0007", "c_0013", "c_0005"],
        "q003": ["c_0010", "c_0004", "c_0005", "c_0007", "c_0002"],
        "q004": ["c_0001", "c_0006", "c_0002", "c_0011", "c_0009"],
        "q005": ["c_0003", "c_0007", "c_0009", "c_0005", "c_0001"],
        "q006": ["c_0003", "c_0008", "c_0009", "c_0010", "c_0007"],
        "q007": ["c_0004", "c_0010", "c_0005", "c_0013", "c_0006"],
        "q008": ["c_0011", "c_0009", "c_0007", "c_0008", "c_0003"],
        "q009": ["c_0009", "c_0012", "c_0007", "c_0004", "c_0008"],
        "q010": ["c_0002", "c_0013", "c_0001", "c_0007", "c_0010"],
    },
    # 关闭 HyDE：口语化表述（q003 显示屏不亮、q009 安全事项）召回退化
    "no_hyde": {
        "q001": ["c_0001", "c_0002", "c_0013", "c_0006", "c_0011"],
        "q002": ["c_0003", "c_0008", "c_0013", "c_0005", "c_0011"],
        "q003": ["c_0010", "c_0002", "c_0013", "c_0006", "c_0011"],
        "q004": ["c_0006", "c_0002", "c_0001", "c_0011", "c_0009"],
        "q005": ["c_0007", "c_0003", "c_0009", "c_0005", "c_0001"],
        "q006": ["c_0008", "c_0009", "c_0010", "c_0007", "c_0006"],
        "q007": ["c_0010", "c_0004", "c_0005", "c_0013", "c_0006"],
        "q008": ["c_0009", "c_0011", "c_0007", "c_0008", "c_0003"],
        "q009": ["c_0009", "c_0007", "c_0004", "c_0008", "c_0003"],
        "q010": ["c_0013", "c_0001", "c_0002", "c_0007", "c_0010"],
    },
}


def main():
    data_dir = ROOT / "data"
    dumps_dir = ROOT / "dumps"
    data_dir.mkdir(parents=True, exist_ok=True)
    dumps_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "eval_dataset.sample.jsonl", "w", encoding="utf-8") as f:
        for item in DATASET:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(dumps_dir / "retrieval_dump.sample.jsonl", "w", encoding="utf-8") as f:
        for config, per_q in DUMPS.items():
            for item in DATASET:
                rec = {
                    "qid": item["qid"],
                    "query": item["query"],
                    "config": config,
                    "ranked_chunk_ids": per_q[item["qid"]],
                    # 样例延迟仅为格式演示；真实数值来自服务端 dump（latency_ms 字段）
                    "latency_ms": 8600 if config == "full" else (5200 if config == "no_rerank" else 6800),
                    "retrieved_at": "SAMPLE",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"dataset -> {data_dir / 'eval_dataset.sample.jsonl'}")
    print(f"dumps   -> {dumps_dir / 'retrieval_dump.sample.jsonl'}")


if __name__ == "__main__":
    main()
