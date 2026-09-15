# 评测体系（Evaluation）

本目录提供一套**可离线复现**的检索评测与消融实验框架，用于量化检索链路（RRF 融合 / FlagReranker 精排 / HyDE / 稀疏向量）的真实贡献，支撑上线决策。

## 设计原则

1. **指标计算全部是纯函数**（`metrics.py`），不依赖 Milvus / LLM / 网络，可直接进 CI 单测；
2. **线上采集与离线评测解耦**：服务端只负责把检索 TopK 追加写入 jsonl（`EVAL_DUMP_PATH` 开关，默认关闭零开销），指标计算与报告生成全部离线完成；
3. **不做假数字**：本目录下 `results/*.sample.md` 由 `make_sample_data.py` 生成的**样例数据**真实运行产出，仅演示格式与解读方式；线上指标请使用自己的业务标注集复现。

## 目录结构

```
evaluation/
├── metrics.py                # 指标库（hit@k / recall@k / MRR / nDCG@k / 可溯源率）
├── run_retrieval_eval.py     # 离线评测入口：标注集 + dump -> 指标报告
├── run_ablation.py           # 消融实验：多配置对比 + 相对 full 的 Δ 值
├── dump_retrieval.py         # 标注集批量回放采集（调用在线服务）
├── make_sample_data.py       # 生成样例数据（演示格式用）
├── data/eval_dataset.sample.jsonl     # 标注集样例（10 条）
├── dumps/retrieval_dump.sample.jsonl  # 检索 dump 样例（4 配置 × 10 条）
└── results/*.sample.md                # 样例报告（真实运行产出，数据为样例）
```

## 数据契约

**标注集**（人工标注，一行一条）：

```json
{"qid": "q001", "query": "HAK 180 烫金机的最大加热温度是多少", "item_name": "HAK 180 烫金机", "relevant_chunk_ids": ["c_0001", "c_0002"]}
```

**检索 dump**（服务端自动采集，一行一次检索）：

```json
{"qid": "q001", "query": "...", "config": "full", "ranked_chunk_ids": ["c_0001", "..."], "latency_ms": 8310, "retrieved_at": "2026-09-15T10:00:00+00:00"}
```

## 复现流程

### 1. 标注评测集

按数据契约人工标注（建议 50~100 条，覆盖单跳事实、多跳关系、口语化表述、型号精确匹配四类）。`relevant_chunk_ids` 建议由两人独立标注、仲裁分歧项。

### 2. 线上采集 dump

```bash
# 服务端开启采集（默认关闭）
EVAL_DUMP_PATH=./retrieval_dump.jsonl python -m app.query_process.api.query_service

# 批量回放标注集
python evaluation/dump_retrieval.py --dataset <你的标注集> --api http://127.0.0.1:8001/query
```

### 3. 生成评测报告

```bash
python evaluation/run_retrieval_eval.py \
    --dataset <标注集> --dump <dump文件> --k 5 \
    --out evaluation/results/eval_report.md
```

### 4. 消融实验

每轮用不同 `EVAL_ABLATION_MODE` 重启服务并回放采集（一次进程一个配置）：

```bash
EVAL_DUMP_PATH=./dump_no_rerank.jsonl EVAL_ABLATION_MODE=no_rerank python -m app.query_process.api.query_service
python evaluation/dump_retrieval.py --dataset <标注集> ...

# 汇总对比（把各配置 dump 合并进一个 jsonl 即可）
python evaluation/run_ablation.py --dataset <标注集> --dump <合并dump> --out evaluation/results/ablation_report.md
```

消融配置说明：

| config | 裁剪内容 | 预期退化方向 |
|---|---|---|
| `no_rrf` | RRF 退化为仅稠密向量单路 | 多跳/交叉文档召回下降 |
| `no_rerank` | 跳过 FlagReranker 精排，保留融合序 | Top1 精度（MRR）下降最明显 |
| `no_hyde` | 跳过 HyDE 检索路 | 口语化/模糊表述召回下降 |
| `no_sparse` | 稀疏向量权重置 0（1.0/0.0） | 型号编码等精准匹配下降 |

## 指标口径

- **hit@k**：Top-K 中至少一个相关块的比例（对应汇报中的「Top-5 命中率」）
- **recall@k**：Top-K 覆盖的相关块比例
- **MRR**：第一个相关块排名倒数的均值，衡量 Top1 精度
- **nDCG@k**：考虑位置衰减的整体排序质量
- **答案可溯源率**（`answer_traceability_rate`）：答案引用的 chunk 是否全部在检索结果内，是幻觉监控的先行指标

生成类指标（忠实度/答案相关性）需要 LLM-as-judge，计划在 `run_gen_eval.py` 中提供（依赖 API Key，不入 CI）。

## badcase 闭环

评测中每个 `qid` 的 dump 都保留了 `ranked_chunk_ids` 全序，可直接与标注对齐定位：

- **hit@k 未命中的样本** → 检查召回层（过滤表达式 / 稀疏权重 / 切片质量）
- **命中但排名靠后的样本** → 检查融合与精排层（RRF 权重 / reranker 分数分布）
- **可溯源率低的样本** → 检查生成层 Prompt 约束与引用注入
