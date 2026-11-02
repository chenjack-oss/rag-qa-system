# 评测体系（Evaluation）

本目录提供一套**可离线复现**的检索评测与消融实验框架，用于量化检索链路（RRF 融合 / FlagReranker 精排 / HyDE / 稀疏向量）的真实贡献，支撑上线决策。

## 设计原则

1. **指标计算全部是纯函数**（`metrics.py`），不依赖 Milvus / LLM / 网络，可直接进 CI 单测；
2. **线上采集与离线评测解耦**：服务端只负责把检索 TopK 追加写入 jsonl（`EVAL_DUMP_PATH` 开关，默认关闭零开销），指标计算与报告生成全部离线完成；
3. **不做假数字**：本目录下 `results/*.sample.md` 由 `make_sample_data.py` 生成的**样例数据**真实运行产出，仅演示格式与解读方式；线上指标请使用自己的业务标注集复现。

## 目录结构

```
evaluation/
├── metrics.py                # 检索层指标库（hit@k / recall@k / MRR / nDCG@k / 可溯源率）
├── gen_metrics.py            # 生成层指标库（忠实度 / 相关性 / 正确性 / 引用 / 延迟分位）
├── run_retrieval_eval.py     # 检索评测入口：标注集 + dump -> 指标报告
├── run_gen_eval.py           # 生成评测入口：answers.jsonl -> 忠实度等指标报告
├── run_ablation.py           # 消融实验：多配置对比 + 相对 full 的 Δ 值
├── badcase_mining.py         # badcase 自动归因：失败样本分类 + 建议排查层
├── compare_eval.py           # 回归门禁：两次评测逐指标 Δ，跌破阈值 exit 1
├── dump_retrieval.py         # 标注集批量回放采集（调用在线服务）
├── make_sample_data.py       # 生成样例数据（演示格式用）
├── data/eval_dataset.sample.jsonl     # 标注集样例（10 条）
├── data/answers.sample.jsonl          # 生成评测样例（10 条问答，含 2 条编造负例）
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

生成类指标（忠实度/答案相关性/正确性）由 `gen_metrics.py` + `run_gen_eval.py` 提供，见下一节。

## 生成层评测（faithfulness / 相关性 / 正确性 / 引用）

检索评测只能证明"对的 chunk 被召回"，证明不了"答案没编"。生成层评测独立成一套指标，**两层分开跑**，避免把召回问题误判成生成问题。

```bash
# 规则口径（离线、零依赖，默认）
python evaluation/run_gen_eval.py --answers evaluation/data/answers.sample.jsonl \
    --out evaluation/results/gen_eval_report.md --metrics-json evaluation/results/gen_eval.json

# LLM-as-judge（沿用 app/lm/lm_utils.get_llm_client，需 .env 里的 OPENAI_*，仓库不落 key）
python evaluation/run_gen_eval.py --answers ... --judge llm --out ...
```

**answers.jsonl 契约**（一行一次问答）：

```json
{"qid": "q001", "config": "baseline", "query": "HAK 180 的最大加热温度",
 "answer": "最大加热温度为 200℃ [c_0001]。", "golden_answer": "200℃", "latency_ms": 8600,
 "retrieved": [{"chunk_id": "c_0001", "text": "...加热温度上限 200℃..."}]}
```

| 指标 | 规则口径算法 | 说明 |
|---|---|---|
| `faithfulness` | 答案切句成 claim，每个 claim 抽关键要素（数字/型号/中文 bigram），在检索块中覆盖率 ≥ 0.6 记为有支撑 | 幻觉主指标，`hallucination_rate = 1 - faithfulness` |
| `relevance` | query 去虚词后的 2-gram 在答案中的覆盖率 | 答非所问检测 |
| `correctness` | 有标准答案：数字/型号全命中=exact(1.0)，覆盖 ≥0.5=partial(0.5)；纯文字标准答案按实义字符覆盖 ≥0.9=exact；无标准答案退化为与检索块的 3-gram Jaccard | 效果是否真的变好 |
| `citation_rate` / `traceability` | `[c_xxx]` 标记落在检索结果内 / 全部引用均可回溯（复用 `answer_traceability_rate`） | 引用可信度 |
| `uncited_rate` | 无任何引用标记的答案占比 | 引用缺失单列，不被合法率掩盖 |
| `p50 / p95` | 线性插值分位数 | 效果-成本权衡，avg 会掩盖长尾 |

**口径红线**：rules 与 llm 两套数字不可直接对比（规则偏保守、对同义改写不敏感）；报告会打印 `judge=` 与 degraded 数量，引用时必须带上。样例数字只用于演示格式，不得作为线上效果对外表述。

## badcase 闭环

```bash
python evaluation/badcase_mining.py --dataset evaluation/data/eval_dataset.sample.jsonl \
    --dump evaluation/dumps/retrieval_dump.sample.jsonl \
    --answers evaluation/data/answers.sample.jsonl \
    --out evaluation/results/badcase_report.md
```

自动把失败样本分成四类并按严重度排序，每条给出**建议排查层**与判定依据：

| 类型 | 判定 | 建议排查层 |
|---|---|---|
| `miss` | Top-K 内没有任何相关块 | 召回层（切片质量 / 稀疏权重 / 过滤表达式） |
| `untraceable` | 答案引用的 chunk 不在检索结果内 | 生成层（引用注入与校验） |
| `low_rank` | 相关块命中，但首个排名 > 3 | 精排层（reranker 分数分布 / RRF 权重） |
| `no_citation` | 答案没有任何引用标记 | 生成层（Prompt 引用约束） |

## 回归门禁

评测不能只看一眼就过。`compare_eval.py` 把两次评测逐指标对比，跌破阈值直接 `exit 1`，可挂 CI：

```bash
# 基线 & 候选各存一份指标
python evaluation/run_retrieval_eval.py --dataset ... --dump base.jsonl --metrics-json base.json
python evaluation/run_retrieval_eval.py --dataset ... --dump new.jsonl  --metrics-json new.json
# 门禁：任一比例型指标跌幅 > 0.02 即失败
python evaluation/compare_eval.py --baseline base.json --candidate new.json --fail-if-drop 0.02
```

门禁只覆盖 0~1 的比例型指标；延迟与计数类指标（`n`/`n_missing`）只展示、不卡阈值——毫秒量纲与绝对阈值不可比，硬卡会出假警报。
