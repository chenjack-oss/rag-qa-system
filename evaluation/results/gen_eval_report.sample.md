# 生成层评测报告（样例数据演示）

- judge=`rules`（rules 为离线规则口径，与 LLM-as-judge 存在系统性差异，两套数字不可直接对比）
- 忠实度 = 有检索支撑的 claim 数 / 总 claim 数；幻觉率 = 1 - 忠实度
- 正确性：exact=1.0 / partial=0.5 / wrong=0.0 的加权均值（无标准答案样本按 no_golden 单列）
- 引用合法率 = 引用标记落在检索结果内的比例；无引用占比单列（引用缺失不被合法率惩罚）

| 配置 | 样本 | 答案句 | 忠实度 | 幻觉率 | 相关性 | 正确性 | exact 占比 | 引用合法率 | 无引用占比 | 可溯源率 | p95 延迟(ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 10 | 10 | 0.8000 | 0.2000 | 0.3311 | 0.6767 | 0.6000 | 0.9000 | 0.1000 | 1.0000 | 8566 |

## 待排查样本（忠实度或正确性未满分）

- `baseline`：q003（faith=0.00，无支撑句 1/1; correctness=wrong）；q007（faith=0.00，无支撑句 1/1; correctness=wrong）；q010（faith=1.00，correctness=wrong）

> 复现：`python evaluation/run_gen_eval.py --answers evaluation/data/answers.sample.jsonl`
> 样例数据仅演示格式与解读方式，不得作为线上效果对外表述。