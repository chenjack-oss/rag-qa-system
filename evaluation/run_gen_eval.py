"""生成层评测入口：answers.jsonl -> 忠实度/相关性/正确性/引用率 + markdown 报告

用法：

    # 规则口径（离线、零依赖，默认）
    python evaluation/run_gen_eval.py --answers evaluation/data/answers.sample.jsonl \
        --out evaluation/results/gen_eval_report.md

    # LLM-as-judge 口径（需要 .env 里的 OPENAI_BASE_URL / OPENAI_API_KEY，
    # 沿用 app/lm/lm_utils.get_llm_client，仓库不落任何 key）
    python evaluation/run_gen_eval.py --answers ... --judge llm --out ...

与检索评测的分工：
- 检索层（metrics.py）回答"对的 chunk 有没有被召回"；
- 生成层（本脚本）回答"答案有没有编、有没有答非所问、引用是否站得住"。
两层分开跑，避免把召回问题误判成生成问题。

LLM judge 失败（限流/超时/返回非 JSON）时**逐条回退 rules** 并在报告标注
degraded 数量，不会整份报告失败——评测工具自身不能成为链路单点。
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# 保证可直接运行（python evaluation/run_gen_eval.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.gen_metrics import (  # noqa: E402
    answer_correctness,
    answer_relevance,
    chunk_ids,
    evaluate_answers,
    extract_citations,
    faithfulness,
    load_answers,
    render_gen_report,
)
from evaluation.metrics import answer_traceability_rate  # noqa: E402

JUDGE_PROMPT = """你是 RAG 系统的答案质量评审。只输出 JSON，不要解释。

问题：{query}
答案：{answer}
检索到的资料：
{context}

请依据资料评分（0~1 的小数）：
- faithfulness：答案中的事实性陈述有多大比例能在资料中找到支撑（编造即扣分）
- relevance：答案是否正面回答了问题
- correctness：答案是否正确（有参考答案时以其为准）

输出格式：{{"faithfulness": 0.0, "relevance": 0.0, "correctness": 0.0, "reason": "一句话"}}
参考答案：{golden}
"""


def _parse_json_block(text: str) -> dict:
    """从模型输出中抠出第一个 JSON 对象（兼容 ```json 包裹与前后废话）。"""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("模型输出中没有 JSON 对象")
        text = text[start : end + 1]
    return json.loads(text)


def _clamp(value) -> float:
    try:
        return round(min(1.0, max(0.0, float(value))), 4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"评分不是数字: {value!r}") from exc


class LLMJudge:
    """LLM-as-judge：逐条打分，失败返回空字典由调用方回退 rules。"""

    def __init__(self, llm, model: str = ""):
        self._llm = llm
        self.model = model
        self.calls = 0
        self.failures = 0

    @classmethod
    def from_env(cls) -> "LLMJudge":
        # 延迟导入：只有显式 --judge llm 时才引入 langchain / 读取 .env，
        # 保证 rules 口径与 CI 完全不受重依赖影响。
        from app.conf.lm_config import lm_config
        from app.lm.lm_utils import get_llm_client

        return cls(get_llm_client(), model=lm_config.llm_model or "")

    def score(self, record: dict) -> dict:
        retrieved = record.get("retrieved") or []
        context = "\n".join(f"[{c.get('chunk_id', '?')}] {c.get('text', '')}" for c in retrieved if isinstance(c, dict))
        prompt = JUDGE_PROMPT.format(
            query=record.get("query", ""),
            answer=record.get("answer", ""),
            context=context[:6000],
            golden=record.get("golden_answer", "") or "（无）",
        )
        self.calls += 1
        try:
            response = self._llm.invoke(prompt)
            content = getattr(response, "content", response)
            data = _parse_json_block(str(content))
            return {
                "faithfulness": _clamp(data.get("faithfulness")),
                "relevance": _clamp(data.get("relevance")),
                "correctness": _clamp(data.get("correctness")),
                "reason": str(data.get("reason", ""))[:200],
            }
        except Exception as exc:  # noqa: BLE001 - 评测工具必须容错，逐条回退
            self.failures += 1
            print(f"[LLMJudge] {record.get('qid', '?')} 评分失败，回退 rules：{exc}")
            return {}


def _traceability_rows(records) -> list:
    """构造 answer_traceability_rate 的输入（该指标此前无调用方，此处正式接入）。"""
    rows = []
    for rec in records:
        rows.append(
            {
                "qid": rec.get("qid"),
                "cited_chunk_ids": extract_citations(rec.get("answer", "")),
                "ranked_chunk_ids": chunk_ids(rec.get("retrieved") or []),
            }
        )
    return rows


def evaluate_config(records, judge: LLMJudge | None = None, min_cover: float = 0.6) -> dict:
    """单配置评测：rules 打底，judge 可用时覆盖三项语义指标（引用/延迟仍走规则）。"""
    metrics = evaluate_answers(records, min_cover=min_cover)
    metrics["traceability"] = round(answer_traceability_rate(_traceability_rows(records)), 4)
    metrics["_judge"] = "rules"

    if judge is None:
        return metrics

    judged_f, judged_r, judged_c, degraded = [], [], [], 0
    for rec in records:
        verdict = judge.score(rec)
        if not verdict:
            degraded += 1
            verdict = {
                "faithfulness": faithfulness(rec.get("answer", ""), rec.get("retrieved") or [], min_cover)["faithfulness"],
                "relevance": answer_relevance(rec.get("query", ""), rec.get("answer", "")),
                "correctness": answer_correctness(
                    rec.get("answer", ""), rec.get("golden_answer", ""), rec.get("retrieved") or []
                )["score"],
            }
        judged_f.append(verdict["faithfulness"])
        judged_r.append(verdict["relevance"])
        judged_c.append(verdict["correctness"])

    n = len(judged_f)
    metrics["faithfulness"] = round(sum(judged_f) / n, 4) if n else 0.0
    metrics["hallucination_rate"] = round(1.0 - metrics["faithfulness"], 4)
    metrics["relevance"] = round(sum(judged_r) / n, 4) if n else 0.0
    metrics["correctness"] = round(sum(judged_c) / n, 4) if n else 0.0
    metrics["_judge"] = "llm"
    metrics["_degraded"] = degraded
    return metrics


def main():
    parser = argparse.ArgumentParser(description="生成层评测（忠实度 / 相关性 / 正确性 / 引用）")
    parser.add_argument("--answers", required=True, help="answers.jsonl 路径")
    parser.add_argument("--judge", choices=["rules", "llm"], default="rules", help="评分口径（默认 rules）")
    parser.add_argument("--min-cover", type=float, default=0.6, help="rules 口径下 claim 支撑的覆盖率阈值")
    parser.add_argument("--out", default="", help="报告输出路径（markdown）；为空则仅打印")
    parser.add_argument("--metrics-json", default="", help="指标输出路径（json，供 compare_eval 使用）")
    parser.add_argument("--title", default="生成层评测报告", help="报告标题")
    args = parser.parse_args()

    records = load_answers(args.answers)
    if not records:
        print("answers 文件为空或全部解析失败，退出")
        sys.exit(1)

    judge = LLMJudge.from_env() if args.judge == "llm" else None

    grouped = defaultdict(list)
    for rec in records:
        grouped[rec.get("config", "default")].append(rec)

    per_config = {}
    for config, rows in sorted(grouped.items()):
        per_config[config] = evaluate_config(rows, judge=judge, min_cover=args.min_cover)

    report = render_gen_report(per_config, judge=args.judge, title=args.title)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"报告已写入: {out_path}")

    if args.metrics_json:
        json_path = Path(args.metrics_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"judge": args.judge, "min_cover": args.min_cover, "per_config": per_config}
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"指标已写入: {json_path}")

    if judge is not None:
        print(f"[LLMJudge] 调用 {judge.calls} 次，失败 {judge.failures} 次")


if __name__ == "__main__":
    main()
