"""生成层质量指标库（纯函数，离线可跑，CI 可直测）

背景：`metrics.py` 只覆盖检索层（hit@k / recall@k / MRR / nDCG@k），能证明
"对的 chunk 被召回"，但证明不了"答案没编"。本模块补上生成层指标：

- faithfulness       答案的每个 claim 有多少能在检索块中找到支撑
- answer_relevance   query 的 2-gram 在答案中的覆盖率（答非所问检测）
- answer_correctness 与标准答案比对（关键要素命中度 / 无标准答案时退化）
- citation_stats     答案引用标记 [c_xxx] 是否落在检索结果内
- latency_percentiles  p50 / p95（现有 metrics 只算 avg，掩盖长尾）

设计原则：
1. 全部纯函数，只吃字符串/字典，不依赖 torch / LLM / Milvus / 网络；
2. 规则口径（judge=rules）**可解释、可复现**：每个判定都能回溯到具体 term；
3. 规则口径与 LLM-as-judge 存在系统性差异（规则偏保守、对同义改写不敏感），
   报告中必须标注 judge 类型，**两套数字不可直接对比**。

数据契约（answers.jsonl 一行 = 一次问答）::

    {"qid": "q001", "query": "HAK 180 的最大加热温度",
     "answer": "最大加热温度为 200℃ [c_0001]。",
     "retrieved": [{"chunk_id": "c_0001", "text": "...加热温度上限 200℃..."}],
     "golden_answer": "200℃", "latency_ms": 8600, "config": "baseline"}
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# ---------------------------------------------------------------- 正则与常量

# 引用标记：[c_0001] / [c_ab-3]
_CITE_RE = re.compile(r"\[(c_[A-Za-z0-9_-]+)\]")
# 句末标点（中英文），用于切 claim
_SENT_SPLIT_RE = re.compile(r"[。！？；!?;\n]+")
# 数字（含小数），单位在归一化后已剥离，故只需数字本身
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# 型号：字母前缀 + 数字，如 HAK180 / E03 / RTX4090（在归一化文本上匹配）
_MODEL_RE = re.compile(r"[a-z]{2,}\d{2,}[a-z]*")
# 英文词：前后不能紧邻字母数字，避免把 "hak180" 拆出 "hak"
_EN_RE = re.compile(r"(?<![a-z0-9])[a-z]{3,}(?![a-z0-9])")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
# 归一化：白名单保留（小写字母 / 数字 / 汉字 / 小数点），其余一律剔除
# 注意必须保留 "."，否则 "0.4" 会变成 "04"，与 "4" 混淆
_NORM_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff.]+")

# claim 切分后丢弃的纯承接句 / 套话（无事实内容，不该计入忠实度分母）
_FILLER_CLAIMS = {
    "综上所述",
    "总之",
    "因此",
    "所以",
    "根据文档",
    "根据上述",
    "根据资料",
    "如上所述",
    "此外",
    "另外",
    "同时",
    "希望对你有帮助",
    "如有疑问",
    "以上",
}

# 高频虚词：由这些字组成的 2-gram 不参与相关性覆盖计算
_STOP_CHARS = set("的了是呢吗么怎哪有些有没和与在中请问下这那个们就都也还很太啊吧呀嘛把被对于以为之其所最多少何如样每并")

def _normalize(text: str) -> str:
    """统一全角/半角、小写、去标点空白，供 term 抽取与覆盖计算使用。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    return _NORM_RE.sub("", text)


def _raw(text: str) -> str:
    """只做 NFKC + 小写，**保留标点**：切 n-gram 前必须靠标点确定短语边界。"""
    return unicodedata.normalize("NFKC", text or "").lower()


def _num_token(raw: str) -> str:
    """数字归一：去掉无意义的小数零（"200.0" 与 "200" 视为同一数字）。"""
    value = raw.rstrip(".")
    if "." in value:
        value = value.rstrip("0").rstrip(".") or "0"
    return value


# ---------------------------------------------------------------- term 抽取

# 套话归一化缓存：避免每次切句重复计算
_FILLER_NORMALIZED = {_normalize(f) for f in _FILLER_CLAIMS}
# 长串优先，避免「根据上述」被「根据」抢先截断
_FILLER_PREFIXES = sorted(_FILLER_CLAIMS, key=len, reverse=True)


def extract_key_terms(text: str, bigram: bool = True) -> set[str]:
    """抽取用于"支撑判定"的关键要素，带前缀避免类型间碰撞。

    - ``n:200``   数字（含小数）
    - ``m:hak180`` 型号（字母+数字）
    - ``w:temperature`` 英文词（>=3 字符）
    - ``b:加热``   中文 2-gram（可选，默认开启）

    bigram 在**保留标点的原文**上按连续汉字段切分：标点被提前剥离会拼出
    「拭禁」这类跨短语假 bigram，把覆盖率算低。
    """
    terms: set[str] = set()
    norm = _normalize(text)
    if not norm:
        return terms

    for num in _NUM_RE.findall(norm):
        terms.add(f"n:{_num_token(num)}")
    for model in _MODEL_RE.findall(norm):
        terms.add(f"m:{model}")
    for word in _EN_RE.findall(norm):
        terms.add(f"w:{word}")
    if bigram:
        for run in _CJK_RE.findall(_raw(text)):
            for i in range(len(run) - 1):
                terms.add(f"b:{run[i : i + 2]}")
    return terms


def extract_strong_terms(text: str) -> set[str]:
    """只取强要素（数字 / 型号）：用于答案正确性，避免虚词稀释。"""
    norm = _normalize(text)
    if not norm:
        return set()
    terms = {f"n:{_num_token(num)}" for num in _NUM_RE.findall(norm)}
    terms |= {f"m:{model}" for model in _MODEL_RE.findall(norm)}
    return terms


def _chunk_text(item) -> str:
    """兼容 retrieved 的多种写法：dict(text/content/chunk_text) 或裸字符串。"""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "chunk_text", "page_content"):
            value = item.get(key)
            if isinstance(value, str):
                return value
        return ""
    return str(item)


def build_corpus_terms(chunks: Sequence) -> set[str]:
    """把多个检索块拼成 term 集合（块间用换行分隔，避免跨块拼出假 bigram）。"""
    corpus = "\n".join(_chunk_text(c) for c in chunks)
    return extract_key_terms(corpus)


def chunk_ids(retrieved: Sequence) -> List[str]:
    """按顺序取回检索块 id（无 id 的块跳过）。"""
    ids: List[str] = []
    for item in retrieved:
        if isinstance(item, dict) and item.get("chunk_id"):
            ids.append(str(item["chunk_id"]))
    return ids


# ---------------------------------------------------------------- 忠实度

def split_claims(answer: str, min_len: int = 6) -> List[str]:
    """把答案切成事实性 claim：按句末标点切分，丢弃过短句与纯承接套话。"""
    if not answer:
        return []
    claims: List[str] = []
    for raw in _SENT_SPLIT_RE.split(answer):
        claim = raw.strip(" \t，,、:：()（）\"'“”‘’")
        # 剥掉句首承接套话（"综上所述，""因此，"），保留其后的事实句
        for filler in _FILLER_PREFIXES:
            if claim.startswith(filler):
                claim = claim[len(filler) :].strip(" \t，,、:：")
                break
        if len(claim) < min_len:
            continue
        if _normalize(claim) in _FILLER_NORMALIZED:
            continue
        # 只保留含实际信息（汉字/字母/数字）的句子
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", claim):
            continue
        claims.append(claim)
    return claims


def claim_support(
    claim: str,
    chunks: Sequence,
    min_cover: float = 0.6,
    corpus_terms: set[str] | None = None,
) -> dict:
    """判定单个 claim 是否被检索块支撑。

    :param min_cover: 关键要素覆盖率阈值，>= 该值视为有支撑
    :param corpus_terms: 预先算好的块 term 集合（批量评测时复用，避免重复抽取）
    :return: {"supported", "cover", "n_terms", "missing"}
    """
    terms = extract_key_terms(claim)
    if not terms:
        # 无任何可校验要素（例如"建议联系厂家"）：无法证伪，不计入幻觉
        return {"supported": True, "cover": 1.0, "n_terms": 0, "missing": []}

    corpus = corpus_terms if corpus_terms is not None else build_corpus_terms(chunks)
    missing = sorted(terms - corpus)
    cover = 1.0 - len(missing) / len(terms)
    return {
        "supported": cover >= min_cover,
        "cover": round(cover, 4),
        "n_terms": len(terms),
        "missing": missing[:8],
    }


def faithfulness(answer: str, retrieved: Sequence, min_cover: float = 0.6) -> dict:
    """答案忠实度：有支撑 claim 数 / 总 claim 数；无 claim 时返回 0 并标注空。"""
    claims = split_claims(answer)
    if not claims:
        return {
            "n_claims": 0,
            "supported": 0,
            "faithfulness": 0.0,
            "hallucination_rate": 0.0,
            "unsupported_claims": [],
        }

    corpus_terms = build_corpus_terms(retrieved)
    supported = 0
    unsupported: List[dict] = []
    for claim in claims:
        detail = claim_support(claim, retrieved, min_cover=min_cover, corpus_terms=corpus_terms)
        if detail["supported"]:
            supported += 1
        else:
            unsupported.append({"claim": claim, "cover": detail["cover"], "missing": detail["missing"]})

    rate = supported / len(claims)
    return {
        "n_claims": len(claims),
        "supported": supported,
        "faithfulness": round(rate, 4),
        "hallucination_rate": round(1.0 - rate, 4),
        "unsupported_claims": unsupported[:5],
    }


# ---------------------------------------------------------------- 相关性

def _is_stop_bigram(bigram: str) -> bool:
    """任一字符是虚词即视为虚词 bigram（"机的""的最""多少"这类不该计入相关性分母）。"""
    return any(ch in _STOP_CHARS for ch in bigram)


def _query_bigrams(query: str) -> set[str]:
    norm = _normalize(query)
    grams: set[str] = set()
    for run in _CJK_RE.findall(norm):
        for i in range(len(run) - 1):
            gram = run[i : i + 2]
            if not _is_stop_bigram(gram):
                grams.add(gram)
    grams |= {f"w:{w}" for w in _EN_RE.findall(norm)}
    return grams


def answer_relevance(query: str, answer: str) -> float:
    """答案相关性：query 的 2-gram（去虚词）在答案中的覆盖率，取值 [0, 1]。

    空 query 或 query 全是虚词时返回 0.0（无信号，不臆造高分）。
    """
    grams = _query_bigrams(query)
    if not grams:
        return 0.0
    norm_answer = _normalize(answer)
    if not norm_answer:
        return 0.0
    hit = 0
    for gram in grams:
        token = gram[2:] if gram.startswith("w:") else gram
        if token in norm_answer:
            hit += 1
    return round(hit / len(grams), 4)


# ---------------------------------------------------------------- 正确性

def _content_chars(text: str) -> set[str]:
    """实义字符集合（去掉虚词与标点）：标准答案为纯文字时的覆盖率基数。"""
    return {ch for ch in _normalize(text) if ch.isalnum() and ch not in _STOP_CHARS}


def _ngrams(text: str, n: int) -> set[str]:
    """按「汉字连续段 / 字母数字连续段」分别切 n-gram。

    不能对整条归一化文本直接滑窗：标点被剥掉后会拼出「拭禁」这类跨短语假 n-gram，
    把覆盖率算低（正确性因此误判为 partial）。
    """
    grams: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+|[0-9a-z]+", _raw(text)):
        for i in range(len(run) - n + 1):
            grams.add(run[i : i + n])
    return grams


def answer_correctness(answer: str, golden_answer: str = "", retrieved: Sequence | None = None) -> dict:
    """答案正确性判定。

    - 有标准答案且含数字/型号：强要素全命中 = exact(1.0)，覆盖 >= 0.5 = partial(0.5)，否则 wrong(0)
    - 有标准答案但无强要素：标准答案的实义字符覆盖率 >= 0.9 = exact，>= 0.5 = partial
      （不用 bigram：中文没有分词，bigram 会把「乙醇」+「擦拭」拼出假边界而误判 partial）
    - 无标准答案：退化为「答案 vs 检索块」的 3-gram Jaccard（只衡量贴文档程度，标签 no_golden）
    """
    if golden_answer:
        golden_strong = extract_strong_terms(golden_answer)
        # 有强要素时要求全命中；纯文字标准答案按实义字符覆盖 0.9（同义改写容忍）
        exact_threshold = 0.999 if golden_strong else 0.9
        if golden_strong:
            answer_terms = extract_strong_terms(answer)
            hit = golden_strong & answer_terms
            cover = len(hit) / len(golden_strong)
            missing = sorted(golden_strong - answer_terms)
        else:
            golden_chars = _content_chars(golden_answer)
            answer_chars = _content_chars(answer)
            cover = len(golden_chars & answer_chars) / len(golden_chars) if golden_chars else 0.0
            missing = sorted(golden_chars - answer_chars)[:8]
        if cover >= exact_threshold:
            label, score = "exact", 1.0
        elif cover >= 0.5:
            label, score = "partial", 0.5
        else:
            label, score = "wrong", 0.0
        return {"label": label, "score": score, "cover": round(cover, 4), "missing": missing[:8]}

    if not retrieved:
        return {"label": "no_golden", "score": 0.0, "cover": 0.0, "missing": []}
    corpus = "\n".join(_chunk_text(c) for c in retrieved)
    answer_grams = _ngrams(answer, 3)
    corpus_grams = _ngrams(corpus, 3)
    union = answer_grams | corpus_grams
    jaccard = len(answer_grams & corpus_grams) / len(union) if union else 0.0
    return {"label": "no_golden", "score": round(jaccard, 4), "cover": round(jaccard, 4), "missing": []}


# ---------------------------------------------------------------- 引用

def extract_citations(answer: str) -> List[str]:
    """按出现顺序抽取答案中的引用标记 id（如 c_0001）。"""
    return _CITE_RE.findall(answer or "")


def citation_stats(answer: str, retrieved: Sequence) -> dict:
    """引用合法率：答案中 [c_xxx] 标记落在检索结果内的比例。"""
    cited = extract_citations(answer)
    retrieved_ids = set(chunk_ids(retrieved))
    valid = [cid for cid in cited if cid in retrieved_ids]
    invalid = [cid for cid in cited if cid not in retrieved_ids]
    return {
        "n_citations": len(cited),
        "n_valid": len(valid),
        "rate": round(len(valid) / len(cited), 4) if cited else 0.0,
        "uncited": len(cited) == 0,
        "invalid_ids": invalid[:8],
    }


# ---------------------------------------------------------------- 延迟

def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认 linear 口径一致）。"""
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError(f"分位点必须在 [0, 100] 内，收到 {p}")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p / 100
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def latency_percentiles(latencies: Iterable[float]) -> dict:
    values = [float(v) for v in latencies if v is not None]
    if not values:
        return {"n": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "avg": round(sum(values) / len(values), 2),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": round(max(values), 2),
    }


# ---------------------------------------------------------------- 聚合

def _worst_reason(faith: dict, corr: dict, cite: dict) -> str:
    """给待排查样本拼一行可读原因（报告直接展示，不用再翻 json）。"""
    reasons = []
    if faith["faithfulness"] < 1.0:
        reasons.append(f"无支撑句 {faith['n_claims'] - faith['supported']}/{faith['n_claims']}")
    if corr["label"] in {"wrong", "partial"}:
        reasons.append(f"correctness={corr['label']}")
    if cite["uncited"]:
        reasons.append("无引用")
    if cite["invalid_ids"]:
        reasons.append("引用越界 " + ",".join(cite["invalid_ids"]))
    return "; ".join(reasons) or "ok"


def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def evaluate_answers(records: Iterable[dict], min_cover: float = 0.6) -> dict:
    """对同一配置的问答记录聚合计算生成层指标。

    :return: 归一化后的指标字典（各指标均为样本均值；延迟为分位数）
    """
    rows = list(records)
    if not rows:
        return {
            "n": 0,
            "n_claims": 0,
            "faithfulness": 0.0,
            "hallucination_rate": 0.0,
            "relevance": 0.0,
            "correctness": 0.0,
            "exact_rate": 0.0,
            "partial_rate": 0.0,
            "wrong_rate": 0.0,
            "no_golden_rate": 0.0,
            "citation_rate": 0.0,
            "uncited_rate": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
        }

    faiths: List[float] = []
    relevances: List[float] = []
    correctness: List[float] = []
    citation_rates: List[float] = []
    labels: Dict[str, int] = {"exact": 0, "partial": 0, "wrong": 0, "no_golden": 0}
    total_claims = 0
    uncited = 0
    latencies: List[float] = []
    worst: List[dict] = []

    for rec in rows:
        answer = rec.get("answer") or ""
        retrieved = rec.get("retrieved") or []
        faith = faithfulness(answer, retrieved, min_cover=min_cover)
        faiths.append(faith["faithfulness"])
        total_claims += faith["n_claims"]

        relevances.append(answer_relevance(rec.get("query") or "", answer))
        corr = answer_correctness(answer, rec.get("golden_answer") or "", retrieved)
        correctness.append(corr["score"])
        labels[corr["label"]] = labels.get(corr["label"], 0) + 1

        cite = citation_stats(answer, retrieved)
        citation_rates.append(cite["rate"])
        if cite["uncited"]:
            uncited += 1

        if rec.get("latency_ms") is not None:
            latencies.append(float(rec["latency_ms"]))

        if faith["faithfulness"] < 1.0 or corr["label"] in {"wrong", "partial"}:
            worst.append(
                {
                    "qid": rec.get("qid", "?"),
                    "faithfulness": faith["faithfulness"],
                    "label": corr["label"],
                    "uncited": cite["uncited"],
                    "reason": _worst_reason(faith, corr, cite),
                }
            )

    n = len(rows)
    latency = latency_percentiles(latencies)
    return {
        "n": n,
        "n_claims": total_claims,
        "faithfulness": _mean(faiths),
        "hallucination_rate": round(1.0 - _mean(faiths), 4),
        "relevance": _mean(relevances),
        "correctness": _mean(correctness),
        "exact_rate": round(labels["exact"] / n, 4),
        "partial_rate": round(labels["partial"] / n, 4),
        "wrong_rate": round(labels["wrong"] / n, 4),
        "no_golden_rate": round(labels["no_golden"] / n, 4),
        "citation_rate": _mean(citation_rates),
        "uncited_rate": round(uncited / n, 4),
        "p50_latency_ms": latency["p50"],
        "p95_latency_ms": latency["p95"],
        "avg_latency_ms": latency["avg"],
        "_worst_cases": worst[:5],
    }


# ---------------------------------------------------------------- 报告

GEN_METRIC_KEYS = [
    ("faithfulness", "忠实度"),
    ("hallucination_rate", "幻觉率"),
    ("relevance", "相关性"),
    ("correctness", "正确性"),
    ("exact_rate", "exact 占比"),
    ("citation_rate", "引用合法率"),
    ("uncited_rate", "无引用占比"),
    ("traceability", "可溯源率"),
    ("p95_latency_ms", "p95 延迟(ms)"),
]


def render_gen_report(per_config: Dict[str, dict], judge: str = "rules", title: str = "生成层评测报告") -> str:
    """把 evaluate_answers 的多配置结果渲染为 markdown（含口径声明）。"""
    keys = [k for k, _ in GEN_METRIC_KEYS]
    headers = [name for _, name in GEN_METRIC_KEYS]
    lines = [
        f"# {title}",
        "",
        f"- judge=`{judge}`（rules 为离线规则口径，与 LLM-as-judge 存在系统性差异，两套数字不可直接对比）",
        "- 忠实度 = 有检索支撑的 claim 数 / 总 claim 数；幻觉率 = 1 - 忠实度",
        "- 正确性：exact=1.0 / partial=0.5 / wrong=0.0 的加权均值（无标准答案样本按 no_golden 单列）",
        "- 引用合法率 = 引用标记落在检索结果内的比例；无引用占比单列（引用缺失不被合法率惩罚）",
        "",
        "| 配置 | 样本 | 答案句 | " + " | ".join(headers) + " |",
        "|---" * (len(headers) + 3) + "|",
    ]
    for config, m in per_config.items():
        row = [config, str(m.get("n", 0)), str(m.get("n_claims", 0))]
        for key in keys:
            value = m.get(key, 0.0)
            row.append(f"{value:.4f}" if key not in {"p95_latency_ms"} else f"{value:.0f}")
        lines.append("| " + " | ".join(row) + " |")

    degraded = [c for c, m in per_config.items() if m.get("_degraded")]
    if degraded:
        lines += ["", f"> ⚠️ LLM judge 调用失败回退 rules 的配置：{', '.join(degraded)}（报告已标注 degraded）"]

    worst_lines: List[str] = []
    for config, m in per_config.items():
        cases = m.get("_worst_cases") or []
        if cases:
            worst_lines.append(
                f"- `{config}`："
                + "；".join(f"{c['qid']}（faith={c['faithfulness']:.2f}，{c['reason']}）" for c in cases)
            )
    if worst_lines:
        lines += ["", "## 待排查样本（忠实度或正确性未满分）", ""] + worst_lines

    lines += [
        "",
        "> 复现：`python evaluation/run_gen_eval.py --answers evaluation/data/answers.sample.jsonl`",
        "> 样例数据仅演示格式与解读方式，不得作为线上效果对外表述。",
    ]
    return "\n".join(lines)


def load_answers(path: str | Path) -> List[dict]:
    """读取 answers.jsonl；空行与坏行跳过（脏数据不应中断整体评测）。"""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[load_answers] 跳过第 {line_no} 行：JSON 解析失败")
    return records
