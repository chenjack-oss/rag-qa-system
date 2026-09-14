"""LLM 三元组输出解析与容错（Graph RAG 图谱构建的前置纯函数）

职责：把 LLM 返回的"实体关系三元组 JSON"清洗成结构化列表。
LLM 实际输出往往不严格是纯 JSON，常见 badcase：
1. 用 markdown 代码块包裹（```json ... ```）
2. JSON 前后夹带说明文字（"以下是抽取结果："）
3. 输出被 max_tokens 截断，最后一个对象不完整
4. 返回单个对象而非数组，或 {"triples": [...]} 包装
5. 字段缺失/为空/类型漂移（数字、null）
"""

import json
from typing import Any, Dict, List

# 单次解析最多保留的三元组数量（与 prompt 中"最多抽取10条"的约束对应）
MAX_TRIPLES = 10

# 必填字段：head / relation / tail
REQUIRED_FIELDS = ("head", "relation", "tail")
# 可选字段及缺省值：head_type / tail_type
DEFAULT_TYPE = "实体"


def _strip_markdown_fences(text: str) -> str:
    """剥掉 markdown 代码块围栏（```json ... ``` 或 ``` ... ```）"""
    stripped = text.strip()
    if stripped.startswith("```"):
        # 去掉开头围栏（可能带语言标记 json/JSON）
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        # 去掉结尾围栏
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _extract_json_candidates(text: str) -> List[Any]:
    """从混杂文本中提取所有可解析的 JSON 值。

    策略：用 raw_decode 从每个 '[' / '{' 位置开始尝试解析，
    能最大程度容忍截断——截断处之后的残缺尾会被自然丢弃，
    而截断前已完整闭合的对象仍能被解析出来。
    """
    candidates: List[Any] = []
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        # 只要数组或对象，跳过裸数字等
        if isinstance(value, (list, dict)):
            candidates.append(value)
    return candidates


def _flatten_to_triple_list(value: Any) -> List[Dict[str, Any]]:
    """把各种 JSON 形态规整成三元组 dict 列表。

    兼容：[t1, t2] / {"triples": [...]} / {"data": [...]} / 单个对象 t1
    """
    if isinstance(value, dict):
        # 常见包装格式：{"triples": [...]} / {"data": [...] / {"result": [...]}
        for key in ("triples", "data", "result", "entities"):
            inner = value.get(key)
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
        # 无包装键的单对象，本身当作一条三元组
        return [value]
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def _clean_field(item: Dict[str, Any], key: str) -> str:
    """读取并清洗字段：强转字符串、去空白；空值返回空串。"""
    val = item.get(key)
    if val is None:
        return ""
    return str(val).strip()


def _normalize_triple(item: Dict[str, Any]) -> Dict[str, str] | None:
    """单条三元组清洗与校验，非法返回 None。"""
    head = _clean_field(item, "head")
    relation = _clean_field(item, "relation")
    tail = _clean_field(item, "tail")

    # 必填字段任一为空则整条丢弃（LLM 常见坏输出）
    if not head or not relation or not tail:
        return None

    head_type = _clean_field(item, "head_type") or DEFAULT_TYPE
    tail_type = _clean_field(item, "tail_type") or DEFAULT_TYPE

    return {
        "head": head,
        "head_type": head_type,
        "relation": relation,
        "tail": tail,
        "tail_type": tail_type,
    }


def parse_triples(raw_output: str, max_triples: int = MAX_TRIPLES) -> List[Dict[str, str]]:
    """解析 LLM 输出为标准化三元组列表（纯函数，无外部依赖）。

    :param raw_output: LLM 返回的原始文本（可能含 markdown 围栏/说明文字/截断）
    :param max_triples: 最多保留条数
    :return: [{"head","head_type","relation","tail","tail_type"}, ...]
             已按 (head, relation, tail) 去重、保序截断；解析失败返回 []
    """
    if not raw_output or not raw_output.strip():
        return []

    text = _strip_markdown_fences(raw_output)

    # 优先整体解析（最快的正常路径）
    results: List[Dict[str, str]] = []
    try:
        candidates = [json.loads(text)]
    except json.JSONDecodeError:
        candidates = _extract_json_candidates(text)

    # 合并所有候选中的三元组（正常情况只有一个候选）
    seen = set()
    for candidate in candidates:
        for item in _flatten_to_triple_list(candidate):
            triple = _normalize_triple(item)
            if triple is None:
                continue
            dedup_key = (triple["head"], triple["relation"], triple["tail"])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            results.append(triple)
            # 提前截断，避免长输出下继续无谓解析
            if len(results) >= max_triples:
                return results

    return results
