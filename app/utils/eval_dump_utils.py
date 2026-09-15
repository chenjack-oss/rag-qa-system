"""评测数据采集与消融开关（线上旁路，默认零影响）

设计原则：
- 全部能力由环境变量门控。未设置任何变量时，检索链路行为与原来完全一致；
  观测与消融属于"旁路"，绝不在主链路上引入默认开销。
- dump 文件为 jsonl 追加写（每行一条独立 JSON），失败只打日志告警，
  不允许因为评测采集拖垮线上问答（可观测性 < 业务可用性）。

环境变量：
- EVAL_DUMP_PATH      设置后，每次问答结束把检索 TopK 与耗时追加写入该文件
- EVAL_ABLATION_MODE  消融模式，仅评测采集期间使用：
                      no_rrf / no_rerank / no_hyde / no_sparse（默认空=完整链路）
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from app.core.logger import logger

_VALID_ABLATION_MODES = {"no_rrf", "no_rerank", "no_hyde", "no_sparse"}


def get_ablation_mode() -> str:
    """读取当前消融模式；非法值一律按完整链路处理并告警。"""
    mode = (os.getenv("EVAL_ABLATION_MODE") or "").strip().lower()
    if mode and mode not in _VALID_ABLATION_MODES:
        logger.warning(f"EVAL_ABLATION_MODE 非法值: {mode}，忽略（合法值: {sorted(_VALID_ABLATION_MODES)}）")
        return ""
    return mode


def should_skip_hyde() -> bool:
    """消融 no_hyde：跳过 HyDE 检索路。"""
    return get_ablation_mode() == "no_hyde"


def should_use_single_source() -> bool:
    """消融 no_rrf：RRF 只使用稠密向量主路（等价于关闭多路融合）。"""
    return get_ablation_mode() == "no_rrf"


def should_skip_rerank() -> bool:
    """消融 no_rerank：跳过 FlagReranker 精排，保留 RRF 融合顺序。"""
    return get_ablation_mode() == "no_rerank"


def should_zero_sparse_weight() -> bool:
    """消融 no_sparse：混合检索稀疏向量权重置 0（稠密 1.0 / 稀疏 0.0）。"""
    return get_ablation_mode() == "no_sparse"


def build_dump_record(
    session_id: str,
    query: str,
    ranked_docs: List[dict],
    latency_ms: Optional[float] = None,
) -> Optional[dict]:
    """构造一条评测 dump 记录（纯函数，便于单测）。

    :param ranked_docs: rerank 节点输出的 topk 文档列表（含 chunk_id/id、score、source）
    :return: dict；ranked_docs 为空时返回 None（空结果不产生评测噪音）
    """
    ranked_ids = [
        str(d.get("chunk_id") or d.get("id"))
        for d in (ranked_docs or [])
        if isinstance(d, dict) and (d.get("chunk_id") or d.get("id"))
    ]
    if not ranked_ids:
        return None
    return {
        "qid": session_id,
        "query": query,
        "config": get_ablation_mode() or "full",
        "ranked_chunk_ids": ranked_ids,
        "latency_ms": latency_ms,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_dump_record(record: dict) -> bool:
    """把记录追加写入 EVAL_DUMP_PATH 指向的 jsonl 文件。

    任何异常都只告警不上抛——评测采集失败不能影响问答主链路。
    """
    dump_path = os.getenv("EVAL_DUMP_PATH")
    if not dump_path:
        return False
    try:
        # 打开-关闭式追加：单行短写入，配合行级 JSON 天然容错
        with open(dump_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError as e:
        logger.warning(f"评测 dump 写入失败（忽略，不影响主链路）: {e}")
        return False
