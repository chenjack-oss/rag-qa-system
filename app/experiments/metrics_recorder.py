"""实验/线上指标记录：MongoDB 旁路写入 + 导出与分析

指标文档结构（ab_metrics 集合）：
{
  "experiment": "rerank_latency_tradeoff",
  "variant":     "control" | "treatment",
  "session_id":  "...",
  "query":       "...",
  "first_token_ms": 8312,     # 首字延迟（对应简历口径「首字响应 <9s」）
  "total_ms":       12400,    # 端到端耗时
  "hit":            true,     # 若能自动判定（有标注）填 bool，否则 null
  "user_feedback":  null,     # 预留点赞/点踩回填
  "ts":             "..."
}

容错原则：所有写入失败只告警不上抛——指标是旁路，主链路可用性优先。
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.core.logger import logger

_AB_COLLECTION = "ab_metrics"


def build_outcome_doc(
    experiment: str,
    variant: str,
    session_id: str,
    query: str,
    first_token_ms: Optional[float] = None,
    total_ms: Optional[float] = None,
    hit: Optional[bool] = None,
    user_feedback: Optional[int] = None,
) -> Dict[str, Any]:
    """构造指标文档（纯函数，便于单测与离线构造）。"""
    return {
        "experiment": experiment,
        "variant": variant,
        "session_id": session_id,
        "query": query,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "hit": hit,
        "user_feedback": user_feedback,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def record_query_outcome(mongo_client: Any, doc: Dict[str, Any]) -> bool:
    """写入 MongoDB ab_metrics 集合；失败降级为日志告警。

    :param mongo_client: pymongo MongoClient（由调用方注入，避免本模块持有连接生命周期）
    """
    if mongo_client is None:
        return False
    try:
        db_name = os.getenv("MONGO_DB_NAME", "rag_qa")
        mongo_client[db_name][_AB_COLLECTION].insert_one(dict(doc))
        return True
    except Exception as e:  # pymongo 异常族 + 网络异常统一降级
        logger.warning(f"实验指标写入失败（忽略，不影响主链路）: {e}")
        return False


def export_to_jsonl(mongo_client: Any, experiment: str, out_path: str) -> int:
    """把某实验的线上指标导出为 jsonl，供 analyze.py 离线分析。返回导出行数。"""
    if mongo_client is None:
        return 0
    db_name = os.getenv("MONGO_DB_NAME", "rag_qa")
    cursor = mongo_client[db_name][_AB_COLLECTION].find({"experiment": experiment})
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in cursor:
            doc.pop("_id", None)
            f.write(__import__("json").dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    return n
