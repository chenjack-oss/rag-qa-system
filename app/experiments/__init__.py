"""A/B 实验框架：稳定分流 + 指标记录 + 分组分析

设计原则：
- 分流必须**确定性**：同一 session_id 在整个实验周期内始终命中同一分组，
  避免用户在一次会话中体验漂移（哈希分流，而非随机数）。
- 指标记录走旁路：写 MongoDB 失败只告警不上抛，观测不能拖垮问答主链路。
- 分析脚本支持离线运行：从 MongoDB 导出 jsonl 后即可分组聚合，不依赖线上库。

典型用法（在 query_service 侧）：

    from app.experiments.traffic_split import assign_variant
    from app.experiments.metrics_recorder import record_query_outcome

    variant = assign_variant(session_id, "rerank_vs_full")   # "control" / "treatment"
    ...  # 按 variant 决定链路行为（如 treatment 跳过精排）
    record_query_outcome(mongo_client, {
        "experiment": "rerank_vs_full",
        "variant": variant,
        "session_id": session_id,
        "query": user_query,
        "first_token_ms": ft_ms,
        "total_ms": total_ms,
        "hit": None,            # 若该 query 在标注集中可自动判 hit 则填 bool
        "user_feedback": None,  # 预留点赞/点踩回填
    })
"""

from app.experiments.traffic_split import assign_variant  # noqa: F401
