"""实验注册表：所有在线实验的变量定义集中在这里，禁止散落在业务节点内

每个实验的三要素：
- variants: 分组名
- ratios:    流量配比（可灰度，如 0.9/0.1 起步）
- apply:     该分组对链路行为的具体改动（以环境变量注入方式实现，
             复用 evaluation 消融开关，保证"消融/实验"同一套旁路机制）

约定：实验下线后把条目整体删除，并在 metrics_recorder 的 Mongo 集合中
保留历史数据（分析脚本按 experiment 字段过滤，不受影响）。
"""

from typing import Dict, Tuple

# 实验 -> 分组/配比/行为注入。apply 为无参函数，在请求入口调用一次。
EXPERIMENTS: Dict[str, dict] = {
    # 示例实验：验证"跳过精排"对首字延迟与答案质量的实际权衡
    # （treatment 组跳过 FlagReranker，期望 first_token 显著下降、hit@k 轻微波动）
    "rerank_latency_tradeoff": {
        "description": "跳过精排换首字延迟：量化 FlagReranker 的延迟-质量权衡",
        "variants": ("control", "treatment"),
        "ratios": (0.9, 0.1),  # 小流量起步
        "apply": {
            "control": lambda: None,
            "treatment": lambda: None,  # 链路行为由 EVAL_ABLATION_MODE 承载，见 apply_env
        },
        # treatment 分组对应的链路旁路开关（与消融共用实现）
        "apply_env": {"treatment": {"EVAL_ABLATION_MODE": "no_rerank"}},
    },
}

# 该进程是否作为实验流量参与分流（运维开关：全量放开/关闭）
EXPERIMENT_GLOBAL_ENABLED = False


def get_experiment(name: str) -> dict:
    if name not in EXPERIMENTS:
        raise KeyError(f"未注册的实验: {name}，已注册: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]


def make_variants_and_ratios(exp: dict) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    return tuple(exp["variants"]), tuple(exp["ratios"])


def variant_apply_env(exp: dict, variant: str) -> dict:
    """返回该分组需要注入的环境变量（如 EVAL_ABLATION_MODE）。"""
    return dict(exp.get("apply_env", {}).get(variant, {}))
