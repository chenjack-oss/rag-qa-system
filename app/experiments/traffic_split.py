"""确定性哈希分流（Traffic Splitting）

为什么用 sha256(session_id + experiment) 而不是随机数：
- 同一用户（session）在整个实验周期内必须稳定命中同一分组，否则同一次
  多轮会话中链路行为不一致，既污染指标又伤害体验；
- 哈希分流不需要额外存储分流关系，重启/扩容后依然成立。

分流算法：sha256 摘要的前 8 字节映射到 [0, 1)，再按 variants 的权重
区间切分落位。权重不必归一化（内部自动按总和切分）。
"""

import hashlib
from typing import Tuple


def hash_to_unit(uid: str, experiment: str) -> float:
    """把 (uid, experiment) 确定性映射到 [0, 1)。

    纯函数。experiment 参与哈希保证不同实验的分组互不相关（避免分组对齐偏差）。
    """
    digest = hashlib.sha256(f"{experiment}::{uid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2 ** 64)


def assign_variant(
    uid: str,
    experiment: str,
    variants: Tuple[str, ...] = ("control", "treatment"),
    ratios: Tuple[float, ...] = (0.5, 0.5),
) -> str:
    """把 uid 稳定分配到某个实验分组。

    :param uid: 分流主体（通常为 session_id）
    :param experiment: 实验名（参与哈希，隔离不同实验的分组）
    :param variants: 分组名，与 ratios 一一对应
    :param ratios: 各分组流量比例，自动按总和归一化
    :return: 命中的分组名
    :raises ValueError: variants/ratios 长度不一致、比例非正、或 uid/experiment 为空
    """
    if not uid or not experiment:
        raise ValueError("uid 与 experiment 不能为空")
    if len(variants) != len(ratios):
        raise ValueError("variants 与 ratios 长度必须一致")
    if any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios 必须为非负数且总和大于 0")

    pos = hash_to_unit(uid, experiment) * sum(ratios)
    cum = 0.0
    for name, ratio in zip(variants, ratios):
        cum += ratio
        if pos < cum:
            return name
    # 浮点边界兜底：pos 恰好落在总权重末端时归入最后一个非零分组
    return variants[-1]
