"""A/B 实验框架单元测试：分流稳定性 / 配比 / 边界，指标文档构造"""

import pytest

from app.experiments.analyze import aggregate
from app.experiments.metrics_recorder import build_outcome_doc
from app.experiments.traffic_split import assign_variant, hash_to_unit


class TestHashToUnit:
    def test_deterministic(self):
        assert hash_to_unit("s1", "exp1") == hash_to_unit("s1", "exp1")

    def test_in_range(self):
        for i in range(500):
            v = hash_to_unit(f"session-{i}", "exp")
            assert 0.0 <= v < 1.0

    def test_experiment_isolates_grouping(self):
        # 不同实验对同一 uid 的哈希应显著不同（分组互不相关）
        same = sum(
            1 for i in range(200) if hash_to_unit(f"s{i}", "exp_a") == hash_to_unit(f"s{i}", "exp_b")
        )
        assert same == 0


class TestAssignVariant:
    def test_stable_assignment(self):
        # 同一 uid 反复调用必须稳定
        first = assign_variant("user-42", "rerank_latency_tradeoff")
        for _ in range(10):
            assert assign_variant("user-42", "rerank_latency_tradeoff") == first

    def test_approximate_ratio(self):
        # 1000 个 uid、50/50 配比，偏差应 < 10%
        counts = {"control": 0, "treatment": 0}
        for i in range(1000):
            counts[assign_variant(f"u{i}", "exp50")] += 1
        assert abs(counts["control"] - counts["treatment"]) < 100

    def test_skewed_ratio(self):
        # 90/10 灰度配比：treatment 应显著少于 control
        counts = {"control": 0, "treatment": 0}
        for i in range(1000):
            counts[assign_variant(f"u{i}", "exp90", ratios=(0.9, 0.1))] += 1
        assert counts["treatment"] < 200

    def test_custom_variant_names(self):
        assert assign_variant("u1", "e", variants=("A", "B", "C"), ratios=(1, 1, 1)) in {"A", "B", "C"}

    def test_empty_uid_raises(self):
        with pytest.raises(ValueError):
            assign_variant("", "exp")

    def test_empty_experiment_raises(self):
        with pytest.raises(ValueError):
            assign_variant("u", "")

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            assign_variant("u", "exp", variants=("a", "b"), ratios=(1.0,))

    def test_bad_ratios_raise(self):
        with pytest.raises(ValueError):
            assign_variant("u", "exp", ratios=(0.0, 0.0))


class TestBuildOutcomeDoc:
    def test_fields(self):
        doc = build_outcome_doc("exp", "control", "s1", "q1", first_token_ms=100.0, total_ms=200.0, hit=True)
        assert doc["experiment"] == "exp"
        assert doc["variant"] == "control"
        assert doc["hit"] is True
        assert doc["user_feedback"] is None
        assert doc["ts"]  # 时间戳已生成

    def test_optional_fields_none(self):
        doc = build_outcome_doc("exp", "treatment", "s1", "q1")
        assert doc["first_token_ms"] is None
        assert doc["hit"] is None


class TestAggregate:
    def test_percentiles_and_hit_rate(self):
        records = [
            {"variant": "control", "first_token_ms": 100, "total_ms": 200, "hit": True},
            {"variant": "control", "first_token_ms": 200, "total_ms": 400, "hit": False},
            {"variant": "control", "first_token_ms": 300, "total_ms": 600, "hit": True},
            {"variant": "treatment", "first_token_ms": 50, "total_ms": 100, "hit": None},
        ]
        ab = aggregate(records)
        assert ab["control"]["n"] == 3
        assert ab["control"]["first_token_p50_ms"] == 200
        assert ab["control"]["hit_rate"] == pytest.approx(2 / 3)
        # treatment 只有 1 条且 hit 为 null -> hit_rate 为 None
        assert ab["treatment"]["hit_rate"] is None

    def test_empty_records(self):
        assert aggregate([]) == {}
