"""评测对比与回归门禁单测：指标对齐 / Δ 符号 / 门禁判定 / 报告渲染 / 载入容错"""

import json

import pytest

from evaluation.compare_eval import (
    check_regressions,
    compare_configs,
    load_metrics_file,
    render_comparison,
)

BASE = {"full": {"n": 10, "hit@5": 0.90, "mrr": 0.80, "avg_latency_ms": 1000.0, "_worst_cases": []}}
CAND = {"full": {"n": 10, "hit@5": 0.86, "mrr": 0.82, "avg_latency_ms": 1200.0, "_worst_cases": []}}


def row_of(rows, metric):
    return next(r for r in rows if r["metric"] == metric)


class TestCompareConfigs:
    def test_delta_sign(self):
        rows = compare_configs(BASE, CAND)
        assert row_of(rows, "hit@5")["delta"] == pytest.approx(-0.04)
        assert row_of(rows, "mrr")["delta"] == pytest.approx(0.02)

    def test_private_keys_excluded(self):
        rows = compare_configs(BASE, CAND)
        assert all(not r["metric"].startswith("_") for r in rows)

    def test_only_common_configs(self):
        rows = compare_configs({"full": BASE["full"], "no_rerank": BASE["full"]}, CAND)
        assert {r["config"] for r in rows} == {"full"}

    def test_only_common_metrics(self):
        rows = compare_configs({"full": {"hit@5": 0.9}}, {"full": {"mrr": 0.8}})
        assert rows == []

    def test_latency_flagged_as_lower_is_better(self):
        rows = compare_configs(BASE, CAND)
        assert row_of(rows, "avg_latency_ms")["lower_is_better"] is True
        assert row_of(rows, "hit@5")["lower_is_better"] is False


class TestCheckRegressions:
    def test_drop_over_threshold(self):
        rows = compare_configs(BASE, CAND)
        hits = check_regressions(rows, 0.02)
        assert [h["metric"] for h in hits] == ["hit@5"]

    def test_within_threshold(self):
        assert check_regressions(compare_configs(BASE, CAND), 0.05) == []

    def test_gate_disabled_by_zero(self):
        assert check_regressions(compare_configs(BASE, CAND), 0.0) == []

    def test_latency_never_gated(self):
        slow = {"full": {"n": 1, "p95_latency_ms": 100.0}}
        fast = {"full": {"n": 1, "p95_latency_ms": 900.0}}
        assert check_regressions(compare_configs(slow, fast), 0.01) == []

    def test_improvement_never_fails(self):
        rows = compare_configs(CAND, BASE)
        assert check_regressions(rows, 0.02) == []


class TestRender:
    def test_pass_verdict(self):
        rows = compare_configs(BASE, CAND)
        report = render_comparison(rows, "base.json", "new.json", 0.05, [])
        assert "**PASS**" in report
        assert "基线：`base.json`" in report

    def test_fail_verdict_and_items(self):
        rows = compare_configs(BASE, CAND)
        hits = check_regressions(rows, 0.02)
        report = render_comparison(rows, "base.json", "new.json", 0.02, hits)
        assert "**FAIL**" in report
        assert "门禁失败项" in report
        assert "hit@5" in report

    def test_no_gate_line_when_threshold_zero(self):
        rows = compare_configs(BASE, CAND)
        assert "门禁阈值" not in render_comparison(rows, "a", "b", 0.0, [])


class TestLoadMetricsFile:
    def test_load_per_config_json(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"k": 5, "per_config": BASE}), encoding="utf-8")
        assert load_metrics_file(path)["full"]["hit@5"] == 0.9

    def test_load_flat_json(self, tmp_path):
        path = tmp_path / "flat.json"
        path.write_text(json.dumps(BASE), encoding="utf-8")
        assert "full" in load_metrics_file(path)

    def test_dump_requires_dataset(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_metrics_file(path)

    def test_unknown_shape_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"k": 5}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_metrics_file(path)

    def test_dump_recomputed_with_dataset(self, tmp_path):
        dataset = tmp_path / "ds.jsonl"
        dataset.write_text(
            json.dumps({"qid": "q1", "relevant_chunk_ids": ["c1"]}) + "\n", encoding="utf-8"
        )
        dump_path = tmp_path / "d.jsonl"
        dump_path.write_text(
            json.dumps({"qid": "q1", "config": "full", "ranked_chunk_ids": ["c1", "c2"]}) + "\n",
            encoding="utf-8",
        )
        metrics = load_metrics_file(dump_path, dataset=str(dataset), k=5)
        assert metrics["full"]["hit@5"] == 1.0
