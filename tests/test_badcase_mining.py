"""badcase 归因单测：四类判定 / 优先级 / 批量统计 / 报告渲染"""

from evaluation.badcase_mining import diagnose, mine_badcases, render_badcases

DATASET = {
    "q1": {"qid": "q1", "relevant_chunk_ids": ["c_0001", "c_0002"]},
    "q2": {"qid": "q2", "relevant_chunk_ids": ["c_0005"]},
    "q3": {"qid": "q3", "relevant_chunk_ids": ["c_0009"]},
    "q4": {"qid": "q4", "relevant_chunk_ids": []},
}


def dump(qid, ranked, config="full"):
    return {"qid": qid, "config": config, "ranked_chunk_ids": ranked}


class TestDiagnose:
    def test_miss_when_no_relevant_in_topk(self):
        issues = diagnose(DATASET["q2"], dump("q2", ["c_0001", "c_0002", "c_0003"]))
        assert [i["type"] for i in issues] == ["miss"]
        assert "召回层" in issues[0]["layer"]
        assert "c_0005" in issues[0]["evidence"]

    def test_miss_when_relevant_beyond_k(self):
        # 相关块排第 6，k=5 时算 miss（不是 low_rank）
        issues = diagnose(DATASET["q2"], dump("q2", ["c_1", "c_2", "c_3", "c_4", "c_5", "c_0005"]), k=5)
        assert [i["type"] for i in issues] == ["miss"]

    def test_low_rank_when_first_relevant_after_threshold(self):
        issues = diagnose(DATASET["q2"], dump("q2", ["c_1", "c_2", "c_3", "c_0005"]), k=5)
        assert [i["type"] for i in issues] == ["low_rank"]
        assert "rank=4" in issues[0]["detail"]

    def test_no_issue_when_top_ranked(self):
        assert diagnose(DATASET["q2"], dump("q2", ["c_0005", "c_1"]), k=5) == []

    def test_untraceable_citation(self):
        issues = diagnose(
            DATASET["q2"],
            dump("q2", ["c_0005"]),
            answer="结论见 [c_9999]。",
        )
        types = [i["type"] for i in issues]
        assert "untraceable" in types
        assert "生成层" in issues[0]["layer"]

    def test_no_citation(self):
        issues = diagnose(DATASET["q2"], dump("q2", ["c_0005"]), answer="结论没有任何引用。")
        assert [i["type"] for i in issues] == ["no_citation"]

    def test_answer_with_valid_citation_is_clean(self):
        assert diagnose(DATASET["q2"], dump("q2", ["c_0005"]), answer="结论 [c_0005]。") == []

    def test_empty_relevant_ids_is_skipped(self):
        assert diagnose(DATASET["q4"], dump("q4", ["c_0001"])) == []

    def test_multiple_issues_can_coexist(self):
        # 未召回 + 无引用：两类问题都要报
        issues = diagnose(DATASET["q2"], dump("q2", ["c_1"]), answer="没有引用的答案。")
        assert {i["type"] for i in issues} == {"miss", "no_citation"}


class TestMineBadcases:
    def test_aggregates_and_sorts_by_severity(self):
        records = [
            dump("q1", ["c_0001"]),  # 正常
            dump("q2", ["c_1", "c_2", "c_3", "c_0005"]),  # low_rank
            dump("q3", ["c_1", "c_2"]),  # miss
        ]
        result = mine_badcases(records, DATASET, answers_by_qid={"q2": "有引用 [c_9999]。"})
        types = [item["type"] for item in result["items"]]
        assert types[0] == "miss"  # 严重度最高排最前
        assert "low_rank" in types
        assert "untraceable" in types
        assert result["n_total"] == 3
        assert result["by_type"]["miss"] == 1

    def test_unlabeled_qid_skipped(self):
        result = mine_badcases([dump("unknown", ["c_1"])], DATASET)
        assert result["items"] == []
        assert result["n_total"] == 0

    def test_counts_labeled(self):
        result = mine_badcases([dump("q1", ["c_0001"]), dump("q4", ["c_1"])], DATASET)
        assert result["n_total"] == 2
        assert result["n_labeled"] == 1


class TestRender:
    def test_render_table(self):
        result = mine_badcases([dump("q3", ["c_1"])], DATASET)
        report = render_badcases(result, k=5)
        assert "# badcase 归因清单" in report
        assert "| qid |" in report
        assert "未召回" in report

    def test_render_empty(self):
        report = render_badcases({"items": [], "n_total": 0, "n_labeled": 0, "by_type": {}})
        assert "没有命中问题样本" in report
