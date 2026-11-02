"""生成层指标单测：term 抽取 / claim 切分 / 忠实度 / 相关性 / 正确性 / 引用 / 延迟 / 聚合

这些用例全部跑纯函数，不依赖 Milvus / torch / LLM，与 CI 的依赖白名单一致。
"""

import pytest

from evaluation.gen_metrics import (
    answer_correctness,
    answer_relevance,
    chunk_ids,
    citation_stats,
    evaluate_answers,
    extract_citations,
    extract_key_terms,
    extract_strong_terms,
    faithfulness,
    latency_percentiles,
    load_answers,
    percentile,
    render_gen_report,
    split_claims,
)

CHUNK = {
    "chunk_id": "c_0001",
    "text": "加热温度：HAK 180 烫金机加热温度范围为 80~200℃，最高不得超过 200℃，出厂默认 150℃。",
}


class TestExtractKeyTerms:
    def test_number(self):
        assert "n:200" in extract_key_terms("最高温度 200℃")

    def test_decimal_is_preserved(self):
        # "0.4" 不能被归一化成 "04"（保留小数点）
        terms = extract_key_terms("压力 0.4MPa")
        assert "n:0.4" in terms
        assert "n:04" not in terms

    def test_trailing_zero_normalized(self):
        assert extract_key_terms("温度 200.0 度") == extract_key_terms("温度 200 度")

    def test_model_code(self):
        assert "m:hak180" in extract_key_terms("HAK 180 烫金机")

    def test_english_word_boundary(self):
        # "hak180" 不应被拆出英文词 "hak"
        terms = extract_key_terms("hak180")
        assert "m:hak180" in terms
        assert "w:hak" not in terms

    def test_bigram_does_not_cross_punctuation(self):
        # 标点两侧不能拼出「拭禁」这类假 bigram
        terms = extract_key_terms("无水乙醇擦拭，禁用丙酮")
        assert "b:无水" in terms
        assert "b:拭禁" not in terms

    def test_bigram_can_be_disabled(self):
        assert extract_key_terms("加热温度", bigram=False) == set()

    def test_empty_text(self):
        assert extract_key_terms("") == set()

    def test_strong_terms_only_numbers_and_models(self):
        strong = extract_strong_terms("HAK 180 最高 200℃，禁止用丙酮")
        assert "n:200" in strong
        assert "m:hak180" in strong
        assert not any(term.startswith("b:") for term in strong)


class TestSplitClaims:
    def test_split_by_sentence_punctuation(self):
        claims = split_claims("最高温度 200 摄氏度。出厂默认 150 摄氏度。")
        assert len(claims) == 2

    def test_drop_short_sentence(self):
        assert split_claims("好。最高温度是两百度。") == ["最高温度是两百度"]

    def test_drop_filler_claim(self):
        claims = split_claims("综上所述，设备需要定期保养。")
        # 「综上所述」被丢弃，只剩事实句
        assert claims == ["设备需要定期保养"]

    def test_strip_leading_punctuation(self):
        claims = split_claims("；最高温度是两百度。")
        assert claims == ["最高温度是两百度"]

    def test_empty_answer(self):
        assert split_claims("") == []


class TestClaimSupport:
    def test_supported_when_numbers_match(self):
        result = claim_support_ok("最高不得超过 200℃", [CHUNK])
        assert result["supported"] is True
        assert result["cover"] >= 0.6

    def test_unsupported_when_fabricated(self):
        result = claim_support_ok("主板烧毁需要返厂更换", [CHUNK])
        assert result["supported"] is False
        assert result["cover"] < 0.6

    def test_no_terms_is_treated_as_unverifiable(self):
        # 抽不出任何要素（空 claim）时无法证伪，按有支撑处理，不计入幻觉
        result = claim_support_ok("", [CHUNK])
        assert result["supported"] is True
        assert result["n_terms"] == 0

    def test_corpus_terms_can_be_reused(self):
        from evaluation.gen_metrics import build_corpus_terms

        corpus = build_corpus_terms([CHUNK])
        a = claim_support_ok("最高不得超过 200℃", [CHUNK], corpus_terms=corpus)
        b = claim_support_ok("最高不得超过 200℃", [CHUNK])
        assert a == b


def claim_support_ok(claim, chunks, **kwargs):
    from evaluation.gen_metrics import claim_support

    return claim_support(claim, chunks, **kwargs)


class TestFaithfulness:
    def test_all_claims_supported(self):
        result = faithfulness("最高不得超过 200℃ [c_0001]。出厂默认 150℃ [c_0001]。", [CHUNK])
        assert result["n_claims"] == 2
        assert result["faithfulness"] == 1.0
        assert result["hallucination_rate"] == 0.0

    def test_fabricated_claim_detected(self):
        result = faithfulness("机器主板烧毁需要整体更换。", [CHUNK])
        assert result["faithfulness"] == 0.0
        assert result["unsupported_claims"]

    def test_mixed_answer(self):
        result = faithfulness("最高不得超过 200℃。主板烧毁需要返厂。", [CHUNK])
        assert result["supported"] == 1
        assert 0 < result["faithfulness"] < 1

    def test_empty_answer(self):
        result = faithfulness("", [CHUNK])
        assert result["n_claims"] == 0
        assert result["faithfulness"] == 0.0

    def test_accepts_plain_string_chunks(self):
        result = faithfulness("最高不得超过 200℃。", [CHUNK["text"]])
        assert result["faithfulness"] == 1.0


class TestAnswerRelevance:
    def test_on_topic_answer_scores_high(self):
        score = answer_relevance("烫金机的最大加热温度是多少", "最大加热温度为 200℃")
        assert score >= 0.5

    def test_off_topic_answer_scores_low(self):
        score = answer_relevance("怎么清洁胶辊", "额定电压 220V，功率 1.8kW")
        assert score <= 0.2

    def test_empty_query_is_zero(self):
        assert answer_relevance("", "任意答案") == 0.0

    def test_stopword_only_query_is_zero(self):
        assert answer_relevance("多少是多少", "任意答案") == 0.0

    def test_empty_answer_is_zero(self):
        assert answer_relevance("怎么清洁胶辊", "") == 0.0


class TestAnswerCorrectness:
    def test_exact_when_all_numbers_hit(self):
        result = answer_correctness("最高温度 200℃", "200℃", [CHUNK])
        assert result["label"] == "exact"
        assert result["score"] == 1.0

    def test_partial_when_half_numbers_hit(self):
        result = answer_correctness("电压 220V", "220V / 1.8kW", [CHUNK])
        assert result["label"] == "partial"
        assert result["score"] == 0.5

    def test_wrong_when_number_mismatch(self):
        result = answer_correctness("压力 0.8MPa", "0.4~0.6MPa", [CHUNK])
        assert result["label"] == "wrong"
        assert result["score"] == 0.0

    def test_text_golden_uses_content_chars(self):
        # 没有数字/型号时按实义字符覆盖，同义改写不算错
        result = answer_correctness("用无水乙醇擦拭胶辊，禁止使用丙酮", "无水乙醇擦拭，禁用丙酮", [CHUNK])
        assert result["label"] == "exact"

    def test_text_golden_partial(self):
        result = answer_correctness("断电挂牌", "断电挂牌并可靠接地", [CHUNK])
        assert result["label"] == "partial"

    def test_no_golden_uses_jaccard(self):
        answer = CHUNK["text"]
        result = answer_correctness(answer, "", [CHUNK])
        assert result["label"] == "no_golden"
        assert result["score"] > 0.9

    def test_no_golden_no_retrieved(self):
        result = answer_correctness("任意答案", "", [])
        assert result == {"label": "no_golden", "score": 0.0, "cover": 0.0, "missing": []}


class TestCitation:
    def test_extract_in_order(self):
        assert extract_citations("结论见 [c_0002] 与 [c_0001]。") == ["c_0002", "c_0001"]

    def test_all_valid(self):
        stats = citation_stats("答案 [c_0001]", [CHUNK])
        assert stats["rate"] == 1.0
        assert stats["uncited"] is False

    def test_out_of_range_detected(self):
        stats = citation_stats("答案 [c_0001][c_9999]", [CHUNK])
        assert stats["n_citations"] == 2
        assert stats["n_valid"] == 1
        assert stats["rate"] == 0.5
        assert stats["invalid_ids"] == ["c_9999"]

    def test_uncited_answer(self):
        stats = citation_stats("答案没有引用标记", [CHUNK])
        assert stats["uncited"] is True
        assert stats["rate"] == 0.0


class TestLatency:
    def test_percentile_single_value(self):
        assert percentile([100], 95) == 100

    def test_percentile_linear_interpolation(self):
        # 与 numpy.percentile 的 linear 口径一致
        assert percentile([10, 20, 30, 40], 50) == 25.0
        assert percentile([10, 20, 30, 40], 95) == 38.5

    def test_percentile_empty(self):
        assert percentile([], 50) == 0.0

    def test_percentile_out_of_range(self):
        with pytest.raises(ValueError):
            percentile([1, 2], 101)

    def test_latency_percentiles(self):
        result = latency_percentiles([100, 200, 300])
        assert result["n"] == 3
        assert result["p50"] == 200
        assert result["max"] == 300

    def test_latency_empty(self):
        assert latency_percentiles([])["p95"] == 0.0


class TestEvaluateAnswers:
    def test_empty_records(self):
        result = evaluate_answers([])
        assert result["n"] == 0
        assert result["faithfulness"] == 0.0

    def test_aggregates_multiple_records(self):
        records = [
            {"qid": "q1", "query": "最大加热温度", "answer": "最高不得超过 200℃ [c_0001]。",
             "retrieved": [CHUNK], "golden_answer": "200℃", "latency_ms": 100},
            {"qid": "q2", "query": "最大加热温度", "answer": "主板烧毁。",
             "retrieved": [CHUNK], "golden_answer": "200℃", "latency_ms": 300},
        ]
        result = evaluate_answers(records)
        assert result["n"] == 2
        assert result["faithfulness"] == 0.5
        assert result["exact_rate"] == 0.5
        assert result["wrong_rate"] == 0.5
        assert result["p50_latency_ms"] == 200

    def test_uncited_rate(self):
        records = [
            {"qid": "q1", "answer": "结论 [c_0001]。", "retrieved": [CHUNK], "query": "温度"},
            {"qid": "q2", "answer": "结论。", "retrieved": [CHUNK], "query": "温度"},
        ]
        assert evaluate_answers(records)["uncited_rate"] == 0.5

    def test_worst_cases_collected(self):
        records = [{"qid": "q9", "answer": "主板烧毁。", "retrieved": [CHUNK], "query": "温度"}]
        assert evaluate_answers(records)["_worst_cases"][0]["qid"] == "q9"


class TestReportAndLoading:
    def test_report_contains_judge_and_table(self):
        report = render_gen_report({"baseline": evaluate_answers([])}, judge="rules")
        assert "judge=`rules`" in report
        assert "| 配置 |" in report
        assert "faithfulness" not in report  # 表头用中文

    def test_report_marks_degraded(self):
        report = render_gen_report({"baseline": {"n": 1, "_degraded": 2}}, judge="llm")
        assert "degraded" in report

    def test_load_answers_skips_bad_lines(self, tmp_path):
        path = tmp_path / "a.jsonl"
        path.write_text('{"qid": "q1"}\n\nnot-json\n{"qid": "q2"}\n', encoding="utf-8")
        records = load_answers(path)
        assert [r["qid"] for r in records] == ["q1", "q2"]

    def test_chunk_ids(self):
        assert chunk_ids([CHUNK, {"text": "无 id"}]) == ["c_0001"]
