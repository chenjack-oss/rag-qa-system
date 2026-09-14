"""reciprocal_rank_fusion 单元测试：带权重 RRF 多路召回融合

RRF 公式: score(d) = Σ_source weight * 1 / (k + rank_source(d))
"""

from app.query_process.agent.nodes.node_rrf import _as_entity_list, reciprocal_rank_fusion


def _doc(chunk_id, content="c"):
    return {"chunk_id": chunk_id, "content": content}


class TestRRFBasic:
    def test_single_source_preserves_rank_order(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        merged = reciprocal_rank_fusion([(docs, 1.0)], k=60)
        assert [d[0]["chunk_id"] for d in merged] == ["a", "b", "c"]

    def test_score_matches_formula(self):
        docs = [_doc("a")]
        merged = reciprocal_rank_fusion([(docs, 1.0)], k=60)
        assert abs(merged[0][1] - 1.0 / 61) < 1e-12

    def test_doc_in_both_sources_outranks_single(self):
        """两路都命中的文档，得分累加，应排在只单路命中的前面"""
        vec_hits = [_doc("a"), _doc("only_vec")]
        hyde_hits = [_doc("a"), _doc("only_hyde")]
        merged = reciprocal_rank_fusion([(vec_hits, 1.0), (hyde_hits, 1.0)], k=60)
        assert merged[0][0]["chunk_id"] == "a"

    def test_weight_boosts_source(self):
        """高权重来源的文档排名靠前"""
        src_a = [_doc("from_a")]
        src_b = [_doc("from_b")]
        merged = reciprocal_rank_fusion([(src_a, 0.5), (src_b, 2.0)], k=60)
        assert merged[0][0]["chunk_id"] == "from_b"

    def test_max_results_truncates(self):
        docs = [_doc(str(i)) for i in range(10)]
        merged = reciprocal_rank_fusion([(docs, 1.0)], max_results=3)
        assert len(merged) == 3

    def test_empty_sources(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([([], 1.0)]) == []

    def test_item_missing_id_skipped(self):
        merged = reciprocal_rank_fusion([([{"content": "no id"}, _doc("a")], 1.0)])
        assert len(merged) == 1
        assert merged[0][0]["chunk_id"] == "a"

    def test_larger_k_smooths_ranking(self):
        """k 越大，排名差异被平滑（第 1 名与第 2 名得分差距缩小）"""
        docs = [_doc("a"), _doc("b")]
        gap_k1 = reciprocal_rank_fusion([(docs, 1.0)], k=1)
        gap_k60 = reciprocal_rank_fusion([(docs, 1.0)], k=60)
        diff_k1 = gap_k1[0][1] - gap_k1[1][1]
        diff_k60 = gap_k60[0][1] - gap_k60[1][1]
        assert diff_k1 > diff_k60


class TestAsEntityList:
    def test_flat_dict_passthrough(self):
        out = _as_entity_list([_doc("a", "hello")])
        assert out == [{"chunk_id": "a", "content": "hello"}]

    def test_nested_entity_dict(self):
        out = _as_entity_list([{"entity": {"chunk_id": "a"}, "id": 1, "distance": 0.5}])
        assert out[0]["chunk_id"] == "a"
        assert out[0]["id"] == 1
        assert out[0]["score"] == 0.5

    def test_none_and_empty_filtered(self):
        assert _as_entity_list([None, {}, _doc("a")]) == [_doc("a")]

    def test_empty_input(self):
        assert _as_entity_list(None) == []
        assert _as_entity_list([]) == []
