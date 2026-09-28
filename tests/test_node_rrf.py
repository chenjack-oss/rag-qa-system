"""node_rrf 节点层单元测试：多路结果规整 → 带权 RRF 融合 → 消融门控

与 test_rrf.py 的分工：
- test_rrf.py   ：`reciprocal_rank_fusion` 算法本身与 `_as_entity_list` 基础分支
- 本文件         ：`node_rrf()` 编排层（state 读写 / TopK 截断 / 返回结构）、
                  pymilvus Hit 对象分支、EVAL_ABLATION_MODE 消融开关对召回路的退化

CI 兼容：node_rrf 模块级依赖仅 loguru / fastapi（task_utils→sse_utils），
不引入 torch / pymilvus，可在轻量测试环境运行。
"""

from app.query_process.agent.nodes.node_rrf import _as_entity_list, node_rrf


class _FakeHit:
    """模拟 pymilvus Hit 对象：具备 entity / id / distance 三个属性。"""

    def __init__(self, entity, hit_id, distance=None):
        self.entity = entity
        self.id = hit_id
        if distance is not None:
            self.distance = distance


class _EntityWithToDict:
    """模拟带 to_dict() 的 Entity 对象（部分 pymilvus 版本的形态）。"""

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


class _MappingLike:
    """非 dict 但实现了 get() 的对象，用于覆盖 _as_entity_list 的分支 C。"""

    def __init__(self, payload):
        self._payload = payload

    def get(self, key, default=None):
        return self._payload.get(key, default)


class TestAsEntityListHitBranch:
    """pymilvus Hit 对象 → 规整为含 score 的 entity dict。"""

    def test_hit_with_dict_entity_keeps_chunk_id_and_distance(self):
        hits = [_FakeHit({"content": "面板图", "chunk_id": "c1"}, hit_id="pk-1", distance=0.91)]

        assert _as_entity_list(hits) == [{"content": "面板图", "chunk_id": "c1", "score": 0.91}]

    def test_hit_with_entity_to_dict_is_unwrapped(self):
        hits = [_FakeHit(_EntityWithToDict({"content": "操作手册", "chunk_id": "c2"}), hit_id="pk-2", distance=0.8)]

        out = _as_entity_list(hits)

        assert out == [{"content": "操作手册", "chunk_id": "c2", "score": 0.8}]

    def test_hit_without_chunk_id_falls_back_to_outer_id(self):
        """entity 内无 chunk_id/id 时，用 Hit 外层主键补齐，避免 RRF 丢失该条。"""
        hits = [_FakeHit({"content": "无主键字段"}, hit_id="pk-3", distance=0.7)]

        out = _as_entity_list(hits)

        assert out[0]["id"] == "pk-3"
        assert out[0]["score"] == 0.7

    def test_hit_without_distance_has_no_score_key(self):
        hits = [_FakeHit({"chunk_id": "c4"}, hit_id="pk-4")]

        out = _as_entity_list(hits)

        assert "score" not in out[0]

    def test_mapping_like_object_is_resolved_via_get(self):
        docs = [_MappingLike({"entity": {"chunk_id": "m1", "content": "映射对象"}})]

        assert _as_entity_list(docs) == [{"chunk_id": "m1", "content": "映射对象"}]

    def test_mapping_like_without_entity_payload_is_dropped(self):
        assert _as_entity_list([_MappingLike({"other": 1})]) == []

    def test_scalar_entries_are_skipped(self):
        assert _as_entity_list(["raw-string", 42, None, {}]) == []

    def test_hit_with_entity_but_no_id_attribute_is_dropped(self):
        class _EntityOnly:
            entity = {"content": "只有 entity 属性"}

        assert _as_entity_list([_EntityOnly()]) == []


def _chunk(chunk_id: str, content: str = "") -> dict:
    """构造一条上游检索结果（扁平结构，含 chunk_id）。"""
    return {"chunk_id": chunk_id, "content": content or f"内容-{chunk_id}"}


class TestNodeRrfOrchestration:
    """node_rrf 编排层：state 读取 → 融合 → 截断 → 返回新 state。"""

    @staticmethod
    def _state(embedding, hyde, session_id="test-rrf-node"):
        return {
            "session_id": session_id,
            "is_stream": False,
            "embedding_chunks": embedding,
            "hyde_embedding_chunks": hyde,
        }

    def test_fused_chunks_returned_under_rrf_chunks_key(self):
        state = self._state([_chunk("a"), _chunk("b")], [_chunk("a"), _chunk("c")])

        result = node_rrf(state)

        assert isinstance(result, dict)
        ids = [d["chunk_id"] for d in result["rrf_chunks"]]
        # 两路共同命中的 a 得分累加，排第一；并集共 3 条
        assert ids[0] == "a"
        assert sorted(ids) == ["a", "b", "c"]

    def test_duplicate_chunk_across_sources_appears_once(self):
        result = node_rrf(self._state([_chunk("dup")], [_chunk("dup")]))

        assert [d["chunk_id"] for d in result["rrf_chunks"]] == ["dup"]

    def test_result_is_truncated_to_ten(self):
        state = self._state([_chunk(f"e{i}") for i in range(12)], [])

        result = node_rrf(state)

        assert len(result["rrf_chunks"]) == 10

    def test_empty_state_returns_empty_result(self):
        result = node_rrf({"session_id": "empty-session", "is_stream": False})

        assert result == {"rrf_chunks": []}

    def test_missing_chunk_id_entries_are_skipped_without_crash(self):
        state = self._state([{"content": "无 id 字段"}, _chunk("ok")], [])

        result = node_rrf(state)

        assert [d["chunk_id"] for d in result["rrf_chunks"]] == ["ok"]

    def test_ablation_no_rrf_degrades_to_dense_source_only(self, monkeypatch):
        """EVAL_ABLATION_MODE=no_rrf：HyDE 路被旁路，仅保留稠密主路。"""
        monkeypatch.setenv("EVAL_ABLATION_MODE", "no_rrf")
        state = self._state([_chunk("dense")], [_chunk("hyde_only")])

        ids = [d["chunk_id"] for d in node_rrf(state)["rrf_chunks"]]

        assert ids == ["dense"]

    def test_full_pipeline_keeps_hyde_only_docs(self, monkeypatch):
        monkeypatch.delenv("EVAL_ABLATION_MODE", raising=False)
        state = self._state([_chunk("dense")], [_chunk("hyde_only")])

        ids = [d["chunk_id"] for d in node_rrf(state)["rrf_chunks"]]

        assert set(ids) == {"dense", "hyde_only"}

    def test_invalid_ablation_mode_falls_back_to_full_pipeline(self, monkeypatch):
        """非法消融值只告警不生效，链路行为与完整模式一致。"""
        monkeypatch.setenv("EVAL_ABLATION_MODE", "not-a-real-mode")
        state = self._state([_chunk("dense")], [_chunk("hyde_only")])

        ids = [d["chunk_id"] for d in node_rrf(state)["rrf_chunks"]]

        assert set(ids) == {"dense", "hyde_only"}

    def test_input_chunks_are_not_mutated(self, monkeypatch):
        monkeypatch.delenv("EVAL_ABLATION_MODE", raising=False)
        embedding = [_chunk("d1"), _chunk("d2")]
        hyde = [_chunk("h1")]

        node_rrf(self._state(embedding, hyde))

        assert embedding == [_chunk("d1"), _chunk("d2")]
        assert hyde == [_chunk("h1")]
