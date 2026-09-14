"""node_rerank 单元测试：文档合并（step_1）与动态 TopK 截断（step_3）

覆盖范围：
- step_1_merge_docs：多路召回异构结果的标准化合并（本地 RRF + 联网搜索）
- step_3_topk：结合硬上限/硬下限 + 断崖阈值的动态 TopK 截断

说明：step_2_rerank_docs 依赖 FlagEmbedding（级联 torch），node_rerank.py 对其
采用延迟导入，因此本文件可在轻量测试环境（CI 仅安装 pytest/numpy/loguru/fastapi）中运行。
"""

from app.query_process.agent.nodes.node_rerank import step_1_merge_docs, step_3_topk


def _local_doc(chunk_id: str, content: str = "正文", **extra) -> dict:
    """构造一条本地知识库召回结果。"""
    return {"chunk_id": chunk_id, "content": content, **extra}


def _scored(*scores: float) -> list:
    """按给定分数构造已降序排列的 scored_docs。"""
    return [{"score": s} for s in scores]


class TestStep1MergeDocs:
    """文档合并与标准化。"""

    def test_merge_empty_sources_returns_empty(self):
        assert step_1_merge_docs({}) == []
        assert step_1_merge_docs({"rrf_chunks": [], "web_search_docs": []}) == []

    def test_merge_rrf_only_maps_fields(self):
        state = {"rrf_chunks": [_local_doc("c1", "首条内容", title="算法介绍"), _local_doc("c2")]}

        docs = step_1_merge_docs(state)

        assert len(docs) == 2
        assert docs[0] == {
            "text": "首条内容",
            "doc_id": "c1",
            "chunk_id": "c1",
            "title": "算法介绍",
            "url": "",
            "source": "local",
        }
        # 无 title / item_name 时标题回退为空串，不抛异常
        assert docs[1]["title"] == ""
        assert docs[1]["source"] == "local"

    def test_merge_rrf_entity_wrapper_is_unwrapped(self):
        state = {"rrf_chunks": [{"entity": {"chunk_id": "c9", "content": "包裹内容"}, "distance": 0.71}]}

        docs = step_1_merge_docs(state)

        assert len(docs) == 1
        assert docs[0]["text"] == "包裹内容"
        assert docs[0]["chunk_id"] == "c9"

    def test_merge_rrf_doc_id_falls_back_to_id(self):
        state = {"rrf_chunks": [{"id": "legacy-1", "content": "旧字段兼容"}]}

        docs = step_1_merge_docs(state)

        assert docs[0]["doc_id"] == "legacy-1"
        assert docs[0]["chunk_id"] == "legacy-1"

    def test_merge_rrf_title_falls_back_to_item_name(self):
        state = {"rrf_chunks": [{"chunk_id": "c3", "content": "说明", "item_name": "HAK 180 烫金机"}]}

        docs = step_1_merge_docs(state)

        assert docs[0]["title"] == "HAK 180 烫金机"

    def test_merge_skips_non_dict_and_empty_content(self):
        state = {
            "rrf_chunks": [
                "not-a-dict",
                123,
                None,
                {"chunk_id": "c1"},  # 缺 content
                {"chunk_id": "c2", "content": ""},  # 空 content
                {"chunk_id": "c3", "content": "唯一有效内容"},
            ]
        }

        docs = step_1_merge_docs(state)

        assert len(docs) == 1
        assert docs[0]["chunk_id"] == "c3"

    def test_merge_web_only_strips_and_marks_source(self):
        state = {"web_search_docs": [{"title": " 标题 ", "url": " http://a.com ", "snippet": " 摘要 "}]}

        docs = step_1_merge_docs(state)

        assert len(docs) == 1
        assert docs[0] == {
            "text": "摘要",
            "doc_id": None,
            "chunk_id": None,
            "title": "标题",
            "url": "http://a.com",
            "source": "web",
        }

    def test_merge_web_text_falls_back_to_content(self):
        state = {"web_search_docs": [{"snippet": "", "content": "摘要兜底"}, {"content": "仅 content"}]}

        docs = step_1_merge_docs(state)

        assert [d["text"] for d in docs] == ["摘要兜底", "仅 content"]

    def test_merge_web_skips_blank_text(self):
        state = {"web_search_docs": [{"snippet": "   "}, {"title": "只有标题"}, {"content": "\n\t"}]}

        assert step_1_merge_docs(state) == []

    def test_merge_mixed_sources_keeps_local_first(self):
        state = {
            "rrf_chunks": [_local_doc("c1", "本地")],
            "web_search_docs": [{"snippet": "联网", "url": "http://b.com"}],
        }

        docs = step_1_merge_docs(state)

        assert [d["source"] for d in docs] == ["local", "web"]
        assert [d["text"] for d in docs] == ["本地", "联网"]


class TestStep3TopK:
    """动态 TopK：硬上限 10、硬下限 1、断崖截断。"""

    def test_topk_empty_returns_empty(self):
        assert step_3_topk([]) == []

    def test_topk_single_doc_kept(self):
        assert len(step_3_topk(_scored(0.9))) == 1

    def test_topk_respects_hard_max(self):
        # 11 条且相邻分差极小（gap=0.001 / rel≈0.1%），不触发断崖 -> 取满硬上限 10
        docs = _scored(*[1.0 - 0.001 * i for i in range(11)])

        result = step_3_topk(docs)

        assert len(result) == 10

    def test_topk_smooth_scores_keep_all_below_max(self):
        docs = _scored(1.0, 0.99, 0.98, 0.97, 0.96)

        assert len(step_3_topk(docs)) == 5

    def test_topk_absolute_gap_triggers_early_cut(self):
        # 5.0 -> 4.9 安全；4.9 -> 4.0 gap=0.9 >= 0.5（相对 18.4% < 25%）-> 仅绝对阈值触发
        docs = _scored(5.0, 4.9, 4.0, 3.9, 3.8)

        result = step_3_topk(docs)

        assert len(result) == 2
        assert [d["score"] for d in result] == [5.0, 4.9]

    def test_topk_relative_gap_triggers_early_cut(self):
        # 1.0 -> 0.98 安全；0.98 -> 0.71 gap=0.27 < 0.5 但相对 27.6% >= 25% -> 仅相对阈值触发
        docs = _scored(1.0, 0.98, 0.71, 0.7, 0.69)

        result = step_3_topk(docs)

        assert len(result) == 2
        assert [d["score"] for d in result] == [1.0, 0.98]

    def test_topk_two_docs_large_gap_cuts_to_one(self):
        result = step_3_topk(_scored(1.0, 0.1))

        assert len(result) == 1
        assert result[0]["score"] == 1.0

    def test_topk_preserves_input_order(self):
        docs = _scored(0.9, 0.89, 0.88, 0.87)

        result = step_3_topk(docs)

        assert [d["score"] for d in result] == [0.9, 0.89, 0.88, 0.87]

    def test_topk_returns_same_dict_objects(self):
        docs = [{"score": 0.9, "chunk_id": "c1"}, {"score": 0.89, "chunk_id": "c2"}]

        result = step_3_topk(docs)

        assert result is not docs
        assert result[0] is docs[0]
