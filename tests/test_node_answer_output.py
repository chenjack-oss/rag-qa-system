"""node_answer_output 单元测试：答案短路 / Prompt 组装 / 图片提取

覆盖范围（全部为纯逻辑，不触发真实 LLM 与 MongoDB）：
- `_extract_images_from_docs`：url 字段后缀判定、Markdown 图片正则、跨文档去重
- `step_2_construct_prompt`：重写问题优先、上下文 12000 字符预算截断、
  history / item_names 组装、文档元数据头格式
- `step_1_check_answer`：answer 已存在时短路并写入 task result

说明：node_answer_output 的 pymongo（写历史）与 langchain（LLM 调用）均改为
函数内延迟导入，因此本文件可在 CI 轻量环境运行；step_2 的 prompt 渲染通过
monkeypatch 捕获模板变量，不依赖 prompts/ 目录的文件内容。
"""

import pytest

from app.query_process.agent.nodes import node_answer_output as node_mod
from app.query_process.agent.nodes.node_answer_output import (
    _extract_images_from_docs,
    step_1_check_answer,
    step_2_construct_prompt,
)
from app.utils.task_utils import get_task_result


class TestExtractImagesFromDocs:
    """从重排文档中抽取可展示图片 URL。"""

    def test_none_and_empty_input_return_empty_list(self):
        assert _extract_images_from_docs(None) == []
        assert _extract_images_from_docs([]) == []

    def test_url_field_with_image_suffix_is_kept(self):
        docs = [{"url": "http://example.com/panel_view.JPG", "text": "外观图"}]

        assert _extract_images_from_docs(docs) == ["http://example.com/panel_view.JPG"]

    def test_url_field_with_non_image_suffix_is_ignored(self):
        docs = [
            {"url": "http://example.com/help.html", "text": "网页"},
            {"url": "http://example.com/manual.pdf", "text": "手册"},
        ]

        assert _extract_images_from_docs(docs) == []

    def test_markdown_images_are_extracted_from_text(self):
        docs = [{"text": "参考下图：![面板图](http://img.local/1.png) 完成。"}]

        assert _extract_images_from_docs(docs) == ["http://img.local/1.png"]

    def test_multiple_markdown_images_keep_document_order(self):
        docs = [{"text": "![a](http://img/1.jpg) 中间 ![b](http://img/2.jpg)"}]

        assert _extract_images_from_docs(docs) == ["http://img/1.jpg", "http://img/2.jpg"]

    def test_duplicates_are_removed_across_docs_and_sources(self):
        docs = [
            {"url": "http://img/dup.jpg", "text": "字段命中"},
            {"text": "正文再次出现 ![重复](http://img/dup.jpg)"},
            {"text": "同一文档内重复 ![x](http://img/dup.jpg)"},
        ]

        assert _extract_images_from_docs(docs) == ["http://img/dup.jpg"]

    def test_blank_fields_are_ignored(self):
        docs = [{"url": "   ", "text": "\n  \n"}, {"url": None, "text": None}]

        assert _extract_images_from_docs(docs) == []


@pytest.fixture
def captured_prompt(monkeypatch):
    """捕获 step_2 传给 load_prompt 的模板变量，避免依赖 prompts/ 文件内容。"""
    captured: dict = {}

    def _fake_load_prompt(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return "RENDERED"

    monkeypatch.setattr(node_mod, "load_prompt", _fake_load_prompt)
    return captured


class TestStep2ConstructPrompt:
    """Prompt 组装：问题选取、上下文字符预算、历史与商品格式化。"""

    def test_rewritten_query_takes_priority_over_original(self, captured_prompt):
        prompt = step_2_construct_prompt(
            {"original_query": "原问题", "rewritten_query": "改写后的问题"}
        )

        assert prompt == "RENDERED"
        assert captured_prompt["name"] == "answer_out"
        assert captured_prompt["question"] == "改写后的问题"

    def test_original_query_used_when_no_rewrite(self, captured_prompt):
        step_2_construct_prompt({"original_query": "原问题"})

        assert captured_prompt["question"] == "原问题"

    def test_empty_docs_fall_back_to_placeholder(self, captured_prompt):
        step_2_construct_prompt({"original_query": "q"})

        assert captured_prompt["context"] == "无参考内容"

    def test_doc_meta_header_contains_source_id_url_score_title(self, captured_prompt):
        docs = [
            {
                "text": "正文内容",
                "source": "local",
                "chunk_id": "c1",
                "url": "http://img.local/a.jpg",
                "title": "操作手册",
                "score": 0.95,
            }
        ]

        step_2_construct_prompt({"original_query": "q", "reranked_docs": docs})

        context = captured_prompt["context"]
        assert "[1] [local] [chunk_id=c1] [url=http://img.local/a.jpg] [score=0.9500] [title=操作手册]" in context
        assert context.endswith("正文内容")

    def test_doc_without_score_omits_score_segment(self, captured_prompt):
        step_2_construct_prompt(
            {"original_query": "q", "reranked_docs": [{"text": "正文", "chunk_id": "c2"}]}
        )

        assert "[score=" not in captured_prompt["context"]

    def test_blank_text_docs_are_skipped(self, captured_prompt):
        docs = [{"text": "   ", "chunk_id": "empty"}, {"text": "有效正文", "chunk_id": "c3"}]

        step_2_construct_prompt({"original_query": "q", "reranked_docs": docs})

        # 空文本文档被 continue 跳过；序号 i 仍按原始顺序递增，因此有效文档为 [2]
        assert captured_prompt["context"] == "[2] [chunk_id=c3]\n有效正文"

    def test_context_budget_truncates_oversized_documents(self, captured_prompt):
        """单文档接近 12000 字符预算时，后续文档应被截断丢弃。"""
        docs = [
            {"text": "A" * 8000, "chunk_id": "big-1"},
            {"text": "B" * 8000, "chunk_id": "big-2"},
        ]

        step_2_construct_prompt({"original_query": "q", "reranked_docs": docs})

        context = captured_prompt["context"]
        assert "A" * 8000 in context
        assert "B" * 8000 not in context

    def test_history_is_formatted_with_roles(self, captured_prompt):
        history = [
            {"role": "user", "text": "怎么开机？"},
            {"role": "assistant", "text": "长按电源键。"},
        ]

        step_2_construct_prompt({"original_query": "q", "history": history})

        history_str = captured_prompt["history"]
        assert "用户: 怎么开机？" in history_str
        assert "助手: 长按电源键。" in history_str

    def test_missing_history_falls_back_to_placeholder(self, captured_prompt):
        step_2_construct_prompt({"original_query": "q"})

        assert captured_prompt["history"] == "无历史对话"

    def test_item_names_are_joined_by_comma(self, captured_prompt):
        step_2_construct_prompt(
            {"original_query": "q", "item_names": ["HAK 180 烫金机", "覆膜机"]}
        )

        assert captured_prompt["item_names"] == "HAK 180 烫金机, 覆膜机"

    def test_missing_item_names_falls_back_to_placeholder(self, captured_prompt):
        step_2_construct_prompt({"original_query": "q"})

        assert captured_prompt["item_names"] == "无指定商品"


class TestStep1CheckAnswerShortCircuit:
    """answer 已存在时的短路行为（图生文 / 二次问答场景）。"""

    def test_returns_false_when_answer_absent(self):
        assert not step_1_check_answer({"session_id": "s-no-answer"})

    def test_returns_false_when_answer_is_blank_string(self):
        assert not step_1_check_answer({"session_id": "s-blank", "answer": ""})

    def test_returns_true_and_stores_task_result_when_answer_present(self):
        state = {"session_id": "s-hit", "answer": "已生成的答案", "is_stream": False}

        assert step_1_check_answer(state) is True
        assert get_task_result("s-hit", "answer") == "已生成的答案"
