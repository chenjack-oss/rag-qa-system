"""app/utils/mcp_result_parser.py 的单元测试。

为什么测这个函数：
`parse_mcp_pages` 是 MCP 联网兜底链路里**唯一的结构化出口**——上游是百炼返回的一段
JSON 字符串，下游直接进生成模型的上下文。它此前内联在节点函数里，且节点模块顶部
`from agents.mcp import MCPServerSse` 会让整个文件在未装 openai-agents SDK 的环境
（CI、本地 venv）导入失败，等于这段代码从来没被自动化检查覆盖过。

这里重点锁三类回归：
1. **顶层结构异常**：MCP 直接返回数组 / 字符串 / null → 必须降级为 []，不能抛；
2. **字段类型异常**：snippet 是数字或布尔时 `.strip()` 会 AttributeError → 必须跳过；
3. **过滤语义**：空摘要条目必须丢弃（否则白占上下文预算），但缺 title/url 要保留整条。
"""

import json

import pytest

from app.utils import mcp_result_parser


def _raw(pages, **extra):
    payload = {"pages": pages}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 正常解析
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_single_page(self):
        raw = _raw([{"title": "手册", "url": "http://a.com", "snippet": "局部转印设置"}])
        assert mcp_result_parser.parse_mcp_pages(raw) == [
            {"title": "手册", "url": "http://a.com", "snippet": "局部转印设置"}
        ]

    def test_multiple_pages_keep_order(self):
        raw = _raw(
            [
                {"title": "t1", "url": "u1", "snippet": "s1"},
                {"title": "t2", "url": "u2", "snippet": "s2"},
                {"title": "t3", "url": "u3", "snippet": "s3"},
            ]
        )
        assert [d["snippet"] for d in mcp_result_parser.parse_mcp_pages(raw)] == ["s1", "s2", "s3"]

    def test_fields_are_stripped(self):
        raw = _raw([{"title": "  标题  ", "url": "  http://a.com  ", "snippet": "  摘要  "}])
        assert mcp_result_parser.parse_mcp_pages(raw) == [
            {"title": "标题", "url": "http://a.com", "snippet": "摘要"}
        ]

    def test_missing_title_and_url_kept_as_empty(self):
        """缺 title/url 不影响摘要可用性，整条保留。"""
        raw = _raw([{"snippet": "只有摘要"}])
        assert mcp_result_parser.parse_mcp_pages(raw) == [
            {"title": "", "url": "", "snippet": "只有摘要"}
        ]

    def test_extra_fields_are_dropped(self):
        raw = _raw([{"title": "t", "url": "u", "snippet": "s", "score": 0.9, "raw": {"x": 1}}])
        assert mcp_result_parser.parse_mcp_pages(raw) == [{"title": "t", "url": "u", "snippet": "s"}]

    def test_chinese_not_escaped(self):
        raw = _raw([{"title": "烫金机", "url": "u", "snippet": "出厂默认状态"}])
        assert "烫金机" in mcp_result_parser.parse_mcp_pages(raw)[0]["title"]


# ---------------------------------------------------------------------------
# 摘要过滤
# ---------------------------------------------------------------------------

class TestSnippetFilter:
    def test_empty_snippet_dropped(self):
        raw = _raw([{"title": "t", "url": "u", "snippet": ""}])
        assert mcp_result_parser.parse_mcp_pages(raw) == []

    def test_whitespace_snippet_dropped(self):
        raw = _raw([{"title": "t", "url": "u", "snippet": "   \n\t  "}])
        assert mcp_result_parser.parse_mcp_pages(raw) == []

    def test_missing_snippet_dropped(self):
        raw = _raw([{"title": "t", "url": "u"}])
        assert mcp_result_parser.parse_mcp_pages(raw) == []

    def test_mixed_pages_keep_only_valid(self):
        raw = _raw(
            [
                {"title": "good", "url": "u1", "snippet": "有效"},
                {"title": "bad", "url": "u2", "snippet": "  "},
                {"title": "good2", "url": "u3", "snippet": "也有效"},
            ]
        )
        assert [d["title"] for d in mcp_result_parser.parse_mcp_pages(raw)] == ["good", "good2"]


# ---------------------------------------------------------------------------
# 回归：字段类型异常（原实现会 AttributeError）
# ---------------------------------------------------------------------------

class TestFieldTypeRegression:
    @pytest.mark.parametrize("bad_snippet", [0, 1, False, True, [], {}, 3.14])
    def test_non_string_snippet_dropped_without_raise(self, bad_snippet):
        """原实现 `(item.get("snippet") or "").strip()` 对数字/布尔会 AttributeError。"""
        raw = _raw([{"title": "t", "url": "u", "snippet": bad_snippet}])
        assert mcp_result_parser.parse_mcp_pages(raw) == []

    def test_non_string_title_becomes_empty(self):
        raw = _raw([{"title": 123, "url": None, "snippet": "有效摘要"}])
        assert mcp_result_parser.parse_mcp_pages(raw) == [
            {"title": "", "url": "", "snippet": "有效摘要"}
        ]


# ---------------------------------------------------------------------------
# 回归：顶层结构异常（原实现会 AttributeError / TypeError 后被外层吞掉）
# ---------------------------------------------------------------------------

class TestTopLevelStructureRegression:
    @pytest.mark.parametrize("raw_text", [None, b'{"pages": []}', 12345, ["a"], {"pages": []}])
    def test_non_string_input_returns_empty(self, raw_text):
        """原实现只 catch JSONDecodeError，非 str 输入会抛 TypeError 并被外层静默吞掉。"""
        assert mcp_result_parser.parse_mcp_pages(raw_text) == []

    @pytest.mark.parametrize("raw_text", ["", "   ", "\n\t"])
    def test_blank_input_returns_empty(self, raw_text):
        assert mcp_result_parser.parse_mcp_pages(raw_text) == []

    def test_invalid_json_returns_empty(self):
        assert mcp_result_parser.parse_mcp_pages("这不是 JSON") == []

    def test_truncated_json_returns_empty(self):
        assert mcp_result_parser.parse_mcp_pages('{"pages": [{"title": "t"') == []

    @pytest.mark.parametrize("payload", [[], ["x"], "abc", 123, None, True])
    def test_non_dict_top_level_returns_empty(self, payload):
        """MCP 若直接返回数组（顶层 list），原实现 `data.get` 会 AttributeError。"""
        assert mcp_result_parser.parse_mcp_pages(json.dumps(payload)) == []

    @pytest.mark.parametrize("pages", [None, {}, "abc", 123, True])
    def test_pages_not_list_returns_empty(self, pages):
        assert mcp_result_parser.parse_mcp_pages(_raw(pages)) == []

    def test_pages_absent_returns_empty(self):
        assert mcp_result_parser.parse_mcp_pages(json.dumps({"total": 0})) == []


# ---------------------------------------------------------------------------
# 回归：pages 元素异常
# ---------------------------------------------------------------------------

class TestPageItemRegression:
    @pytest.mark.parametrize("bad_item", ["字符串", None, [], 123, True])
    def test_non_dict_items_skipped(self, bad_item):
        raw = _raw([bad_item, {"title": "ok", "url": "u", "snippet": "有效"}])
        assert mcp_result_parser.parse_mcp_pages(raw) == [
            {"title": "ok", "url": "u", "snippet": "有效"}
        ]

    def test_all_items_invalid_returns_empty(self):
        raw = _raw(["a", None, [], 1])
        assert mcp_result_parser.parse_mcp_pages(raw) == []


# ---------------------------------------------------------------------------
# 日志契约：异常路径要留痕，正常空输入不该刷 error
# ---------------------------------------------------------------------------

class TestLogging:
    def test_invalid_json_logs_error(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mcp_result_parser.logger, "error", lambda msg, *a, **kw: calls.append(msg)
        )
        mcp_result_parser.parse_mcp_pages("not json")
        assert any("解析 JSON 失败" in msg for msg in calls)

    def test_invalid_json_truncates_long_text(self, monkeypatch):
        """脏数据可能极长，日志只截前 100 字符。"""
        calls = []
        monkeypatch.setattr(
            mcp_result_parser.logger, "error", lambda msg, *a, **kw: calls.append(msg)
        )
        mcp_result_parser.parse_mcp_pages("x" * 5000)
        assert len(calls[0]) < 300

    def test_blank_input_does_not_log_error(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mcp_result_parser.logger, "error", lambda msg, *a, **kw: calls.append(msg)
        )
        mcp_result_parser.parse_mcp_pages("   ")
        assert calls == []

    def test_non_dict_top_level_logs_warning(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mcp_result_parser.logger, "warning", lambda msg, *a, **kw: calls.append(msg)
        )
        mcp_result_parser.parse_mcp_pages(json.dumps([{"title": "t"}]))
        assert any("顶层不是对象" in msg for msg in calls)
