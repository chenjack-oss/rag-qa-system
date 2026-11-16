"""node_web_search_mcp 节点编排逻辑的单元测试。

为什么现在才测得了：
该节点此前在模块顶部 `from agents.mcp import MCPServerSse`，而 openai-agents SDK
既不在 CI 的安装列表里、本地 venv 也没装，导致**整个模块导入即失败**——这段联网兜底
逻辑长期处于自动化检查的盲区，只能靠手工连真实 MCP 服务验证。

把 SDK 导入下沉到 `mcp_call()` 内部后，节点编排本身（query 回退、空查询短路、
结果解析、异常隔离、返回值形态）都可以用替身 `mcp_call` 完整覆盖，不需要网络。

覆盖点：
- 正常路径：result.content[0].text → parse_mcp_pages → {"web_search_docs": [...]}
- 短路路径：无 query 时不发起调用
- 降级路径：isError / content 为空 / result 为 None / JSON 损坏 → 返回 {}
- 健壮性：mcp_call 抛异常必须被吞掉，节点不能把主链路带崩
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

node_module = importlib.import_module(
    "app.query_process.agent.nodes.node_web_search_mcp"
)
node_web_search_mcp = node_module.node_web_search_mcp


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------

class FakeContent:
    def __init__(self, text):
        self.text = text


class FakeResult:
    def __init__(self, text=None, is_error=False, contents=None):
        if contents is not None:
            self.content = contents
        else:
            self.content = [FakeContent(text)] if text is not None else []
        self.isError = is_error


def _state(**overrides):
    state = {
        "session_id": "sess-test",
        "rewritten_query": "HAK 180 局部烫金设置",
        "is_stream": False,
    }
    state.update(overrides)
    return state


@pytest.fixture()
def calls(monkeypatch):
    """读取被调用过的 query 列表，返回这个列表本身。"""
    seen = []

    def make(result):
        async def fake_mcp_call(query):
            seen.append(query)
            if isinstance(result, Exception):
                raise result
            return result

        return fake_mcp_call

    def install(result):
        monkeypatch.setattr(node_module, "mcp_call", make(result))
        return seen

    return install


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------

def test_valid_result_returns_structured_docs(calls):
    raw = json.dumps(
        {"pages": [{"title": "手册", "url": "http://a.com", "snippet": "局部转印设置"}]},
        ensure_ascii=False,
    )
    calls(FakeResult(text=raw))

    out = node_web_search_mcp(_state())

    assert out == {"web_search_docs": [{"title": "手册", "url": "http://a.com", "snippet": "局部转印设置"}]}


def test_query_passed_to_mcp(calls):
    seen = calls(FakeResult(text='{"pages": []}'))

    node_web_search_mcp(_state(rewritten_query="查这个"))

    assert seen == ["查这个"]


def test_filters_empty_snippet_entries(calls):
    raw = json.dumps(
        {
            "pages": [
                {"title": "a", "url": "u1", "snippet": "有效"},
                {"title": "b", "url": "u2", "snippet": "   "},
            ]
        },
        ensure_ascii=False,
    )
    calls(FakeResult(text=raw))

    out = node_web_search_mcp(_state())

    assert len(out["web_search_docs"]) == 1
    assert out["web_search_docs"][0]["title"] == "a"


# ---------------------------------------------------------------------------
# query 回退与短路
# ---------------------------------------------------------------------------

def test_falls_back_to_original_query(calls):
    seen = calls(FakeResult(text='{"pages": []}'))

    node_web_search_mcp(_state(rewritten_query="", original_query="原始问题"))

    assert seen == ["原始问题"]


def test_empty_query_skips_mcp_call(calls):
    seen = calls(FakeResult(text='{"pages": []}'))

    out = node_web_search_mcp(_state(rewritten_query="", original_query=""))

    assert seen == []
    assert out == {}


# ---------------------------------------------------------------------------
# 降级路径
# ---------------------------------------------------------------------------

def test_is_error_returns_empty(calls):
    calls(FakeResult(is_error=True, contents=[FakeContent("err")]))

    assert node_web_search_mcp(_state()) == {}


def test_none_result_returns_empty(calls):
    calls(None)

    assert node_web_search_mcp(_state()) == {}


def test_empty_content_returns_empty(calls):
    calls(FakeResult(contents=[]))

    assert node_web_search_mcp(_state()) == {}


def test_malformed_json_returns_empty(calls):
    calls(FakeResult(text="<html>502 Bad Gateway</html>"))

    assert node_web_search_mcp(_state()) == {}


def test_pages_empty_returns_empty(calls):
    calls(FakeResult(text='{"pages": []}'))

    assert node_web_search_mcp(_state()) == {}


# ---------------------------------------------------------------------------
# 异常隔离
# ---------------------------------------------------------------------------

def test_mcp_exception_is_swallowed(calls):
    calls(RuntimeError("SSE 连接超时"))

    # 联网兜底失败不能让检索主链路 500
    assert node_web_search_mcp(_state()) == {}


def test_state_missing_rewritten_query_key_is_safe(calls):
    calls(FakeResult(text='{"pages": []}'))

    assert node_web_search_mcp({"session_id": "s", "original_query": "原始"}) == {}
