"""MCP 联网搜索结果的解析工具。

背景：
`node_web_search_mcp` 是检索链路里"本地向量检索拿不到东西时兜底联网"的最后一环。
百炼 MCP 返回的 `result.content[0].text` 是一段 JSON 字符串，形如：

    {"pages": [{"title": "...", "url": "...", "snippet": "..."}, ...]}

解析逻辑原先内联在节点函数里，导致两个问题：
1. 无法单独测试——必须真的连上 MCP 服务才能跑到这段代码；
2. 结构异常会静默吞掉——`data` 不是 dict、`pages` 不是 list、`snippet` 不是字符串
   都会抛 `AttributeError`，被节点外层 `except Exception` 兜住，表现为
   "搜索结果明明有，页面却是空的"，线上极难定位。

因此把解析抽成纯函数，并明确容错契约：**任何结构异常一律降级为空列表，不向上抛**。
理由是联网兜底属于可选增强路径，解析失败不能让主检索链路 500。
"""

import json
from typing import Any, Dict, List

from app.core.logger import logger


def _as_text(value: Any) -> str:
    """把字段值收敛成去首尾空白的字符串。

    只接受 `str`：MCP 的 schema 里 title/url/snippet 都是字符串，出现数字、
    布尔、None、嵌套结构说明上游异常，按缺失处理比强行 `str()` 更安全——
    否则会把 `0` / `False` 之类当成有效摘要塞进上下文，污染生成答案。
    """
    if isinstance(value, str):
        return value.strip()
    return ""


def parse_mcp_pages(raw_text: Any) -> List[Dict[str, str]]:
    """把 MCP 返回的原始文本解析为 `[{title, url, snippet}]`。

    容错规则（全部返回空列表或跳过该条，不抛异常）：
    - `raw_text` 非字符串 / 空白 → `[]`
    - JSON 解析失败（含非 str 输入导致的 TypeError 场景已在上一步拦掉）→ `[]`
    - 顶层不是 dict（例如 MCP 直接返回数组）→ `[]`
    - `pages` 缺失 / 为 None / 不是 list → `[]`
    - `pages` 元素不是 dict → 跳过该条
    - `snippet` 缺失或为空白 → 跳过该条（无摘要的结果对生成无价值，只会占上下文预算）
    - `title` / `url` 缺失 → 保留空串，不丢整条（摘要仍可用）

    :param raw_text: MCP `TextContent.text` 的原始字符串
    :return: 结构化文档列表，顺序与上游返回一致
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []

    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        # 只截前 100 字符，避免超长脏数据把日志刷爆
        logger.error(f"MCP 返回结果解析 JSON 失败: {raw_text[:100]}...")
        return []

    if not isinstance(data, dict):
        logger.warning(f"MCP 返回结果顶层不是对象（实际 {type(data).__name__}），已忽略")
        return []

    pages = data.get("pages")
    if not isinstance(pages, list):
        if pages is not None:
            logger.warning(f"MCP 返回结果 pages 字段不是列表（实际 {type(pages).__name__}），已忽略")
        return []

    docs: List[Dict[str, str]] = []
    for item in pages:
        if not isinstance(item, dict):
            continue
        snippet = _as_text(item.get("snippet"))
        if not snippet:
            continue
        docs.append(
            {
                "title": _as_text(item.get("title")),
                "url": _as_text(item.get("url")),
                "snippet": snippet,
            }
        )

    return docs
