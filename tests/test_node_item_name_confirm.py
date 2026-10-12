"""node_item_name_confirm 单元测试：商品名提取 / 检索对齐 / 确认分支

覆盖范围（纯逻辑 + 延迟导入替身，全程不触发真实 LLM、Milvus 与 MongoDB）：
- `step_5_align_item_names`：三级评分决策（>0.85 高置信 / >=0.6 中置信）的规则 A/B/C/D、
  0.85 与 0.6 两个阈值边界、无匹配跳过与结果去重
- `step_3_extract_info`：Prompt 变量传递、LLM JSON 解析、Markdown 代码块清理、
  缺字段兜底、非法 JSON 与 Prompt 加载失败的容错
- `step_4_vectorize_and_query`：环境变量缺失 / Milvus 不可用时短路、
  hit → matches 映射、混合检索权重与 limit 传参
- `step_6_check_confirmation`：确认 / 反问 / 拒绝三分支、只更新未关联商品名的历史消息、
  align_result=None 的健壮性处理
- `step_7_write_history`：助手消息仅在存在 answer 时写入、用户消息参数校验
- `node_item_name_confirm`：全流程 state 流转冒烟 + 任务状态标记

说明：本模块的 langchain_core / pymilvus / pymongo / FlagEmbedding 已改为函数内延迟导入，
因此本文件可在 CI 轻量环境运行。对延迟导入的依赖通过在 `sys.modules` 注入替身模块拦截，
无需安装 langchain、pymilvus、pymongo、torch。
"""

import sys
import types

import pytest

from app.query_process.agent.nodes import node_item_name_confirm as node_mod
from app.query_process.agent.nodes.node_item_name_confirm import (
    node_item_name_confirm,
    step_3_extract_info,
    step_4_vectorize_and_query,
    step_5_align_item_names,
    step_6_check_confirmation,
    step_7_write_history,
)

# --------------------------------------------------------------------------------------
# 延迟导入替身基础设施
# --------------------------------------------------------------------------------------


def _install_module(monkeypatch, name, **attrs):
    """把 `name` 的替身模块注入 sys.modules（monkeypatch 保证用例结束后自动还原）。"""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _match(item_name, score):
    """构造一条 Milvus 命中记录（与 step_4 产出的 matches 元素同构）。"""
    return {"item_name": item_name, "score": score}


class _StubMessage:
    """替代 langchain_core.messages 的 HumanMessage / SystemMessage。"""

    def __init__(self, content):
        self.content = content


class _StubLLMClient:
    """按顺序返回预设响应的 LLM 客户端替身，用于替代 get_llm_client() 的返回值。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        payload = self._responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return types.SimpleNamespace(content=payload)


@pytest.fixture
def stub_llm(monkeypatch):
    """注入 langchain_core.messages 与 app.lm.lm_utils 的替身。

    返回安装函数：`stub_llm(响应文本或异常, ...) -> _StubLLMClient`。
    """
    _install_module(
        monkeypatch,
        "langchain_core.messages",
        HumanMessage=_StubMessage,
        SystemMessage=_StubMessage,
    )
    clients = []

    def _fake_get_llm_client(json_mode=False, **kwargs):
        return clients[-1]

    _install_module(monkeypatch, "app.lm.lm_utils", get_llm_client=_fake_get_llm_client)

    def _install(*responses):
        client = _StubLLMClient(responses)
        clients.append(client)
        return client

    return _install


@pytest.fixture
def fake_mongo(monkeypatch):
    """注入 app.clients.mongo_history_utils 替身，并记录全部写库参数。

    替身签名与真实 `save_chat_message` 完全一致：node 入口用位置参数调用，
    step_7 用关键字参数调用，两者都要能接住。
    """
    calls = {"saved": [], "updated": [], "recent": [], "history": []}

    def _save_chat_message(
        session_id,
        role,
        text,
        rewritten_query="",
        item_names=None,
        image_urls=None,
        message_id=None,
    ):
        calls["saved"].append(
            {
                "session_id": session_id,
                "role": role,
                "text": text,
                "rewritten_query": rewritten_query,
                "item_names": item_names,
                "image_urls": image_urls,
                "message_id": message_id,
            }
        )
        return f"mid-{len(calls['saved'])}"

    def _update_message_item_names(ids, item_names):
        calls["updated"].append({"ids": ids, "item_names": item_names})

    def _get_recent_messages(session_id, limit=10):
        calls["recent"].append({"session_id": session_id, "limit": limit})
        return calls["history"]

    _install_module(
        monkeypatch,
        "app.clients.mongo_history_utils",
        save_chat_message=_save_chat_message,
        update_message_item_names=_update_message_item_names,
        get_recent_messages=_get_recent_messages,
    )
    return calls


@pytest.fixture
def fake_milvus(monkeypatch):
    """注入 Milvus 与 Embedding 替身，并记录混合检索的传参。"""
    recorder = {
        "client": object(),
        "search_result": [],
        "collections": [],
        "ranker_weights": None,
        "search_limit": None,
    }

    def _get_milvus_client():
        return recorder["client"]

    def _create_hybrid_search_requests(dense_vector=None, sparse_vector=None, limit=5):
        return {"dense": dense_vector, "sparse": sparse_vector, "limit": limit}

    def _hybrid_search(
        client=None,
        collection_name=None,
        reqs=None,
        ranker_weights=None,
        limit=5,
        **kwargs,
    ):
        recorder["collections"].append(collection_name)
        recorder["ranker_weights"] = ranker_weights
        recorder["search_limit"] = limit
        return recorder["search_result"]

    def _generate_embeddings(texts):
        return {"dense": [[0.1, 0.2] for _ in texts], "sparse": [{1: 0.5} for _ in texts]}

    _install_module(
        monkeypatch,
        "app.clients.milvus_utils",
        get_milvus_client=_get_milvus_client,
        create_hybrid_search_requests=_create_hybrid_search_requests,
        hybrid_search=_hybrid_search,
    )
    _install_module(monkeypatch, "app.lm.embedding_utils", generate_embeddings=_generate_embeddings)
    return recorder


@pytest.fixture
def captured_prompt(monkeypatch):
    """捕获 step_3 传给 load_prompt 的模板变量，避免依赖 prompts/ 文件内容。"""
    captured = {}

    def _fake_load_prompt(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return "RENDERED"

    monkeypatch.setattr(node_mod, "load_prompt", _fake_load_prompt)
    return captured


# --------------------------------------------------------------------------------------
# step_5：三级评分决策（纯逻辑，全组最有价值）
# --------------------------------------------------------------------------------------


class TestStep5AlignItemNames:
    """商品名对齐：>0.85 直接确认；0.6~0.85 进候选；全 <0.6 判无匹配。"""

    def test_rule_a_single_high_confidence_is_confirmed(self):
        result = step_5_align_item_names(
            [{"extracted_name": "烫金机", "matches": [_match("HAK 180 烫金机", 0.93)]}]
        )

        assert result["confirmed_item_names"] == ["HAK 180 烫金机"]
        assert result["options"] == []

    def test_rule_b_multiple_high_prefers_exact_same_name(self):
        """同名优先于最高分：extracted_name 与较低分命中同名时，应取同名项。"""
        result = step_5_align_item_names(
            [
                {
                    "extracted_name": "HAK 180 半自动",
                    "matches": [
                        _match("HAK 180 全自动", 0.97),
                        _match("HAK 180 半自动", 0.88),
                    ],
                }
            ]
        )

        assert result["confirmed_item_names"] == ["HAK 180 半自动"]

    def test_rule_b_multiple_high_without_exact_match_uses_highest_score(self):
        result = step_5_align_item_names(
            [
                {
                    "extracted_name": "烫金机",
                    "matches": [
                        _match("HAK 180 半自动", 0.90),
                        _match("HAK 180 全自动", 0.96),
                    ],
                }
            ]
        )

        assert result["confirmed_item_names"] == ["HAK 180 全自动"]

    def test_rule_c_mid_confidence_goes_to_options(self):
        result = step_5_align_item_names(
            [{"extracted_name": "烫金机", "matches": [_match("HAK 180 半自动", 0.70)]}]
        )

        assert result["confirmed_item_names"] == []
        assert result["options"] == ["HAK 180 半自动"]

    def test_rule_c_options_are_truncated_to_top5(self):
        """中置信候选超过 5 个时，按分数降序保留前 5（防止反问列表过长）。"""
        matches = [_match(f"型号-{i}", 0.60 + i * 0.01) for i in range(7)]

        result = step_5_align_item_names([{"extracted_name": "烫金机", "matches": matches}])

        assert len(result["options"]) == 5
        assert set(result["options"]) == {f"型号-{i}" for i in range(2, 7)}

    def test_rule_d_all_low_confidence_yields_no_match(self):
        result = step_5_align_item_names(
            [{"extracted_name": "烫金机", "matches": [_match("无关产品", 0.59)]}]
        )

        assert result["confirmed_item_names"] == []
        assert result["options"] == []

    def test_boundary_score_exactly_0_85_is_not_high_confidence(self):
        """0.85 是开区间上限：判定条件是 > 0.85，恰好 0.85 应落入中置信候选。"""
        result = step_5_align_item_names(
            [{"extracted_name": "烫金机", "matches": [_match("HAK 180 烫金机", 0.85)]}]
        )

        assert result["confirmed_item_names"] == []
        assert result["options"] == ["HAK 180 烫金机"]

    def test_boundary_score_exactly_0_6_is_mid_confidence(self):
        """0.6 是闭区间下限：判定条件是 >= 0.6，恰好 0.6 应进候选。"""
        result = step_5_align_item_names(
            [{"extracted_name": "烫金机", "matches": [_match("HAK 180 烫金机", 0.6)]}]
        )

        assert result["options"] == ["HAK 180 烫金机"]

    def test_unsorted_input_is_sorted_before_picking(self):
        """输入乱序时也要取最高分兜底。"""
        result = step_5_align_item_names(
            [
                {
                    "extracted_name": "烫金机",
                    "matches": [
                        _match("低分型号", 0.87),
                        _match("最高分型号", 0.99),
                        _match("中分型号", 0.91),
                    ],
                }
            ]
        )

        assert result["confirmed_item_names"] == ["最高分型号"]

    def test_empty_matches_are_skipped(self):
        result = step_5_align_item_names(
            [
                {"extracted_name": "无结果商品", "matches": []},
                {"extracted_name": "缺字段商品"},
                {"extracted_name": "正常商品", "matches": [_match("HAK 180 烫金机", 0.95)]},
            ]
        )

        assert result["confirmed_item_names"] == ["HAK 180 烫金机"]

    def test_results_are_deduplicated_across_extracted_names(self):
        result = step_5_align_item_names(
            [
                {"extracted_name": "烫金机", "matches": [_match("HAK 180 烫金机", 0.95)]},
                {"extracted_name": "烫金机 型号", "matches": [_match("HAK 180 烫金机", 0.92)]},
            ]
        )

        assert result["confirmed_item_names"] == ["HAK 180 烫金机"]


# --------------------------------------------------------------------------------------
# step_3：LLM 提取商品名与改写问题
# --------------------------------------------------------------------------------------


class TestStep3ExtractInfo:
    """LLM 提取：JSON 解析与各类容错路径。"""

    def test_normal_json_response_is_parsed(self, stub_llm, captured_prompt):
        client = stub_llm('{"item_names": ["HAK 180 烫金机"], "rewritten_query": "HAK 180 烫金机多少钱"}')

        result = step_3_extract_info("这个多少钱？", [{"role": "user", "text": "HAK 180"}])

        assert result == {
            "item_names": ["HAK 180 烫金机"],
            "rewritten_query": "HAK 180 烫金机多少钱",
        }
        # 传给 LLM 的应是 System + Human 两条消息
        assert len(client.calls[0]) == 2

    def test_prompt_receives_history_text_and_query(self, stub_llm, captured_prompt):
        stub_llm('{"item_names": ["A"], "rewritten_query": "q"}')

        step_3_extract_info("这个多少钱？", [{"role": "user", "text": "HAK 180"}])

        assert captured_prompt["name"] == "rewritten_query_and_itemnames"
        assert captured_prompt["query"] == "这个多少钱？"
        assert "user: HAK 180" in captured_prompt["history_text"]

    def test_markdown_code_fence_is_stripped(self, stub_llm, captured_prompt):
        stub_llm('```json\n{"item_names": ["HAK 180"], "rewritten_query": "q"}\n```')

        result = step_3_extract_info("q", [])

        assert result["item_names"] == ["HAK 180"]

    def test_missing_rewritten_query_falls_back_to_original(self, stub_llm, captured_prompt):
        stub_llm('{"item_names": ["HAK 180"]}')

        result = step_3_extract_info("原问题", [])

        assert result == {"item_names": ["HAK 180"], "rewritten_query": "原问题"}

    def test_missing_item_names_falls_back_to_empty_list(self, stub_llm, captured_prompt):
        stub_llm('{"rewritten_query": "改写后的问题"}')

        result = step_3_extract_info("原问题", [])

        assert result == {"item_names": [], "rewritten_query": "改写后的问题"}

    def test_empty_item_names_is_tolerated(self, stub_llm, captured_prompt):
        """问题不含商品名（如「你好」）时应正常返回空列表，而不是报错。"""
        stub_llm('{"item_names": [], "rewritten_query": "你好"}')

        result = step_3_extract_info("你好", [])

        assert result == {"item_names": [], "rewritten_query": "你好"}

    def test_malformed_json_falls_back_to_original_query(self, stub_llm, captured_prompt):
        stub_llm("这不是 JSON")

        result = step_3_extract_info("原问题", [])

        assert result == {"item_names": [], "rewritten_query": "原问题"}

    def test_llm_exception_falls_back_to_original_query(self, stub_llm, captured_prompt):
        stub_llm(RuntimeError("LLM 超时"))

        result = step_3_extract_info("原问题", [])

        assert result == {"item_names": [], "rewritten_query": "原问题"}

    def test_prompt_load_failure_returns_early_without_calling_llm(self, stub_llm, monkeypatch):
        def _boom(*args, **kwargs):
            raise FileNotFoundError("prompts/xxx.prompt 缺失")

        monkeypatch.setattr(node_mod, "load_prompt", _boom)
        client = stub_llm('{"item_names": ["不该被调用"]}')

        result = step_3_extract_info("原问题", [])

        assert result == {"item_names": [], "rewritten_query": "原问题"}
        assert client.calls == []


# --------------------------------------------------------------------------------------
# step_4：向量化 + Milvus 混合检索
# --------------------------------------------------------------------------------------


class TestStep4VectorizeAndQuery:
    """混合检索：配置缺失短路、命中映射与检索参数。"""

    def test_missing_collection_env_returns_empty(self, fake_milvus, monkeypatch):
        monkeypatch.delenv("ITEM_NAME_COLLECTION", raising=False)

        assert step_4_vectorize_and_query(["HAK 180"]) == []

    def test_milvus_client_unavailable_returns_empty(self, fake_milvus, monkeypatch):
        monkeypatch.setenv("ITEM_NAME_COLLECTION", "item_name_collection")
        fake_milvus["client"] = None

        assert step_4_vectorize_and_query(["HAK 180"]) == []

    def test_hits_are_mapped_to_extracted_name_and_score(self, fake_milvus, monkeypatch):
        monkeypatch.setenv("ITEM_NAME_COLLECTION", "item_name_collection")
        fake_milvus["search_result"] = [
            [
                {"entity": {"item_name": "HAK 180 烫金机"}, "distance": 0.93},
                {"entity": {"item_name": "HAK 180 半自动"}, "distance": 0.71},
                {"entity": {}, "distance": 0.50},  # 缺 item_name 的命中应被丢弃
            ]
        ]

        result = step_4_vectorize_and_query(["HAK180"])

        assert result == [
            {
                "extracted_name": "HAK180",
                "matches": [
                    {"item_name": "HAK 180 烫金机", "score": 0.93},
                    {"item_name": "HAK 180 半自动", "score": 0.71},
                ],
            }
        ]

    def test_hybrid_search_uses_0_8_0_2_weights_and_limit_5(self, fake_milvus, monkeypatch):
        monkeypatch.setenv("ITEM_NAME_COLLECTION", "item_name_collection")
        fake_milvus["search_result"] = [[]]

        step_4_vectorize_and_query(["HAK 180"])

        assert fake_milvus["ranker_weights"] == (0.8, 0.2)
        assert fake_milvus["search_limit"] == 5
        assert fake_milvus["collections"] == ["item_name_collection"]


# --------------------------------------------------------------------------------------
# step_6：确认状态判定
# --------------------------------------------------------------------------------------


class TestStep6CheckConfirmation:
    """分支 A 确认 / 分支 B 反问 / 分支 C 拒绝。"""

    def test_confirmed_branch_sets_item_names_and_clears_old_answer(self, fake_mongo):
        state = {"item_names": [], "answer": "上一轮的旧答案"}
        align_result = {"confirmed_item_names": ["HAK 180 烫金机"], "options": []}

        result = step_6_check_confirmation(state, align_result, "s1", [], "改写后的问题")

        assert result["item_names"] == ["HAK 180 烫金机"]
        assert result["rewritten_query"] == "改写后的问题"
        assert "answer" not in result

    def test_confirmed_branch_only_updates_history_without_item_names(self, fake_mongo):
        """历史中已关联商品名的消息不应被覆盖，只补写未关联的那些。"""
        history = [
            {"_id": "m1", "item_names": []},
            {"_id": "m2", "item_names": ["已有商品"]},
            {"_id": None, "item_names": []},
        ]
        align_result = {"confirmed_item_names": ["HAK 180 烫金机"], "options": []}

        step_6_check_confirmation({}, align_result, "s1", history, "rw")

        assert fake_mongo["updated"] == [
            {"ids": ["m1"], "item_names": ["HAK 180 烫金机"]}
        ]

    def test_options_branch_asks_back_with_top3_candidates(self, fake_mongo):
        align_result = {"confirmed_item_names": [], "options": ["A 型", "B 型", "C 型", "D 型"]}

        result = step_6_check_confirmation({"item_names": ["旧值"]}, align_result, "s1", [], "rw")

        assert result["item_names"] == []
        assert result["answer"] == "您是想问以下哪个产品：A 型、B 型、C 型？请明确一下型号。"

    def test_no_match_branch_returns_rejection_message(self, fake_mongo):
        result = step_6_check_confirmation({}, {"confirmed_item_names": [], "options": []}, "s1", [], "rw")

        assert result["item_names"] == []
        assert result["answer"] == "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"

    def test_none_align_result_is_handled(self, fake_mongo):
        """align_result 为 None（上游异常）时应走拒绝分支而非抛异常。"""
        result = step_6_check_confirmation({}, None, "s1", [], "rw")

        assert result["item_names"] == []
        assert "未找到相关产品" in result["answer"]


# --------------------------------------------------------------------------------------
# step_7：会话历史落库
# --------------------------------------------------------------------------------------


class TestStep7WriteHistory:
    """写库时机：仅在有 answer 时补写助手消息；用户消息始终更新。"""

    def test_assistant_message_written_only_when_answer_present(self, fake_mongo):
        state = {
            "original_query": "这个多少钱？",
            "item_names": [],
            "answer": "您是想问以下哪个产品：A 型、B 型？请明确一下型号。",
        }

        result = step_7_write_history(state, "s1", [], "改写后的问题", "mid-9")

        assert result is state
        assert [call["role"] for call in fake_mongo["saved"]] == ["assistant", "user"]
        assert fake_mongo["saved"][0]["text"] == state["answer"]
        assert fake_mongo["saved"][0]["item_names"] == []
        assert fake_mongo["saved"][0]["rewritten_query"] == ""

    def test_user_message_carries_rewritten_query_and_item_names(self, fake_mongo):
        state = {"original_query": "这个多少钱？", "item_names": ["HAK 180 烫金机"]}

        step_7_write_history(state, "s1", [], "改写后的问题", "mid-9")

        assert len(fake_mongo["saved"]) == 1
        saved = fake_mongo["saved"][0]
        assert saved["session_id"] == "s1"
        assert saved["role"] == "user"
        assert saved["text"] == "这个多少钱？"
        assert saved["rewritten_query"] == "改写后的问题"
        assert saved["item_names"] == ["HAK 180 烫金机"]
        assert saved["message_id"] == "mid-9"


# --------------------------------------------------------------------------------------
# node 入口：全流程冒烟
# --------------------------------------------------------------------------------------


@pytest.fixture
def task_marker(monkeypatch):
    """记录任务状态标记的调用，替代真实 SSE 推送。"""
    markers = {"running": [], "done": []}

    monkeypatch.setattr(
        node_mod,
        "add_running_task",
        lambda session_id, node, is_stream: markers["running"].append((session_id, node, is_stream)),
    )
    monkeypatch.setattr(
        node_mod,
        "add_done_task",
        lambda session_id, node, is_stream: markers["done"].append((session_id, node, is_stream)),
    )
    return markers


class TestNodeItemNameConfirm:
    """入口节点的 state 流转冒烟。"""

    def test_flow_without_item_names_skips_retrieval(
        self, stub_llm, fake_mongo, fake_milvus, captured_prompt, task_marker
    ):
        """问题不含商品名（如「你好」）→ 跳过向量检索 → 走拒绝分支并落库。"""
        stub_llm('{"item_names": [], "rewritten_query": "你好"}')

        result = node_item_name_confirm(
            {"session_id": "s-noitem", "original_query": "你好", "is_stream": False, "item_names": []}
        )

        assert result["item_names"] == []
        assert "未找到相关产品" in result["answer"]
        assert result["rewritten_query"] == "你好"
        assert result["history"] == []
        # 向量检索整体被跳过
        assert fake_milvus["collections"] == []
        # 初始保存 + 助手回答 + 用户消息更新
        assert [call["role"] for call in fake_mongo["saved"]] == ["user", "assistant", "user"]
        # 初始保存走位置参数：rewritten_query 为空、item_names 取 state 现值
        assert fake_mongo["saved"][0]["rewritten_query"] == ""
        assert fake_mongo["saved"][0]["item_names"] == []
        assert task_marker["running"] == [("s-noitem", "node_item_name_confirm", False)]
        assert task_marker["done"] == [("s-noitem", "node_item_name_confirm", False)]

    def test_flow_with_item_names_confirms_and_stores_history(
        self, stub_llm, fake_mongo, fake_milvus, captured_prompt, task_marker, monkeypatch
    ):
        """单条高置信命中 → 直接确认商品名，不再反问。"""
        monkeypatch.setenv("ITEM_NAME_COLLECTION", "item_name_collection")
        history = [{"role": "user", "text": "HAK 180 多少钱", "_id": "m1", "item_names": []}]
        fake_mongo["history"] = history
        fake_milvus["search_result"] = [
            [{"entity": {"item_name": "HAK 180 烫金机"}, "distance": 0.93}]
        ]
        stub_llm('{"item_names": ["HAK 180"], "rewritten_query": "HAK 180 烫金机多少钱"}')

        result = node_item_name_confirm(
            {
                "session_id": "s-hit",
                "original_query": "这个多少钱？",
                "is_stream": True,
                "item_names": [],
            }
        )

        assert result["item_names"] == ["HAK 180 烫金机"]
        assert result["rewritten_query"] == "HAK 180 烫金机多少钱"
        assert "answer" not in result
        assert result["history"] == history
        # 命中历史消息被补写商品名
        assert fake_mongo["updated"] == [{"ids": ["m1"], "item_names": ["HAK 180 烫金机"]}]
        # 无 answer → 只保存初始用户消息与最终用户消息更新
        assert [call["role"] for call in fake_mongo["saved"]] == ["user", "user"]
        assert task_marker["running"] == [("s-hit", "node_item_name_confirm", True)]
        assert task_marker["done"] == [("s-hit", "node_item_name_confirm", True)]
