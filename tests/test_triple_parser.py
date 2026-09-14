"""parse_triples 单元测试：LLM 三元组输出解析容错

覆盖 badcase：markdown 围栏 / 夹带说明文字 / 截断 / 包装格式 / 字段缺失 /
类型漂移 / 去重 / 截断上限 / 空输入。
"""

from app.utils.triple_parser import parse_triples

# 标准输出样例（其他用例的公共素材）
STD = '[{"head": "温度传感器", "head_type": "部件", "relation": "属于", "tail": "HAK 180 烫金机", "tail_type": "产品"}]'


class TestHappyPath:
    def test_plain_json_array(self):
        out = parse_triples(STD)
        assert len(out) == 1
        assert out[0]["head"] == "温度传感器"
        assert out[0]["relation"] == "属于"
        assert out[0]["tail"] == "HAK 180 烫金机"
        assert out[0]["head_type"] == "部件"
        assert out[0]["tail_type"] == "产品"

    def test_empty_array(self):
        assert parse_triples("[]") == []

    def test_missing_type_gets_default(self):
        out = parse_triples('[{"head": "a", "relation": "具有", "tail": "b"}]')
        assert out[0]["head_type"] == "实体"
        assert out[0]["tail_type"] == "实体"


class TestMarkdownFence:
    def test_json_fence_stripped(self):
        raw = "```json\n" + STD + "\n```"
        assert len(parse_triples(raw)) == 1

    def test_bare_fence_stripped(self):
        raw = "```\n" + STD + "\n```"
        assert len(parse_triples(raw)) == 1

    def test_fence_without_closing(self):
        # 截断导致结尾围栏丢失
        raw = "```json\n" + STD
        assert len(parse_triples(raw)) == 1


class TestSurroundingText:
    def test_leading_trailing_prose(self):
        raw = "以下是抽取结果：\n" + STD + "\n以上共1条。"
        assert len(parse_triples(raw)) == 1

    def test_truncated_array_keeps_complete_objects(self):
        # 最后一个对象被截断：应保留前两条完整的
        raw = (
            '[{"head": "a", "relation": "属于", "tail": "b"}, '
            '{"head": "c", "relation": "具有", "tail": "d"}, '
            '{"head": "e", "relation": "属于", "tail": "f", "tail_ty'
        )
        out = parse_triples(raw)
        assert len(out) == 2
        assert out[0]["head"] == "a"
        assert out[1]["head"] == "c"


class TestWrappedFormats:
    def test_triples_key_wrapper(self):
        raw = '{"triples": ' + STD + "}"
        assert len(parse_triples(raw)) == 1

    def test_single_object_not_array(self):
        raw = '{"head": "a", "relation": "属于", "tail": "b"}'
        out = parse_triples(raw)
        assert len(out) == 1
        assert out[0]["head"] == "a"

    def test_object_wrapper_without_known_key_treated_as_single(self):
        raw = '{"foo": "bar", "head": "a", "relation": "属于", "tail": "b"}'
        assert len(parse_triples(raw)) == 1


class TestFieldValidation:
    def test_missing_head_dropped(self):
        raw = '[{"relation": "属于", "tail": "b"}, {"head": "a", "relation": "属于", "tail": "b"}]'
        out = parse_triples(raw)
        assert len(out) == 1
        assert out[0]["head"] == "a"

    def test_empty_relation_dropped(self):
        raw = '[{"head": "a", "relation": "", "tail": "b"}]'
        assert parse_triples(raw) == []

    def test_null_tail_dropped(self):
        raw = '[{"head": "a", "relation": "属于", "tail": null}]'
        assert parse_triples(raw) == []

    def test_numeric_fields_coerced(self):
        # LLM 有时把参数值输出成数字
        raw = '[{"head": "额定电压", "relation": "具有", "tail": 220}]'
        out = parse_triples(raw)
        assert out[0]["tail"] == "220"

    def test_whitespace_stripped(self):
        raw = '[{"head": "  a  ", "relation": " 属于 ", "tail": "b"}]'
        out = parse_triples(raw)
        assert out[0]["head"] == "a"
        assert out[0]["relation"] == "属于"

    def test_non_dict_items_skipped(self):
        raw = '["oops", 42, {"head": "a", "relation": "属于", "tail": "b"}]'
        assert len(parse_triples(raw)) == 1


class TestDedupAndLimit:
    def test_duplicate_triples_merged(self):
        one = '{"head": "a", "relation": "属于", "tail": "b"}'
        raw = f"[{one}, {one}, {one}]"
        assert len(parse_triples(raw)) == 1

    def test_same_pair_different_relation_kept(self):
        raw = (
            '[{"head": "a", "relation": "属于", "tail": "b"}, '
            '{"head": "a", "relation": "具有", "tail": "b"}]'
        )
        assert len(parse_triples(raw)) == 2

    def test_max_triples_limit(self):
        raw = json_many(25)
        assert len(parse_triples(raw)) == 10

    def test_custom_max_triples(self):
        raw = json_many(25)
        assert len(parse_triples(raw, max_triples=3)) == 3


class TestGarbageInput:
    def test_empty_string(self):
        assert parse_triples("") == []
        assert parse_triples("   \n  ") == []

    def test_no_json_at_all(self):
        assert parse_triples("抱歉，我无法完成该任务。") == []

    def test_json_number_only(self):
        assert parse_triples("42") == []

    def test_none_input(self):
        assert parse_triples(None) == []


def json_many(n: int) -> str:
    """构造 n 条合法三元组的 JSON 数组字符串"""
    items = ", ".join(
        f'{{"head": "e{i}", "relation": "属于", "tail": "p{i}"}}' for i in range(n)
    )
    return f"[{items}]"
