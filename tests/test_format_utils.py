"""format_utils 单元测试：JSON 格式化"""

from app.utils.format_utils import format_json, format_state


class TestFormatJson:
    def test_dict_with_chinese_not_escaped(self):
        out = format_json({"name": "掌柜"})
        assert "掌柜" in out  # ensure_ascii=False 保留中文

    def test_indent_applied(self):
        out = format_json({"a": 1}, indent=2)
        assert out == '{\n  "a": 1\n}'

    def test_ensure_ascii_true_escapes(self):
        out = format_json({"name": "掌柜"}, ensure_ascii=True)
        assert "\\u" in out

    def test_list_supported(self):
        out = format_json([1, 2, 3])
        assert "1" in out and "3" in out


class TestFormatState:
    def test_default_indent_4(self):
        out = format_state({"task_id": "001"})
        assert out == '{\n    "task_id": "001"\n}'

    def test_custom_indent(self):
        out = format_state({"task_id": "001"}, indent=2)
        assert out == '{\n  "task_id": "001"\n}'
