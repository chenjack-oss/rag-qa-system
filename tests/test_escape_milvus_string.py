"""escape_milvus_string 单元测试：Milvus filter 表达式安全转义"""

from app.utils.escape_milvus_string_utils import escape_milvus_string


class TestEscapeBackslash:
    def test_single_backslash_doubled(self):
        assert escape_milvus_string(r"a\b") == r"a\\b"

    def test_multiple_backslashes(self):
        assert escape_milvus_string(r"C:\models\bge") == r"C:\\models\\bge"

    def test_no_backslash_unchanged(self):
        assert escape_milvus_string("normal") == "normal"


class TestEscapeDoubleQuote:
    def test_double_quote_escaped(self):
        assert escape_milvus_string('say "hi"') == 'say \\"hi\\"'

    def test_mixed_quote_and_backslash(self):
        # 先转义反斜杠，再转义引号
        assert escape_milvus_string(r'"a\"') == r'\"a\\\"'


class TestControlChars:
    def test_newline_replaced_by_space(self):
        assert escape_milvus_string("line1\nline2") == "line1 line2"

    def test_crlf_replaced_by_two_spaces(self):
        assert escape_milvus_string("a\r\nb") == "a  b"

    def test_tab_replaced_by_space(self):
        assert escape_milvus_string("a\tb") == "a b"


class TestEdgeCases:
    def test_none_returns_empty(self):
        assert escape_milvus_string(None) == ""

    def test_non_string_coerced(self):
        assert escape_milvus_string(12345) == "12345"

    def test_empty_string(self):
        assert escape_milvus_string("") == ""

    def test_chinese_content_unchanged(self):
        assert escape_milvus_string("掌柜问数") == "掌柜问数"
