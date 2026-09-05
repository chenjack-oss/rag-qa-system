"""normalize_sparse_vector 单元测试：稀疏向量 L2 归一化"""

import numpy as np

from app.utils.normalize_sparse_vector import normalize_sparse_vector


class TestBasicNormalize:
    def test_simple_vector_normalized(self):
        vec = {0: 3.0, 1: 4.0}  # L2 = 5
        out = normalize_sparse_vector(vec)
        assert np.isclose(out[0], 0.6)
        assert np.isclose(out[1], 0.8)

    def test_l2_norm_is_one(self):
        vec = {10: 1.0, 20: 2.0, 30: 3.0}
        out = normalize_sparse_vector(vec)
        norm = np.linalg.norm(list(out.values()))
        assert np.isclose(norm, 1.0)

    def test_keys_preserved(self):
        vec = {100: 5.0, 200: 5.0, 300: 5.0}
        out = normalize_sparse_vector(vec)
        assert set(out.keys()) == {100, 200, 300}


class TestEdgeCases:
    def test_empty_dict_returns_empty(self):
        assert normalize_sparse_vector({}) == {}

    def test_zero_vector_returned_as_is(self):
        vec = {0: 0.0, 1: 0.0}
        out = normalize_sparse_vector(vec)
        assert out == vec  # 范数为 0，避免除零，原样返回

    def test_near_zero_vector_returned_as_is(self):
        vec = {0: 1e-12}
        out = normalize_sparse_vector(vec)
        assert out == vec

    def test_single_dimension(self):
        out = normalize_sparse_vector({7: -2.0})
        assert np.isclose(out[7], -1.0)  # 保留符号

    def test_negative_values(self):
        out = normalize_sparse_vector({0: -3.0, 1: 4.0})
        assert np.isclose(out[0], -0.6)
        assert np.isclose(out[1], 0.8)
