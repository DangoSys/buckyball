"""
Tailors算法实现
基于缓存约束的静态分块策略，专注于K维度的分块优化
"""

import numpy as np
from scipy.sparse import csr_matrix
import math


class TailorsAlgorithm:
    def __init__(self, cache_size=4 * 1024 * 1024, element_size=4):
        """
        初始化Tailors算法

        Args:
            cache_size: 缓存大小(字节)
            element_size: 每个元素的字节数
        """
        self.cache_size = cache_size
        self.element_size = element_size
        self.cache_capacity = cache_size // element_size

    def find_optimal_k_tiling(self, B_csc, J, K):
        """
        找到K维度的最优分块大小

        Args:
            B_csc: B矩阵的CSC格式
            J, K: 矩阵维度

        Returns:
            optimal_k_tile: 最优的K维度分块大小
        """
        # 计算基础边界pbound
        pbound = K
        left_bound = 0
        sum_now = 0

        for k in range(K):
            # 估算第k列的数据大小(非零元素数 * 3 + 1)
            col_size = len(B_csc.getcol(k).data) * 3 + 1
            sum_now += col_size

            # 滑动窗口确保不超过缓存容量
            while sum_now > self.cache_capacity and left_bound < k:
                pbound = min(pbound, k - left_bound)
                left_bound += 1
                left_col_size = len(B_csc.getcol(left_bound).data) * 3 + 1
                sum_now -= left_col_size

        # 使用倍数扩大分块大小
        multiples = [
            1.0,
            1.0625,
            1.125,
            1.25,
            1.5,
            2,
            3,
            5,
            9,
            17,
            33,
            65,
            129,
            257,
            513,
            1025,
        ]

        for mult in multiples[1:]:
            kkk = int(pbound * mult)
            if kkk >= K:
                break

            mis_tile_cnt = 0
            total_tile_cnt = 0

            # 检查每个分块的溢出情况
            for tk_start in range(0, K, kkk):
                tile_size = 0
                for kk in range(tk_start, min(tk_start + kkk, K)):
                    tile_size += len(B_csc.getcol(kk).data) * 3 + 1

                if tile_size > self.cache_capacity:
                    mis_tile_cnt += 1
                total_tile_cnt += 1

            # 如果溢出率超过10%，停止搜索
            if total_tile_cnt > 0 and mis_tile_cnt / total_tile_cnt > 0.1:
                break

            pbound = kkk

        return pbound

    def execute_tiling(self, A_csr, B_csc, optimal_k_tile):
        """
        执行Tailors分块矩阵乘法

        Args:
            A_csr: A矩阵的CSR格式
            B_csc: B矩阵的CSC格式
            optimal_k_tile: 最优K分块大小

        Returns:
            C: 结果矩阵
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape
        assert J == J_b, "矩阵维度不匹配"

        C = np.zeros((I, K))

        # IJK迭代顺序，只在K维度分块
        for i in range(I):
            for j in range(J):
                if A_csr[i, j] != 0:
                    # K维度分块处理
                    for k_start in range(0, K, optimal_k_tile):
                        k_end = min(k_start + optimal_k_tile, K)
                        for k in range(k_start, k_end):
                            if B_csc[j, k] != 0:
                                C[i, k] += A_csr[i, j] * B_csc[j, k]

        return C


def demo_tailors():
    """演示Tailors算法"""
    print("=== Tailors算法演示 ===")

    # 创建示例稀疏矩阵
    np.random.seed(42)
    I, J, K = 100, 150, 200

    # 生成稀疏矩阵A和B
    A_dense = np.random.random((I, J))
    A_dense[A_dense < 0.9] = 0  # 90%稀疏度
    A_csr = csr_matrix(A_dense)

    B_dense = np.random.random((J, K))
    B_dense[B_dense < 0.9] = 0  # 90%稀疏度
    B_csc = csr_matrix(B_dense).tocsc()

    # 初始化Tailors算法
    tailors = TailorsAlgorithm(cache_size=1024 * 1024)  # 1MB缓存

    # 找到最优K分块大小
    optimal_k = tailors.find_optimal_k_tiling(B_csc, J, K)
    print(f"最优K分块大小: {optimal_k}")

    # 执行分块矩阵乘法
    result = tailors.execute_tiling(A_csr, B_csc, optimal_k)
    print(f"结果矩阵形状: {result.shape}")
    print(f"结果矩阵非零元素数: {np.count_nonzero(result)}")

    # 验证正确性(与直接乘法比较)
    reference = A_csr.dot(B_csc).toarray()
    error = np.max(np.abs(result - reference))
    print(f"与参考结果的最大误差: {error}")


if __name__ == "__main__":
    demo_tailors()
