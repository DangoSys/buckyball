"""
DRT算法实现
Dynamic Restructuring Tiling - 均衡的多维度分块策略
"""

import numpy as np
from scipy.sparse import csr_matrix
import math


class DRTAlgorithm:
    def __init__(self, cache_size=4 * 1024 * 1024, element_size=4):
        """
        初始化DRT算法

        Args:
            cache_size: 缓存大小(字节)
            element_size: 每个元素的字节数
        """
        self.cache_size = cache_size
        self.element_size = element_size
        self.cache_capacity = cache_size // element_size

    def compute_base_tile_size(self, B_csc, K):
        """
        基于Tailors方法计算基础分块大小

        Args:
            B_csc: B矩阵的CSC格式
            K: K维度大小

        Returns:
            base_tile_size: 基础分块因子
        """
        # 简化的pbound计算
        pbound = K
        left_bound = 0
        sum_now = 0

        for k in range(K):
            col_size = len(B_csc.getcol(k).data) * 3 + 1
            sum_now += col_size

            while sum_now > self.cache_capacity and left_bound < k:
                pbound = min(pbound, k - left_bound)
                left_bound += 1
                left_col_size = len(B_csc.getcol(left_bound).data) * 3 + 1
                sum_now -= left_col_size

        tile_size = K / pbound
        return tile_size

    def compute_jk_tiles(self, J, K, base_tile_size):
        """
        计算J和K维度的分块大小

        Args:
            J, K: 矩阵维度
            base_tile_size: 基础分块因子

        Returns:
            jjj, kkk: J和K维度的分块大小
        """
        # DRT使用平方根策略平衡J和K维度
        tt = math.sqrt(base_tile_size)

        jjj = max(1, int(J / tt))
        kkk = max(1, int(K / tt))

        return jjj, kkk

    def execute_tiling(self, A_csr, B_csc, jjj, kkk):
        """
        执行DRT分块矩阵乘法

        Args:
            A_csr: A矩阵的CSR格式
            B_csc: B矩阵的CSC格式
            jjj, kkk: J和K维度的分块大小

        Returns:
            C: 结果矩阵
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape
        assert J == J_b, "矩阵维度不匹配"

        C = np.zeros((I, K))

        # JKI迭代顺序，在J和K维度都分块
        for j_start in range(0, J, jjj):
            j_end = min(j_start + jjj, J)

            for k_start in range(0, K, kkk):
                k_end = min(k_start + kkk, K)

                # 预取当前J-K分块的数据
                B_tile = B_csc[j_start:j_end, k_start:k_end].toarray()

                for i in range(I):
                    # 获取A矩阵第i行在当前J分块范围内的数据
                    A_row = A_csr[i, j_start:j_end].toarray().flatten()

                    # 计算当前分块的矩阵乘法
                    C_tile = np.dot(A_row, B_tile)
                    C[i, k_start:k_end] += C_tile

        return C

    def adaptive_tile_adjustment(self, A_csr, B_csc, initial_jjj, initial_kkk):
        """
        自适应分块大小调整

        Args:
            A_csr, B_csc: 输入矩阵
            initial_jjj, initial_kkk: 初始分块大小

        Returns:
            adjusted_jjj, adjusted_kkk: 调整后的分块大小
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape

        # 评估不同分块大小的性能
        best_jjj, best_kkk = initial_jjj, initial_kkk
        best_score = self._evaluate_tiling_efficiency(
            A_csr, B_csc, initial_jjj, initial_kkk
        )

        # 尝试调整分块大小
        for j_factor in [0.5, 1.0, 2.0]:
            for k_factor in [0.5, 1.0, 2.0]:
                test_jjj = max(1, int(initial_jjj * j_factor))
                test_kkk = max(1, int(initial_kkk * k_factor))

                score = self._evaluate_tiling_efficiency(
                    A_csr, B_csc, test_jjj, test_kkk
                )

                if score > best_score:
                    best_score = score
                    best_jjj, best_kkk = test_jjj, test_kkk

        return best_jjj, best_kkk

    def _evaluate_tiling_efficiency(self, A_csr, B_csc, jjj, kkk):
        """
        评估分块效率

        Args:
            A_csr, B_csc: 输入矩阵
            jjj, kkk: 分块大小

        Returns:
            efficiency_score: 效率评分
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape

        # 计算分块内数据局部性得分
        locality_score = 0
        total_tiles = 0

        for j_start in range(0, J, jjj):
            j_end = min(j_start + jjj, J)
            for k_start in range(0, K, kkk):
                k_end = min(k_start + kkk, K)

                # 估算分块内的非零元素密度
                B_tile = B_csc[j_start:j_end, k_start:k_end]
                tile_density = B_tile.nnz / ((j_end - j_start) * (k_end - k_start))

                # 分块大小适中且密度较高时得分更高
                size_score = 1.0 / (1.0 + abs(jjj * kkk - self.cache_capacity / 10))
                density_score = tile_density

                locality_score += size_score * density_score
                total_tiles += 1

        return locality_score / max(total_tiles, 1)


def demo_drt():
    """演示DRT算法"""
    print("=== DRT算法演示 ===")

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

    # 初始化DRT算法
    drt = DRTAlgorithm(cache_size=1024 * 1024)  # 1MB缓存

    # 计算基础分块大小
    base_tile_size = drt.compute_base_tile_size(B_csc, K)
    print(f"基础分块因子: {base_tile_size:.2f}")

    # 计算J和K维度分块大小
    initial_jjj, initial_kkk = drt.compute_jk_tiles(J, K, base_tile_size)
    print(f"初始J分块大小: {initial_jjj}, K分块大小: {initial_kkk}")

    # 自适应调整分块大小
    optimal_jjj, optimal_kkk = drt.adaptive_tile_adjustment(
        A_csr, B_csc, initial_jjj, initial_kkk
    )
    print(f"优化后J分块大小: {optimal_jjj}, K分块大小: {optimal_kkk}")

    # 执行分块矩阵乘法
    result = drt.execute_tiling(A_csr, B_csc, optimal_jjj, optimal_kkk)
    print(f"结果矩阵形状: {result.shape}")
    print(f"结果矩阵非零元素数: {np.count_nonzero(result)}")

    # 验证正确性
    reference = A_csr.dot(B_csc).toarray()
    error = np.max(np.abs(result - reference))
    print(f"与参考结果的最大误差: {error}")


if __name__ == "__main__":
    demo_drt()
