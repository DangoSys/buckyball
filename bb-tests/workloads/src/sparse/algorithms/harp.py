"""
Harp算法实现
单维度深度分块策略，专注于I维度的分块而保持J和K维度不分块
"""

import numpy as np
from scipy.sparse import csr_matrix
import math


class HarpAlgorithm:
    def __init__(self, cache_size=4 * 1024 * 1024, element_size=4):
        """
        初始化Harp算法

        Args:
            cache_size: 缓存大小(字节)
            element_size: 每个元素的字节数
        """
        self.cache_size = cache_size
        self.element_size = element_size
        self.cache_capacity = cache_size // element_size

    def compute_i_tile_size(self, A_csr, base_tile_factor):
        """
        计算I维度的分块大小

        Args:
            A_csr: A矩阵的CSR格式
            base_tile_factor: 基础分块因子

        Returns:
            iii: I维度的分块大小
            tti: I维度的分块数量
        """
        I = A_csr.shape[0]

        # Harp算法：iii = I / tilesize
        iii = max(1, int(I / base_tile_factor))
        tti = (I + iii - 1) // iii  # 向上取整计算分块数量

        return iii, tti

    def estimate_memory_usage(self, A_csr, B_csc, iii):
        """
        估算内存使用量

        Args:
            A_csr: A矩阵的CSR格式
            B_csc: B矩阵的CSC格式
            iii: I维度分块大小

        Returns:
            memory_usage: 估算的内存使用量
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape

        # A矩阵分块的内存使用
        avg_nnz_per_row = A_csr.nnz / I
        a_tile_memory = iii * avg_nnz_per_row * 3  # CSR格式存储

        # B矩阵完整存储的内存使用
        b_memory = B_csc.nnz * 3  # CSC格式存储

        # 中间结果C的内存使用
        c_tile_memory = iii * K

        total_memory = a_tile_memory + b_memory + c_tile_memory
        return total_memory

    def optimize_i_tiling(self, A_csr, B_csc, base_tile_factor):
        """
        优化I维度分块大小

        Args:
            A_csr: A矩阵的CSR格式
            B_csc: B矩阵的CSC格式
            base_tile_factor: 基础分块因子

        Returns:
            optimal_iii: 最优I维度分块大小
        """
        I = A_csr.shape[0]

        # 尝试不同的分块大小
        best_iii = max(1, int(I / base_tile_factor))
        best_score = float("-inf")

        candidates = [
            int(I / (base_tile_factor * 0.5)),
            int(I / base_tile_factor),
            int(I / (base_tile_factor * 2.0)),
            int(I / (base_tile_factor * 4.0)),
        ]

        for iii in candidates:
            if iii <= 0 or iii > I:
                continue

            # 计算性能评分
            memory_usage = self.estimate_memory_usage(A_csr, B_csc, iii)

            # 评分函数：考虑内存使用效率和计算局部性
            if memory_usage <= self.cache_capacity:
                cache_efficiency = 1.0
            else:
                cache_efficiency = self.cache_capacity / memory_usage

            # 分块数量适中时得分更高
            tile_count = (I + iii - 1) // iii
            parallelism_score = min(1.0, tile_count / 8.0)  # 假设8个并行单元

            score = cache_efficiency * parallelism_score

            if score > best_score:
                best_score = score
                best_iii = iii

        return best_iii

    def execute_tiling(self, A_csr, B_csc, iii):
        """
        执行Harp分块矩阵乘法

        Args:
            A_csr: A矩阵的CSR格式
            B_csc: B矩阵的CSC格式
            iii: I维度分块大小

        Returns:
            C: 结果矩阵
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape
        assert J == J_b, "矩阵维度不匹配"

        C = np.zeros((I, K))

        # JKI迭代顺序，只在I维度分块
        # 预先转换B矩阵为密集格式以提高访问效率
        B_dense = B_csc.toarray()

        for i_start in range(0, I, iii):
            i_end = min(i_start + iii, I)

            # 提取当前I分块的A矩阵部分
            A_tile = A_csr[i_start:i_end, :]

            # 对于当前I分块，遍历所有J和K
            for j in range(J):
                if A_tile[:, j].nnz > 0:  # 只处理非零列
                    A_col = A_tile[:, j].toarray().flatten()

                    for k in range(K):
                        if B_dense[j, k] != 0:
                            C[i_start:i_end, k] += A_col * B_dense[j, k]

        return C

    def analyze_sparsity_pattern(self, A_csr):
        """
        分析稀疏模式以优化分块策略

        Args:
            A_csr: A矩阵的CSR格式

        Returns:
            pattern_info: 稀疏模式信息
        """
        I, J = A_csr.shape

        # 计算每行的非零元素数量
        nnz_per_row = np.array([len(A_csr.getrow(i).data) for i in range(I)])

        # 计算行稀疏度的统计信息
        row_density = nnz_per_row / J

        pattern_info = {
            "avg_nnz_per_row": np.mean(nnz_per_row),
            "std_nnz_per_row": np.std(nnz_per_row),
            "max_nnz_per_row": np.max(nnz_per_row),
            "min_nnz_per_row": np.min(nnz_per_row),
            "avg_row_density": np.mean(row_density),
            "density_variance": np.var(row_density),
        }

        return pattern_info


def demo_harp():
    """演示Harp算法"""
    print("=== Harp算法演示 ===")

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

    # 初始化Harp算法
    harp = HarpAlgorithm(cache_size=1024 * 1024)  # 1MB缓存

    # 分析A矩阵的稀疏模式
    pattern_info = harp.analyze_sparsity_pattern(A_csr)
    print("A矩阵稀疏模式分析:")
    for key, value in pattern_info.items():
        print(f"  {key}: {value:.2f}")

    # 计算基础分块因子（简化版本）
    base_tile_factor = max(2.0, pattern_info["avg_nnz_per_row"] / 10.0)
    print(f"基础分块因子: {base_tile_factor:.2f}")

    # 优化I维度分块大小
    optimal_iii = harp.optimize_i_tiling(A_csr, B_csc, base_tile_factor)
    print(f"最优I分块大小: {optimal_iii}")

    # 估算内存使用
    memory_usage = harp.estimate_memory_usage(A_csr, B_csc, optimal_iii)
    print(f"估算内存使用: {memory_usage:.0f} 元素")
    print(f"缓存利用率: {min(100.0, memory_usage/harp.cache_capacity*100):.1f}%")

    # 执行分块矩阵乘法
    result = harp.execute_tiling(A_csr, B_csc, optimal_iii)
    print(f"结果矩阵形状: {result.shape}")
    print(f"结果矩阵非零元素数: {np.count_nonzero(result)}")

    # 验证正确性
    reference = A_csr.dot(B_csc).toarray()
    error = np.max(np.abs(result - reference))
    print(f"与参考结果的最大误差: {error}")


if __name__ == "__main__":
    demo_harp()
