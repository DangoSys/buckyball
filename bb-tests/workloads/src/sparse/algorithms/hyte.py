"""
HYTE算法实现
Hybrid Static-Dynamic Tiling - 混合静态动态分块策略
"""

import numpy as np
from scipy.sparse import csr_matrix
import math
import time
from collections import defaultdict


class HYTEAlgorithm:
    def __init__(self, cache_size=4 * 1024 * 1024, element_size=4, pe_count=32):
        """
        初始化HYTE算法

        Args:
            cache_size: 缓存大小(字节)
            element_size: 每个元素的字节数
            pe_count: 处理单元数量
        """
        self.cache_size = cache_size
        self.element_size = element_size
        self.cache_capacity = cache_size // element_size
        self.pe_count = pe_count

        # 动态调整参数
        self.tile_history = []
        self.performance_stats = defaultdict(list)

    def sample_matrix_parameters(self, A_csr, B_csc):
        """
        采样矩阵参数用于静态优化

        Args:
            A_csr: A矩阵的CSR格式
            B_csc: B矩阵的CSC格式

        Returns:
            params: 采样得到的矩阵参数
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape

        # 计算采样参数
        sample_k = int(math.sqrt(I))
        sample_p = 1.0 / sample_k

        # 分析矩阵稀疏度和分布
        a_density = A_csr.nnz / (I * J)
        b_density = B_csc.nnz / (J * K)

        # 分析行/列的非零元素分布
        a_nnz_per_row = np.array(
            [len(A_csr.getrow(i).data) for i in range(min(I, 100))]
        )
        b_nnz_per_col = np.array(
            [len(B_csc.getcol(k).data) for k in range(min(K, 100))]
        )

        params = {
            "I": I,
            "J": J,
            "K": K,
            "sample_k": sample_k,
            "sample_p": sample_p,
            "a_density": a_density,
            "b_density": b_density,
            "a_nnz_variance": np.var(a_nnz_per_row),
            "b_nnz_variance": np.var(b_nnz_per_col),
            "avg_a_nnz_per_row": np.mean(a_nnz_per_row),
            "avg_b_nnz_per_col": np.mean(b_nnz_per_col),
        }

        return params

    def compute_tile_bounds(self, params):
        """
        计算分块搜索边界

        Args:
            params: 矩阵参数

        Returns:
            bounds: 各维度的分块边界
        """
        I, J, K = params["I"], params["J"], params["K"]

        # 基于缓存容量和矩阵大小计算边界
        cache_elements = self.cache_capacity

        # K维度边界（基于B矩阵列的大小）
        avg_col_size = params["avg_b_nnz_per_col"] * 3 + 1
        k_bound = max(1, int(cache_elements / (avg_col_size * 4)))
        k_bound = min(k_bound, K // 2)

        # J维度边界
        j_bound = max(1, int(J / 8))

        # I维度边界
        i_bound = max(1, int(I / 8))

        bounds = {"i_bound": i_bound, "j_bound": j_bound, "k_bound": k_bound}

        return bounds

    def static_tile_search(self, A_csr, B_csc, params, bounds):
        """
        静态分块搜索

        Args:
            A_csr, B_csc: 输入矩阵
            params: 矩阵参数
            bounds: 搜索边界

        Returns:
            best_config: 最优分块配置
        """
        I, J, K = params["I"], params["J"], params["K"]
        i_bound, j_bound, k_bound = (
            bounds["i_bound"],
            bounds["j_bound"],
            bounds["k_bound"],
        )

        best_cost = float("inf")
        best_config = {"iii": I, "jjj": J, "kkk": K}

        # 分层搜索策略
        for iii in [I // (2**i) for i in range(0, int(math.log2(I // i_bound)) + 1)]:
            iii = max(i_bound, iii)

            # 只改变J
            for jjj in [
                J // (2**j) for j in range(0, int(math.log2(J // j_bound)) + 1)
            ]:
                jjj = max(j_bound, jjj)
                kkk = K

                cost = self._estimate_tile_cost(A_csr, B_csc, iii, jjj, kkk, params)
                if cost < best_cost:
                    best_cost = cost
                    best_config = {"iii": iii, "jjj": jjj, "kkk": kkk}

            # 只改变K
            jjj = J
            for kkk in [
                K // (2**k) for k in range(1, int(math.log2(K // k_bound)) + 1)
            ]:
                kkk = max(k_bound, kkk)

                cost = self._estimate_tile_cost(A_csr, B_csc, iii, jjj, kkk, params)
                if cost < best_cost:
                    best_cost = cost
                    best_config = {"iii": iii, "jjj": jjj, "kkk": kkk}

            # 同时改变J和K
            for kkk in [
                K // (2**k) for k in range(1, int(math.log2(K // k_bound)) + 1)
            ]:
                kkk = max(k_bound, kkk)
                for jjj in [
                    J // (2**j) for j in range(1, int(math.log2(J // j_bound)) + 1)
                ]:
                    jjj = max(j_bound, jjj)

                    cost = self._estimate_tile_cost(A_csr, B_csc, iii, jjj, kkk, params)
                    if cost < best_cost:
                        best_cost = cost
                        best_config = {"iii": iii, "jjj": jjj, "kkk": kkk}

        return best_config

    def _estimate_tile_cost(self, A_csr, B_csc, iii, jjj, kkk, params):
        """
        估算分块成本

        Args:
            A_csr, B_csc: 输入矩阵
            iii, jjj, kkk: 分块大小
            params: 矩阵参数

        Returns:
            cost: 估算成本
        """
        I, J, K = params["I"], params["J"], params["K"]

        # 计算分块数量
        ti = (I + iii - 1) // iii
        tj = (J + jjj - 1) // jjj
        tk = (K + kkk - 1) // kkk

        # 估算内存访问成本
        # A矩阵访问成本
        a_access_cost = ti * A_csr.nnz * 3

        # B矩阵访问成本
        b_tiles = tj * tk
        avg_b_tile_size = (B_csc.nnz / b_tiles) * 3
        b_access_cost = b_tiles * avg_b_tile_size

        # C矩阵写入成本
        c_write_cost = I * K * 3

        # 缓存溢出惩罚
        tile_memory = iii * jjj * 3 + jjj * kkk * 3 + iii * kkk * 3
        if tile_memory > self.cache_capacity:
            cache_penalty = (tile_memory / self.cache_capacity) ** 2
        else:
            cache_penalty = 1.0

        # 并行效率
        parallel_efficiency = min(1.0, (ti * tj * tk) / self.pe_count)

        total_cost = (
            (a_access_cost + b_access_cost + c_write_cost)
            * cache_penalty
            / parallel_efficiency
        )

        return total_cost

    def dynamic_tile_adjustment(self, current_config, tile_stats):
        """
        动态分块调整

        Args:
            current_config: 当前分块配置
            tile_stats: 分块统计信息

        Returns:
            new_config: 新的分块配置
        """
        iii, jjj, kkk = (
            current_config["iii"],
            current_config["jjj"],
            current_config["kkk"],
        )

        # 分析当前分块的性能
        cache_hit_rate = tile_stats.get("cache_hit_rate", 0.8)
        tile_utilization = tile_stats.get("tile_utilization", 0.7)

        new_config = current_config.copy()

        # 基于性能指标调整分块大小
        if cache_hit_rate < 0.6:  # 缓存命中率低，减小分块
            if jjj > 1:
                new_config["jjj"] = max(1, jjj // 2)
            elif kkk > 1:
                new_config["kkk"] = max(1, kkk // 2)
        elif cache_hit_rate > 0.9 and tile_utilization < 0.5:  # 可以增大分块
            if jjj * 2 <= self.cache_capacity**0.5:
                new_config["jjj"] = jjj * 2
            elif kkk * 2 <= self.cache_capacity**0.5:
                new_config["kkk"] = kkk * 2

        return new_config

    def execute_hybrid_tiling(self, A_csr, B_csc, initial_config):
        """
        执行混合分块矩阵乘法

        Args:
            A_csr: A矩阵的CSR格式
            B_csc: B矩阵的CSC格式
            initial_config: 初始分块配置

        Returns:
            C: 结果矩阵
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape
        assert J == J_b, "矩阵维度不匹配"

        C = np.zeros((I, K))
        current_config = initial_config.copy()

        # 执行分块矩阵乘法，周期性进行动态调整
        adjustment_interval = 10  # 每10个分块调整一次
        tile_count = 0

        iii, jjj, kkk = (
            current_config["iii"],
            current_config["jjj"],
            current_config["kkk"],
        )

        for i_start in range(0, I, iii):
            i_end = min(i_start + iii, I)

            for j_start in range(0, J, jjj):
                j_end = min(j_start + jjj, J)

                for k_start in range(0, K, kkk):
                    k_end = min(k_start + kkk, K)

                    # 执行当前分块的矩阵乘法
                    start_time = time.time()

                    A_tile = A_csr[i_start:i_end, j_start:j_end]
                    B_tile = B_csc[j_start:j_end, k_start:k_end].toarray()

                    C_tile = A_tile.dot(B_tile)
                    C[i_start:i_end, k_start:k_end] += C_tile

                    exec_time = time.time() - start_time

                    # 收集性能统计
                    tile_stats = {
                        "execution_time": exec_time,
                        "cache_hit_rate": np.random.uniform(
                            0.6, 0.95
                        ),  # 模拟缓存命中率
                        "tile_utilization": A_tile.nnz
                        / (A_tile.shape[0] * A_tile.shape[1]),
                    }

                    self.performance_stats["exec_times"].append(exec_time)
                    tile_count += 1

                    # 周期性动态调整
                    if tile_count % adjustment_interval == 0:
                        new_config = self.dynamic_tile_adjustment(
                            current_config, tile_stats
                        )
                        if new_config != current_config:
                            print(f"动态调整分块大小: {current_config} -> {new_config}")
                            current_config = new_config
                            iii, jjj, kkk = (
                                current_config["iii"],
                                current_config["jjj"],
                                current_config["kkk"],
                            )

        return C

    def run_hyte(self, A_csr, B_csc):
        """
        运行完整的HYTE算法

        Args:
            A_csr: A矩阵的CSR格式
            B_csc: B矩阵的CSC格式

        Returns:
            result: 计算结果和统计信息
        """
        print("开始HYTE算法...")

        # 1. 采样矩阵参数
        print("1. 采样矩阵参数...")
        params = self.sample_matrix_parameters(A_csr, B_csc)

        # 2. 计算搜索边界
        print("2. 计算分块搜索边界...")
        bounds = self.compute_tile_bounds(params)

        # 3. 静态分块搜索
        print("3. 执行静态分块搜索...")
        start_time = time.time()
        best_config = self.static_tile_search(A_csr, B_csc, params, bounds)
        search_time = time.time() - start_time

        print(f"静态优化完成，用时: {search_time:.3f}s")
        print(f"最优分块配置: {best_config}")

        # 4. 执行混合分块矩阵乘法
        print("4. 执行混合分块矩阵乘法...")
        result_matrix = self.execute_hybrid_tiling(A_csr, B_csc, best_config)

        result = {
            "matrix": result_matrix,
            "static_config": best_config,
            "search_time": search_time,
            "performance_stats": dict(self.performance_stats),
        }

        return result


def demo_hyte():
    """演示HYTE算法"""
    print("=== HYTE算法演示 ===")

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

    print(f"矩阵A: {A_csr.shape}, 非零元素: {A_csr.nnz}")
    print(f"矩阵B: {B_csc.shape}, 非零元素: {B_csc.nnz}")

    # 初始化并运行HYTE算法
    hyte = HYTEAlgorithm(cache_size=1024 * 1024, pe_count=16)  # 1MB缓存, 16个PE

    result = hyte.run_hyte(A_csr, B_csc)

    print(f"\n结果矩阵形状: {result['matrix'].shape}")
    print(f"结果矩阵非零元素数: {np.count_nonzero(result['matrix'])}")

    # 验证正确性
    reference = A_csr.dot(B_csc).toarray()
    error = np.max(np.abs(result["matrix"] - reference))
    print(f"与参考结果的最大误差: {error}")

    # 性能统计
    if result["performance_stats"]["exec_times"]:
        avg_tile_time = np.mean(result["performance_stats"]["exec_times"])
        print(f"平均分块执行时间: {avg_tile_time:.6f}s")


if __name__ == "__main__":
    demo_hyte()
