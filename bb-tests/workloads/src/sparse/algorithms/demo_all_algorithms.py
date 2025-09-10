"""
演示所有四种稀疏矩阵分块算法的对比
"""

import numpy as np
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt

from tailors import TailorsAlgorithm
from drt import DRTAlgorithm
from harp import HarpAlgorithm
from hyte import HYTEAlgorithm


def generate_test_matrices(I, J, K, sparsity=0.9, seed=42):
    """
    生成测试用的稀疏矩阵

    Args:
        I, J, K: 矩阵维度
        sparsity: 稀疏度 (0-1)
        seed: 随机种子

    Returns:
        A_csr, B_csc: CSR和CSC格式的稀疏矩阵
    """
    np.random.seed(seed)

    # 生成稀疏矩阵A
    A_dense = np.random.random((I, J))
    A_dense[A_dense < sparsity] = 0
    A_csr = csr_matrix(A_dense)

    # 生成稀疏矩阵B
    B_dense = np.random.random((J, K))
    B_dense[B_dense < sparsity] = 0
    B_csc = csr_matrix(B_dense).tocsc()

    return A_csr, B_csc


def run_algorithm_comparison():
    """运行四种算法的性能对比"""

    print("=" * 60)
    print("稀疏矩阵分块算法性能对比")
    print("=" * 60)

    # 测试配置
    test_configs = [
        {"I": 100, "J": 150, "K": 200, "name": "小规模"},
        {"I": 200, "J": 300, "K": 400, "name": "中规模"},
        {"I": 400, "J": 500, "K": 600, "name": "大规模"},
    ]

    cache_size = 2 * 1024 * 1024  # 2MB缓存

    results = []

    for config in test_configs:
        I, J, K = config["I"], config["J"], config["K"]
        print(f"\n测试 {config['name']} 矩阵: {I}x{J} × {J}x{K}")
        print("-" * 50)

        # 生成测试矩阵
        A_csr, B_csc = generate_test_matrices(I, J, K, sparsity=0.9)
        print(f"矩阵A非零元素: {A_csr.nnz}, 矩阵B非零元素: {B_csc.nnz}")

        # 参考结果
        reference = A_csr.dot(B_csc).toarray()

        config_results = {"config": config, "algorithms": {}}

        # 1. Tailors算法
        print("\n1. 测试Tailors算法...")
        try:
            tailors = TailorsAlgorithm(cache_size=cache_size)

            start_time = time.time()
            optimal_k = tailors.find_optimal_k_tiling(B_csc, J, K)
            result_tailors = tailors.execute_tiling(A_csr, B_csc, optimal_k)
            exec_time = time.time() - start_time

            error = np.max(np.abs(result_tailors - reference))

            config_results["algorithms"]["Tailors"] = {
                "execution_time": exec_time,
                "error": error,
                "k_tile_size": optimal_k,
                "success": True,
            }

            print(f"   执行时间: {exec_time:.4f}s")
            print(f"   最大误差: {error:.2e}")
            print(f"   K分块大小: {optimal_k}")

        except Exception as e:
            print(f"   Tailors算法执行失败: {e}")
            config_results["algorithms"]["Tailors"] = {
                "success": False,
                "error": str(e),
            }

        # 2. DRT算法
        print("\n2. 测试DRT算法...")
        try:
            drt = DRTAlgorithm(cache_size=cache_size)

            start_time = time.time()
            base_tile_size = drt.compute_base_tile_size(B_csc, K)
            jjj, kkk = drt.compute_jk_tiles(J, K, base_tile_size)
            optimal_jjj, optimal_kkk = drt.adaptive_tile_adjustment(
                A_csr, B_csc, jjj, kkk
            )
            result_drt = drt.execute_tiling(A_csr, B_csc, optimal_jjj, optimal_kkk)
            exec_time = time.time() - start_time

            error = np.max(np.abs(result_drt - reference))

            config_results["algorithms"]["DRT"] = {
                "execution_time": exec_time,
                "error": error,
                "j_tile_size": optimal_jjj,
                "k_tile_size": optimal_kkk,
                "success": True,
            }

            print(f"   执行时间: {exec_time:.4f}s")
            print(f"   最大误差: {error:.2e}")
            print(f"   J分块大小: {optimal_jjj}, K分块大小: {optimal_kkk}")

        except Exception as e:
            print(f"   DRT算法执行失败: {e}")
            config_results["algorithms"]["DRT"] = {"success": False, "error": str(e)}

        # 3. Harp算法
        print("\n3. 测试Harp算法...")
        try:
            harp = HarpAlgorithm(cache_size=cache_size)

            start_time = time.time()
            pattern_info = harp.analyze_sparsity_pattern(A_csr)
            base_tile_factor = max(2.0, pattern_info["avg_nnz_per_row"] / 10.0)
            optimal_iii = harp.optimize_i_tiling(A_csr, B_csc, base_tile_factor)
            result_harp = harp.execute_tiling(A_csr, B_csc, optimal_iii)
            exec_time = time.time() - start_time

            error = np.max(np.abs(result_harp - reference))

            config_results["algorithms"]["Harp"] = {
                "execution_time": exec_time,
                "error": error,
                "i_tile_size": optimal_iii,
                "success": True,
            }

            print(f"   执行时间: {exec_time:.4f}s")
            print(f"   最大误差: {error:.2e}")
            print(f"   I分块大小: {optimal_iii}")

        except Exception as e:
            print(f"   Harp算法执行失败: {e}")
            config_results["algorithms"]["Harp"] = {"success": False, "error": str(e)}

        # 4. HYTE算法
        print("\n4. 测试HYTE算法...")
        try:
            hyte = HYTEAlgorithm(cache_size=cache_size, pe_count=16)

            start_time = time.time()
            hyte_result = hyte.run_hyte(A_csr, B_csc)
            exec_time = time.time() - start_time

            error = np.max(np.abs(hyte_result["matrix"] - reference))

            config_results["algorithms"]["HYTE"] = {
                "execution_time": exec_time,
                "error": error,
                "static_config": hyte_result["static_config"],
                "search_time": hyte_result["search_time"],
                "success": True,
            }

            print(f"   总执行时间: {exec_time:.4f}s")
            print(f"   静态搜索时间: {hyte_result['search_time']:.4f}s")
            print(f"   最大误差: {error:.2e}")
            print(f"   静态配置: {hyte_result['static_config']}")

        except Exception as e:
            print(f"   HYTE算法执行失败: {e}")
            config_results["algorithms"]["HYTE"] = {"success": False, "error": str(e)}

        results.append(config_results)

    # 生成性能对比报告
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)

    for result in results:
        config = result["config"]
        print(f"\n{config['name']} ({config['I']}×{config['J']}×{config['K']}):")
        print("-" * 40)

        algorithms = result["algorithms"]

        # 按执行时间排序
        successful_algos = [
            (name, data)
            for name, data in algorithms.items()
            if data.get("success", False)
        ]
        successful_algos.sort(key=lambda x: x[1]["execution_time"])

        print("执行时间排名:")
        for i, (name, data) in enumerate(successful_algos, 1):
            print(
                f"  {i}. {name}: {data['execution_time']:.4f}s (误差: {data['error']:.2e})"
            )

        print("\n分块策略:")
        for name, data in algorithms.items():
            if data.get("success", False):
                if name == "Tailors":
                    print(f"  {name}: K={data['k_tile_size']}")
                elif name == "DRT":
                    print(f"  {name}: J={data['j_tile_size']}, K={data['k_tile_size']}")
                elif name == "Harp":
                    print(f"  {name}: I={data['i_tile_size']}")
                elif name == "HYTE":
                    cfg = data["static_config"]
                    print(f"  {name}: I={cfg['iii']}, J={cfg['jjj']}, K={cfg['kkk']}")

    return results


if __name__ == "__main__":
    results = run_algorithm_comparison()
    print("\n算法对比完成!")
