"""
Demonstrate comparison of all four sparse matrix tiling algorithms
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
    Generate sparse matrices for testing

    Args:
        I, J, K: matrix dimensions
        sparsity: sparsity (0-1)
        seed: random seed

    Returns:
        A_csr, B_csc: sparse matrices in CSR and CSC formats
    """
    np.random.seed(seed)

    # Generate sparse matrix A
    A_dense = np.random.random((I, J))
    A_dense[A_dense < sparsity] = 0
    A_csr = csr_matrix(A_dense)

    # Generate sparse matrix B
    B_dense = np.random.random((J, K))
    B_dense[B_dense < sparsity] = 0
    B_csc = csr_matrix(B_dense).tocsc()

    return A_csr, B_csc


def run_algorithm_comparison():
    """Run performance comparison of four algorithms"""

    print("=" * 60)
    print("Sparse Matrix Tiling Algorithm Performance Comparison")
    print("=" * 60)

    # Test configurations
    test_configs = [
        {"I": 100, "J": 150, "K": 200, "name": "Small scale"},
        {"I": 200, "J": 300, "K": 400, "name": "Medium scale"},
        {"I": 400, "J": 500, "K": 600, "name": "Large scale"},
    ]

    # 2MB cache
    cache_size = 2 * 1024 * 1024

    results = []

    for config in test_configs:
        I, J, K = config["I"], config["J"], config["K"]
        print(f"\nTest {config['name']} matrix: {I}x{J} × {J}x{K}")
        print("-" * 50)

        # Generate test matrices
        A_csr, B_csc = generate_test_matrices(I, J, K, sparsity=0.9)
        print(f"Matrix A non-zero elements: {A_csr.nnz}, Matrix B non-zero elements: {B_csc.nnz}")

        # Reference result
        reference = A_csr.dot(B_csc).toarray()

        config_results = {"config": config, "algorithms": {}}

        # 1. Tailors algorithm
        print("\n1. Test Tailors algorithm...")
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

            print(f"   Execution time: {exec_time:.4f}s")
            print(f"   Maximum error: {error:.2e}")
            print(f"   K tile size: {optimal_k}")

        except Exception as e:
            print(f"   Tailors algorithm execution failed: {e}")
            config_results["algorithms"]["Tailors"] = {
                "success": False,
                "error": str(e),
            }

        # 2. DRT algorithm
        print("\n2. Test DRT algorithm...")
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

            print(f"   Execution time: {exec_time:.4f}s")
            print(f"   Maximum error: {error:.2e}")
            print(f"   J tile size: {optimal_jjj}, K tile size: {optimal_kkk}")

        except Exception as e:
            print(f"   DRT algorithm execution failed: {e}")
            config_results["algorithms"]["DRT"] = {"success": False, "error": str(e)}

        # 3. Harp algorithm
        print("\n3. Test Harp algorithm...")
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

            print(f"   Execution time: {exec_time:.4f}s")
            print(f"   Maximum error: {error:.2e}")
            print(f"   I tile size: {optimal_iii}")

        except Exception as e:
            print(f"   Harp algorithm execution failed: {e}")
            config_results["algorithms"]["Harp"] = {"success": False, "error": str(e)}

        # 4. HYTE algorithm
        print("\n4. Test HYTE algorithm...")
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

            print(f"   Total execution time: {exec_time:.4f}s")
            print(f"   Static search time: {hyte_result['search_time']:.4f}s")
            print(f"   Maximum error: {error:.2e}")
            print(f"   Static config: {hyte_result['static_config']}")

        except Exception as e:
            print(f"   HYTE algorithm execution failed: {e}")
            config_results["algorithms"]["HYTE"] = {"success": False, "error": str(e)}

        results.append(config_results)

    # Generate performance comparison report
    print("\n" + "=" * 60)
    print("Performance Comparison Summary")
    print("=" * 60)

    for result in results:
        config = result["config"]
        print(f"\n{config['name']} ({config['I']}×{config['J']}×{config['K']}):")
        print("-" * 40)

        algorithms = result["algorithms"]

        # Sort by execution time
        successful_algos = [
            (name, data)
            for name, data in algorithms.items()
            if data.get("success", False)
        ]
        successful_algos.sort(key=lambda x: x[1]["execution_time"])

        print("Execution time ranking:")
        for i, (name, data) in enumerate(successful_algos, 1):
            print(
                f"  {i}. {name}: {data['execution_time']:.4f}s (error: {data['error']:.2e})"
            )

        print("\nTiling strategy:")
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
    print("\nAlgorithm comparison completed!")
