"""
HYTE algorithm implementation
Hybrid Static-Dynamic Tiling - hybrid static-dynamic tiling strategy
"""

import numpy as np
from scipy.sparse import csr_matrix
import math
import time
from collections import defaultdict


class HYTEAlgorithm:
    def __init__(self, cache_size=4 * 1024 * 1024, element_size=4, pe_count=32):
        """
        Initialize HYTE algorithm

        Args:
            cache_size: cache size (bytes)
            element_size: bytes per element
            pe_count: number of processing units
        """
        self.cache_size = cache_size
        self.element_size = element_size
        self.cache_capacity = cache_size // element_size
        self.pe_count = pe_count

        # Dynamic adjustment parameters
        self.tile_history = []
        self.performance_stats = defaultdict(list)

    def sample_matrix_parameters(self, A_csr, B_csc):
        """
        Sample matrix parameters for static optimization

        Args:
            A_csr: A matrix in CSR format
            B_csc: B matrix in CSC format

        Returns:
            params: sampled matrix parameters
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape

        # Compute sampling parameters
        sample_k = int(math.sqrt(I))
        sample_p = 1.0 / sample_k

        # Analyze matrix sparsity and distribution
        a_density = A_csr.nnz / (I * J)
        b_density = B_csc.nnz / (J * K)

        # Analyze non-zero element distribution of rows/columns
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
        Compute tile search bounds

        Args:
            params: matrix parameters

        Returns:
            bounds: tiling bounds for each dimension
        """
        I, J, K = params["I"], params["J"], params["K"]

        # Compute bounds based on cache capacity and matrix size
        cache_elements = self.cache_capacity

        # K dimension bound (based on B matrix column size)
        avg_col_size = params["avg_b_nnz_per_col"] * 3 + 1
        k_bound = max(1, int(cache_elements / (avg_col_size * 4)))
        k_bound = min(k_bound, K // 2)

        # J dimension bound
        j_bound = max(1, int(J / 8))

        # I dimension bound
        i_bound = max(1, int(I / 8))

        bounds = {"i_bound": i_bound, "j_bound": j_bound, "k_bound": k_bound}

        return bounds

    def static_tile_search(self, A_csr, B_csc, params, bounds):
        """
        Static tile search

        Args:
            A_csr, B_csc: input matrices
            params: matrix parameters
            bounds: search bounds

        Returns:
            best_config: optimal tiling configuration
        """
        I, J, K = params["I"], params["J"], params["K"]
        i_bound, j_bound, k_bound = (
            bounds["i_bound"],
            bounds["j_bound"],
            bounds["k_bound"],
        )

        best_cost = float("inf")
        best_config = {"iii": I, "jjj": J, "kkk": K}

        # Hierarchical search strategy
        for iii in [I // (2**i) for i in range(0, int(math.log2(I // i_bound)) + 1)]:
            iii = max(i_bound, iii)

            # Only change J
            for jjj in [
                J // (2**j) for j in range(0, int(math.log2(J // j_bound)) + 1)
            ]:
                jjj = max(j_bound, jjj)
                kkk = K

                cost = self._estimate_tile_cost(A_csr, B_csc, iii, jjj, kkk, params)
                if cost < best_cost:
                    best_cost = cost
                    best_config = {"iii": iii, "jjj": jjj, "kkk": kkk}

            # Only change K
            jjj = J
            for kkk in [
                K // (2**k) for k in range(1, int(math.log2(K // k_bound)) + 1)
            ]:
                kkk = max(k_bound, kkk)

                cost = self._estimate_tile_cost(A_csr, B_csc, iii, jjj, kkk, params)
                if cost < best_cost:
                    best_cost = cost
                    best_config = {"iii": iii, "jjj": jjj, "kkk": kkk}

            # Change both J and K
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
        Estimate tile cost

        Args:
            A_csr, B_csc: input matrices
            iii, jjj, kkk: tile sizes
            params: matrix parameters

        Returns:
            cost: estimated cost
        """
        I, J, K = params["I"], params["J"], params["K"]

        # Calculate tile count
        ti = (I + iii - 1) // iii
        tj = (J + jjj - 1) // jjj
        tk = (K + kkk - 1) // kkk

        # Estimate memory access cost
        # A matrix access cost
        a_access_cost = ti * A_csr.nnz * 3

        # B matrix access cost
        b_tiles = tj * tk
        avg_b_tile_size = (B_csc.nnz / b_tiles) * 3
        b_access_cost = b_tiles * avg_b_tile_size

        # C matrix write cost
        c_write_cost = I * K * 3

        # Cache overflow penalty
        tile_memory = iii * jjj * 3 + jjj * kkk * 3 + iii * kkk * 3
        if tile_memory > self.cache_capacity:
            cache_penalty = (tile_memory / self.cache_capacity) ** 2
        else:
            cache_penalty = 1.0

        # Parallel efficiency
        parallel_efficiency = min(1.0, (ti * tj * tk) / self.pe_count)

        total_cost = (
            (a_access_cost + b_access_cost + c_write_cost)
            * cache_penalty
            / parallel_efficiency
        )

        return total_cost

    def dynamic_tile_adjustment(self, current_config, tile_stats):
        """
        Dynamic tile adjustment

        Args:
            current_config: current tiling configuration
            tile_stats: tile statistics

        Returns:
            new_config: new tiling configuration
        """
        iii, jjj, kkk = (
            current_config["iii"],
            current_config["jjj"],
            current_config["kkk"],
        )

        # Analyze performance of current tiles
        cache_hit_rate = tile_stats.get("cache_hit_rate", 0.8)
        tile_utilization = tile_stats.get("tile_utilization", 0.7)

        new_config = current_config.copy()

        # Adjust tile size based on performance metrics
        # Low cache hit rate, reduce tile size
        if cache_hit_rate < 0.6:
            if jjj > 1:
                new_config["jjj"] = max(1, jjj // 2)
            elif kkk > 1:
                new_config["kkk"] = max(1, kkk // 2)
        elif cache_hit_rate > 0.9 and tile_utilization < 0.5:
            if jjj * 2 <= self.cache_capacity**0.5:
                new_config["jjj"] = jjj * 2
            elif kkk * 2 <= self.cache_capacity**0.5:
                new_config["kkk"] = kkk * 2

        return new_config

    def execute_hybrid_tiling(self, A_csr, B_csc, initial_config):
        """
        Execute hybrid tiled matrix multiplication

        Args:
            A_csr: A matrix in CSR format
            B_csc: B matrix in CSC format
            initial_config: initial tiling configuration

        Returns:
            C: result matrix
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape
        assert J == J_b, "Matrix dimensions do not match"

        C = np.zeros((I, K))
        current_config = initial_config.copy()

        # Execute tiled matrix multiplication, perform dynamic adjustment periodically
        # Adjust every 10 tiles
        adjustment_interval = 10
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

                    # Execute matrix multiplication for current tile
                    start_time = time.time()

                    A_tile = A_csr[i_start:i_end, j_start:j_end]
                    B_tile = B_csc[j_start:j_end, k_start:k_end].toarray()

                    C_tile = A_tile.dot(B_tile)
                    C[i_start:i_end, k_start:k_end] += C_tile

                    exec_time = time.time() - start_time

                    # Collect performance statistics
                    # Simulate cache hit rate
                    tile_stats = {
                        "execution_time": exec_time,
                        "cache_hit_rate": np.random.uniform(
                            0.6, 0.95
                        ),
                        "tile_utilization": A_tile.nnz
                        / (A_tile.shape[0] * A_tile.shape[1]),
                    }

                    self.performance_stats["exec_times"].append(exec_time)
                    tile_count += 1

                    # Periodic dynamic adjustment
                    if tile_count % adjustment_interval == 0:
                        new_config = self.dynamic_tile_adjustment(
                            current_config, tile_stats
                        )
                        if new_config != current_config:
                            print(f"Dynamic tile size adjustment: {current_config} -> {new_config}")
                            current_config = new_config
                            iii, jjj, kkk = (
                                current_config["iii"],
                                current_config["jjj"],
                                current_config["kkk"],
                            )

        return C

    def run_hyte(self, A_csr, B_csc):
        """
        Run complete HYTE algorithm

        Args:
            A_csr: A matrix in CSR format
            B_csc: B matrix in CSC format

        Returns:
            result: computation result and statistics
        """
        print("Starting HYTE algorithm...")

        # 1. Sample matrix parameters
        print("1. Sampling matrix parameters...")
        params = self.sample_matrix_parameters(A_csr, B_csc)

        # 2. Compute search bounds
        print("2. Computing tile search bounds...")
        bounds = self.compute_tile_bounds(params)

        # 3. Static tile search
        print("3. Executing static tile search...")
        start_time = time.time()
        best_config = self.static_tile_search(A_csr, B_csc, params, bounds)
        search_time = time.time() - start_time

        print(f"Static optimization completed, time taken: {search_time:.3f}s")
        print(f"Optimal tiling configuration: {best_config}")

        # 4. Execute hybrid tiled matrix multiplication
        print("4. Executing hybrid tiled matrix multiplication...")
        result_matrix = self.execute_hybrid_tiling(A_csr, B_csc, best_config)

        result = {
            "matrix": result_matrix,
            "static_config": best_config,
            "search_time": search_time,
            "performance_stats": dict(self.performance_stats),
        }

        return result


def demo_hyte():
    """Demonstrate HYTE algorithm"""
    print("=== HYTE Algorithm Demo ===")

    # Create example sparse matrices
    np.random.seed(42)
    I, J, K = 100, 150, 200

    # Generate sparse matrices A and B
    A_dense = np.random.random((I, J))
    # 90% sparsity
    A_dense[A_dense < 0.9] = 0
    A_csr = csr_matrix(A_dense)

    B_dense = np.random.random((J, K))
    # 90% sparsity
    B_dense[B_dense < 0.9] = 0
    B_csc = csr_matrix(B_dense).tocsc()

    print(f"Matrix A: {A_csr.shape}, non-zero elements: {A_csr.nnz}")
    print(f"Matrix B: {B_csc.shape}, non-zero elements: {B_csc.nnz}")

    # Initialize and run HYTE algorithm
    # 1MB cache, 16 PEs
    hyte = HYTEAlgorithm(cache_size=1024 * 1024, pe_count=16)

    result = hyte.run_hyte(A_csr, B_csc)

    print(f"\nResult matrix shape: {result['matrix'].shape}")
    print(f"Result matrix non-zero elements: {np.count_nonzero(result['matrix'])}")

    # Verify correctness
    reference = A_csr.dot(B_csc).toarray()
    error = np.max(np.abs(result["matrix"] - reference))
    print(f"Maximum error vs reference: {error}")

    # Performance statistics
    if result["performance_stats"]["exec_times"]:
        avg_tile_time = np.mean(result["performance_stats"]["exec_times"])
        print(f"Average tile execution time: {avg_tile_time:.6f}s")


if __name__ == "__main__":
    demo_hyte()
