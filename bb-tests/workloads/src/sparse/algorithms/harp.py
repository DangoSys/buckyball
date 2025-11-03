"""
Harp algorithm implementation
Single-dimension deep tiling strategy, focusing on I-dimension tiling while keeping J and K dimensions un-tiled
"""

import numpy as np
from scipy.sparse import csr_matrix
import math


class HarpAlgorithm:
    def __init__(self, cache_size=4 * 1024 * 1024, element_size=4):
        """
        Initialize Harp algorithm

        Args:
            cache_size: cache size (bytes)
            element_size: bytes per element
        """
        self.cache_size = cache_size
        self.element_size = element_size
        self.cache_capacity = cache_size // element_size

    def compute_i_tile_size(self, A_csr, base_tile_factor):
        """
        Compute I-dimension tile size

        Args:
            A_csr: A matrix in CSR format
            base_tile_factor: base tiling factor

        Returns:
            iii: I-dimension tile size
            tti: I-dimension tile count
        """
        I = A_csr.shape[0]

        # Harp algorithm: iii = I / tilesize
        iii = max(1, int(I / base_tile_factor))
        # Calculate tile count with ceiling division
        tti = (I + iii - 1) // iii

        return iii, tti

    def estimate_memory_usage(self, A_csr, B_csc, iii):
        """
        Estimate memory usage

        Args:
            A_csr: A matrix in CSR format
            B_csc: B matrix in CSC format
            iii: I-dimension tile size

        Returns:
            memory_usage: estimated memory usage
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape

        # Memory usage for A matrix tile
        avg_nnz_per_row = A_csr.nnz / I
        # CSR format storage
        a_tile_memory = iii * avg_nnz_per_row * 3

        # Memory usage for complete B matrix storage
        # CSC format storage
        b_memory = B_csc.nnz * 3

        # Memory usage for intermediate result C
        c_tile_memory = iii * K

        total_memory = a_tile_memory + b_memory + c_tile_memory
        return total_memory

    def optimize_i_tiling(self, A_csr, B_csc, base_tile_factor):
        """
        Optimize I-dimension tile size

        Args:
            A_csr: A matrix in CSR format
            B_csc: B matrix in CSC format
            base_tile_factor: base tiling factor

        Returns:
            optimal_iii: optimal I-dimension tile size
        """
        I = A_csr.shape[0]

        # Try different tile sizes
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

            # Compute performance score
            memory_usage = self.estimate_memory_usage(A_csr, B_csc, iii)

            # Scoring function: consider memory usage efficiency and computation locality
            if memory_usage <= self.cache_capacity:
                cache_efficiency = 1.0
            else:
                cache_efficiency = self.cache_capacity / memory_usage

            # Higher score when tile count is moderate
            tile_count = (I + iii - 1) // iii
            # Assume 8 parallel units
            parallelism_score = min(1.0, tile_count / 8.0)

            score = cache_efficiency * parallelism_score

            if score > best_score:
                best_score = score
                best_iii = iii

        return best_iii

    def execute_tiling(self, A_csr, B_csc, iii):
        """
        Execute Harp tiled matrix multiplication

        Args:
            A_csr: A matrix in CSR format
            B_csc: B matrix in CSC format
            iii: I-dimension tile size

        Returns:
            C: result matrix
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape
        assert J == J_b, "Matrix dimensions do not match"

        C = np.zeros((I, K))

        # JKI iteration order, only tile in I dimension
        # Pre-convert B matrix to dense format for better access efficiency
        B_dense = B_csc.toarray()

        for i_start in range(0, I, iii):
            i_end = min(i_start + iii, I)

            # Extract A matrix portion for current I tile
            A_tile = A_csr[i_start:i_end, :]

            # For current I tile, iterate over all J and K
            for j in range(J):
                # Only process non-zero columns
                if A_tile[:, j].nnz > 0:
                    A_col = A_tile[:, j].toarray().flatten()

                    for k in range(K):
                        if B_dense[j, k] != 0:
                            C[i_start:i_end, k] += A_col * B_dense[j, k]

        return C

    def analyze_sparsity_pattern(self, A_csr):
        """
        Analyze sparsity pattern to optimize tiling strategy

        Args:
            A_csr: A matrix in CSR format

        Returns:
            pattern_info: sparsity pattern information
        """
        I, J = A_csr.shape

        # Calculate non-zero element count per row
        nnz_per_row = np.array([len(A_csr.getrow(i).data) for i in range(I)])

        # Calculate row sparsity statistics
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
    """Demonstrate Harp algorithm"""
    print("=== Harp Algorithm Demo ===")

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

    # Initialize Harp algorithm
    # 1MB cache
    harp = HarpAlgorithm(cache_size=1024 * 1024)

    # Analyze A matrix sparsity pattern
    pattern_info = harp.analyze_sparsity_pattern(A_csr)
    print("A matrix sparsity pattern analysis:")
    for key, value in pattern_info.items():
        print(f"  {key}: {value:.2f}")

    # Compute base tiling factor (simplified version)
    base_tile_factor = max(2.0, pattern_info["avg_nnz_per_row"] / 10.0)
    print(f"Base tiling factor: {base_tile_factor:.2f}")

    # Optimize I-dimension tile size
    optimal_iii = harp.optimize_i_tiling(A_csr, B_csc, base_tile_factor)
    print(f"Optimal I tile size: {optimal_iii}")

    # Estimate memory usage
    memory_usage = harp.estimate_memory_usage(A_csr, B_csc, optimal_iii)
    print(f"Estimated memory usage: {memory_usage:.0f} elements")
    print(f"Cache utilization: {min(100.0, memory_usage/harp.cache_capacity*100):.1f}%")

    # Execute tiled matrix multiplication
    result = harp.execute_tiling(A_csr, B_csc, optimal_iii)
    print(f"Result matrix shape: {result.shape}")
    print(f"Result matrix non-zero elements: {np.count_nonzero(result)}")

    # Verify correctness
    reference = A_csr.dot(B_csc).toarray()
    error = np.max(np.abs(result - reference))
    print(f"Maximum error vs reference: {error}")


if __name__ == "__main__":
    demo_harp()
