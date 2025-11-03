"""
DRT algorithm implementation
Dynamic Restructuring Tiling - balanced multi-dimension tiling strategy
"""

import numpy as np
from scipy.sparse import csr_matrix
import math


class DRTAlgorithm:
    def __init__(self, cache_size=4 * 1024 * 1024, element_size=4):
        """
        Initialize DRT algorithm

        Args:
            cache_size: cache size (bytes)
            element_size: bytes per element
        """
        self.cache_size = cache_size
        self.element_size = element_size
        self.cache_capacity = cache_size // element_size

    def compute_base_tile_size(self, B_csc, K):
        """
        Compute base tile size based on Tailors method

        Args:
            B_csc: B matrix in CSC format
            K: K dimension size

        Returns:
            base_tile_size: base tiling factor
        # Simplified pbound calculation
        """
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
        Compute tile sizes for J and K dimensions

        Args:
            J, K: matrix dimensions
            base_tile_size: base tiling factor

        Returns:
            jjj, kkk: tile sizes for J and K dimensions
        # DRT uses square root strategy to balance J and K dimensions
        """
        tt = math.sqrt(base_tile_size)

        jjj = max(1, int(J / tt))
        kkk = max(1, int(K / tt))

        return jjj, kkk

    def execute_tiling(self, A_csr, B_csc, jjj, kkk):
        """
        Execute DRT tiled matrix multiplication

        Args:
            A_csr: A matrix in CSR format
            B_csc: B matrix in CSC format
            jjj, kkk: tile sizes for J and K dimensions

        Returns:
            C: result matrix
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape
        assert J == J_b, "Matrix dimensions do not match"

        C = np.zeros((I, K))

        # JKI iteration order, tile in both J and K dimensions
        for j_start in range(0, J, jjj):
            j_end = min(j_start + jjj, J)

            for k_start in range(0, K, kkk):
                k_end = min(k_start + kkk, K)

                # Prefetch data for current J-K tile
                B_tile = B_csc[j_start:j_end, k_start:k_end].toarray()

                # Get data for row i of matrix A in current J tile range
                for i in range(I):
                    A_row = A_csr[i, j_start:j_end].toarray().flatten()

                    # Compute matrix multiplication for current tile
                    C_tile = np.dot(A_row, B_tile)
                    C[i, k_start:k_end] += C_tile

        return C

    def adaptive_tile_adjustment(self, A_csr, B_csc, initial_jjj, initial_kkk):
        """
        Adaptive tile size adjustment

        Args:
            A_csr, B_csc: input matrices
            initial_jjj, initial_kkk: initial tile sizes

        Returns:
            adjusted_jjj, adjusted_kkk: adjusted tile sizes
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape

        # Evaluate performance of different tile sizes
        best_jjj, best_kkk = initial_jjj, initial_kkk
        best_score = self._evaluate_tiling_efficiency(
            A_csr, B_csc, initial_jjj, initial_kkk
        )

        # Try adjusting tile sizes
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
        Evaluate tiling efficiency

        Args:
            A_csr, B_csc: input matrices
            jjj, kkk: tile sizes

        Returns:
            efficiency_score: efficiency score
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape

        # Compute data locality score within tiles
        locality_score = 0
        total_tiles = 0

        for j_start in range(0, J, jjj):
            j_end = min(j_start + jjj, J)
            for k_start in range(0, K, kkk):
                k_end = min(k_start + kkk, K)

                # Estimate non-zero element density within tile
                B_tile = B_csc[j_start:j_end, k_start:k_end]
                tile_density = B_tile.nnz / ((j_end - j_start) * (k_end - k_start))

                # Higher score when tile size is moderate and density is high
                size_score = 1.0 / (1.0 + abs(jjj * kkk - self.cache_capacity / 10))
                density_score = tile_density

                locality_score += size_score * density_score
                total_tiles += 1

        return locality_score / max(total_tiles, 1)


def demo_drt():
    """Demonstrate DRT algorithm"""
    print("=== DRT Algorithm Demo ===")

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

    # Initialize DRT algorithm
    # 1MB cache
    drt = DRTAlgorithm(cache_size=1024 * 1024)

    # Compute base tile size
    base_tile_size = drt.compute_base_tile_size(B_csc, K)
    print(f"Base tiling factor: {base_tile_size:.2f}")

    # Compute J and K dimension tile sizes
    initial_jjj, initial_kkk = drt.compute_jk_tiles(J, K, base_tile_size)
    print(f"Initial J tile size: {initial_jjj}, K tile size: {initial_kkk}")

    # Adaptive tile size adjustment
    optimal_jjj, optimal_kkk = drt.adaptive_tile_adjustment(
        A_csr, B_csc, initial_jjj, initial_kkk
    )
    print(f"Optimized J tile size: {optimal_jjj}, K tile size: {optimal_kkk}")

    # Execute tiled matrix multiplication
    result = drt.execute_tiling(A_csr, B_csc, optimal_jjj, optimal_kkk)
    print(f"Result matrix shape: {result.shape}")
    print(f"Result matrix non-zero elements: {np.count_nonzero(result)}")

    # Verify correctness
    reference = A_csr.dot(B_csc).toarray()
    error = np.max(np.abs(result - reference))
    print(f"Maximum error vs reference: {error}")


if __name__ == "__main__":
    demo_drt()
