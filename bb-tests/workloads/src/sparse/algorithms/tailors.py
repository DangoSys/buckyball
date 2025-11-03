"""
Tailors algorithm implementation
Static tiling strategy based on cache constraints, focusing on K-dimension tiling optimization
"""

import numpy as np
from scipy.sparse import csr_matrix
import math


class TailorsAlgorithm:
    def __init__(self, cache_size=4 * 1024 * 1024, element_size=4):
        """
        Initialize Tailors algorithm

        Args:
            cache_size: cache size (bytes)
            element_size: bytes per element
        """
        self.cache_size = cache_size
        self.element_size = element_size
        self.cache_capacity = cache_size // element_size

    def find_optimal_k_tiling(self, B_csc, J, K):
        """
        Find optimal K-dimension tile size

        Args:
            B_csc: B matrix in CSC format
            J, K: matrix dimensions

        Returns:
            optimal_k_tile: optimal K-dimension tile size
        # Compute base boundary pbound
        """
        pbound = K
        left_bound = 0
        sum_now = 0

        # Estimate data size of k-th column (non-zero elements * 3 + 1)
        for k in range(K):
            col_size = len(B_csc.getcol(k).data) * 3 + 1
            sum_now += col_size

            # Sliding window to ensure not exceeding cache capacity
            while sum_now > self.cache_capacity and left_bound < k:
                pbound = min(pbound, k - left_bound)
                left_bound += 1
                left_col_size = len(B_csc.getcol(left_bound).data) * 3 + 1
                sum_now -= left_col_size

        # Use multiples to expand tile size
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

            # Check overflow situation for each tile
            for tk_start in range(0, K, kkk):
                tile_size = 0
                for kk in range(tk_start, min(tk_start + kkk, K)):
                    tile_size += len(B_csc.getcol(kk).data) * 3 + 1

                if tile_size > self.cache_capacity:
                    mis_tile_cnt += 1
                total_tile_cnt += 1

            # If overflow rate exceeds 10%, stop searching
            if total_tile_cnt > 0 and mis_tile_cnt / total_tile_cnt > 0.1:
                break

            pbound = kkk

        return pbound

    def execute_tiling(self, A_csr, B_csc, optimal_k_tile):
        """
        Execute Tailors tiled matrix multiplication

        Args:
            A_csr: A matrix in CSR format
            B_csc: B matrix in CSC format
            optimal_k_tile: optimal K tile size

        Returns:
            C: result matrix
        """
        I, J = A_csr.shape
        J_b, K = B_csc.shape
        assert J == J_b, "Matrix dimensions do not match"

        C = np.zeros((I, K))

        # IJK iteration order, only tile in K dimension
        # K dimension tiling processing
        for i in range(I):
            for j in range(J):
                if A_csr[i, j] != 0:
                    for k_start in range(0, K, optimal_k_tile):
                        k_end = min(k_start + optimal_k_tile, K)
                        for k in range(k_start, k_end):
                            if B_csc[j, k] != 0:
                                C[i, k] += A_csr[i, j] * B_csc[j, k]

        return C


def demo_tailors():
    """Demonstrate Tailors algorithm"""
    print("=== Tailors Algorithm Demo ===")

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

    # Initialize Tailors algorithm
    # 1MB cache
    tailors = TailorsAlgorithm(cache_size=1024 * 1024)

    # Find optimal K tile size
    optimal_k = tailors.find_optimal_k_tiling(B_csc, J, K)
    print(f"Optimal K tile size: {optimal_k}")

    # Execute tiled matrix multiplication
    result = tailors.execute_tiling(A_csr, B_csc, optimal_k)
    print(f"Result matrix shape: {result.shape}")
    print(f"Result matrix non-zero elements: {np.count_nonzero(result)}")

    # Verify correctness (compare with direct multiplication)
    reference = A_csr.dot(B_csc).toarray()
    error = np.max(np.abs(result - reference))
    print(f"Maximum error vs reference: {error}")


if __name__ == "__main__":
    demo_tailors()
