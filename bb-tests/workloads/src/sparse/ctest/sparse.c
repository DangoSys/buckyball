#include "buckyball.h"
#include "inst.h"
#include "mgen.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DIM 16

// Scratchpad bank allocation
int32_t AspAddr = SPAD_ADDR(0, 0);
int32_t BspAddr = SPAD_ADDR(1, 0);
int32_t CspAddr = SPAD_ADDR(2, 0);

// Register file bank allocation
int32_t RowRfAddr = 0; // RF Bank 0: Row indices for sparse A
int32_t ColRfAddr = 1; // RF Bank 1: Column pointers for sparse A

// Add BSR block mapping structure
typedef struct {
  int block_row;
  int block_col;
  int bsr_index;
} BlockMapping;

int main() {
  // Matrix dimensions: A(M×K) × B(K×N) = C(M×N)
  int M = 64, K = 64, N = 64;   // Matrix dimensions
  int block_size = 16;          // Block size (should match DIM)
  int nnz = 256;                // Non-zero elements in A
  int val_min = 1, val_max = 3; // Value range

  // Derived parameters
  int M_blocks = (M + block_size - 1) / block_size;
  int K_blocks = (K + block_size - 1) / block_size;
  int N_blocks = (N + block_size - 1) / block_size;

  elem_t *A_dense = malloc(M * K * sizeof(elem_t));
  elem_t *B_dense = malloc(K * N * sizeof(elem_t));
  elem_t *C_result = calloc(M * N, sizeof(elem_t));
  elem_t *C_expected = calloc(M * N, sizeof(elem_t));

  if (!A_dense || !B_dense || !C_result || !C_expected) {
    printf("Memory allocation failed\n");
    return 1;
  }

  // Generate matrices
  generate_sparse_matrix(A_dense, M, K, nnz);
  generate_dense_matrix(B_dense, K, N);

  // Compute expected result using CPU
  compute_expected_result(A_dense, B_dense, C_expected, M, K, N);

  // Debug: count non-zero elements in A
  int nnz_count = 0;
  for (int i = 0; i < M * K; i++) {
    if (A_dense[i] != 0)
      nnz_count++;
  }
  printf("Matrix A (%dx%d) has %d non-zero elements\n", M, K, nnz_count);

  printf("Converting A to BSR format...\n");

  // Convert A to BSR format
  BSRMatrix *bsr_A = convert_to_bsr(A_dense, M, K, block_size);
  if (!bsr_A) {
    printf("BSR conversion failed\n");
    return 1;
  }

  printf("BSR conversion completed. %d blocks found.\n", bsr_A->num_blocks);

  // Debug: print first CSC block details
  if (bsr_A->num_blocks > 0) {
    CSCBlock *block0 = &bsr_A->blocks[0];
    printf("Debug - First CSC block (block 0):\n");
    printf("  nnz: %d, rows: %d, cols: %d\n", block0->nnz, block0->rows,
           block0->cols);
    printf("  col_ptr: ");
    for (int i = 0; i <= block0->cols && i < 8; i++) {
      printf("%d ", block0->col_ptr[i]);
    }
    printf("\n  row_indices: ");
    for (int i = 0; i < block0->nnz && i < 16; i++) {
      printf("%d ", block0->row_indices[i]);
    }
    printf("\n  values: ");
    for (int i = 0; i < block0->nnz && i < 16; i++) {
      printf("%d ", block0->values[i]);
    }
    printf("\n");
  }

  // Create BSR block mapping
  BlockMapping block_map[bsr_A->num_blocks];
  int block_idx = 0;

  // Map BSR blocks to (block_row, block_col) coordinates
  for (int br = 0; br < bsr_A->num_block_rows; br++) {
    for (int bc = 0; bc < bsr_A->num_block_cols; bc++) {
      // Check if this block exists in BSR format
      bool block_exists = false;

      // Check original matrix for non-zero elements in this block
      for (int i = 0; i < block_size && !block_exists; i++) {
        for (int j = 0; j < block_size && !block_exists; j++) {
          int row = br * block_size + i;
          int col = bc * block_size + j;
          if (row < M && col < K && A_dense[row * K + col] != 0) {
            block_exists = true;
          }
        }
      }

      if (block_exists && block_idx < bsr_A->num_blocks) {
        block_map[block_idx].block_row = br;
        block_map[block_idx].block_col = bc;
        block_map[block_idx].bsr_index = block_idx;
        printf("Block mapping: BSR[%d] -> matrix block (%d,%d)\n", block_idx,
               br, bc);
        block_idx++;
      }
    }
  }

  printf("Block mapping created for %d blocks\n", block_idx);

  // Initialize result matrix
  memset(C_result, 0, M * N * sizeof(elem_t));

  printf("Starting sparse matrix multiplication...\n");

  // Process sparse matrix multiplication using dedicated sparse instructions
  printf("Matrix configuration: A(%dx%d) = %dx%d BSR blocks, B(%dx%d) dense, "
         "C(%dx%d) = %dx%d result blocks\n",
         M, K, M_blocks, K_blocks, K, N, M, N, M_blocks, N_blocks);
  for (int c_block_row = 0; c_block_row < M_blocks;
       c_block_row++) { // C block rows
    printf("Processing C block row %d...\n", c_block_row);
    for (int c_block_col = 0; c_block_col < N_blocks;
         c_block_col++) { // C block cols
      printf("  Processing C block col %d...\n", c_block_col);

      // Sum over all K blocks (A block columns)
      for (int k_block = 0; k_block < K_blocks; k_block++) { // A block columns
        printf("  Processing k_block %d...\n", k_block);

        // Find corresponding BSR block for (c_block_row, k_block)
        CSCBlock *sparse_block = NULL;
        for (int b = 0; b < bsr_A->num_blocks; b++) {
          if (block_map[b].block_row == c_block_row &&
              block_map[b].block_col == k_block) {
            sparse_block = &bsr_A->blocks[b];
            printf("  Found sparse block %d with %d nnz\n", b,
                   sparse_block->nnz);
            break;
          }
        }

        if (!sparse_block || sparse_block->nnz == 0) {
          printf("  Skipping empty block\n");
          continue;
        }

        // Extract corresponding B block (k_block, c_block_col)
        elem_t B_block[block_size * block_size];
        for (int i = 0; i < block_size; i++) {
          for (int j = 0; j < block_size; j++) {
            int row = k_block * block_size + i;
            int col = c_block_col * block_size + j;
            if (row < K && col < N) {
              B_block[i * block_size + j] = B_dense[row * N + col];
            } else {
              B_block[i * block_size + j] = 0;
            }
          }
        }

        // Load sparse A values to scratchpad bank 0
        bb_mvin((uint64_t)sparse_block->values, AspAddr,
                (sparse_block->nnz + block_size - 1) /
                    block_size); // Round up to full rows

        // Load B matrix to scratchpad bank 1
        bb_mvin((uint64_t)B_block, BspAddr, block_size);

        // Load row indices to register file bank 0
        bb_scatter_mvin((uint64_t)sparse_block->row_indices, RowRfAddr,
                        sparse_block->nnz);

        // Load column pointers to register file bank 1
        bb_scatter_mvin((uint64_t)sparse_block->col_ptr, ColRfAddr,
                        sparse_block->cols + 1);

        // Execute sparse matrix multiplication using dedicated instruction
        bb_sparse_mul_addr(AspAddr, BspAddr, RowRfAddr, ColRfAddr, CspAddr,
                           sparse_block->nnz);

        // NPU instruction accumulates result directly in scratchpad
      }

      // Read final C_block result from scratchpad and copy to result
      elem_t C_block[block_size * block_size];
      bb_mvout((uint64_t)C_block, CspAddr, block_size);

      for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
          int result_row = c_block_row * block_size + i;
          int result_col = c_block_col * block_size + j;
          if (result_row < M && result_col < N) {
            C_result[result_row * N + result_col] = C_block[i * block_size + j];
          }
        }
      }
    }
  }

  // Verify results
  printf("Verifying results...\n");
  int errors = 0;
  int check_rows = (M < 4) ? M : 4;
  int check_cols = (N < 4) ? N : 4;
  for (int i = 0; i < check_rows; i++) {
    for (int j = 0; j < check_cols; j++) {
      if (C_result[i * N + j] != C_expected[i * N + j]) {
        printf("Mismatch at (%d,%d): got %d, expected %d\n", i, j,
               C_result[i * N + j], C_expected[i * N + j]);
        errors++;
        if (errors >= 10)
          break;
      }
    }
    if (errors >= 10)
      break;
  }

  if (errors == 0) {
    printf(
        "*** PASSED *** NPU-accelerated sparse matrix multiplication test\n");
  } else {
    printf("*** FAILED *** %d mismatches found\n", errors);

    // Print full comparison for first block
    printf("\nFirst %dx%d block of results:\n", check_rows, check_cols);
    for (int i = 0; i < check_rows; i++) {
      printf("Row %d: ", i);
      for (int j = 0; j < check_cols; j++) {
        printf("%3d ", C_result[i * N + j]);
      }
      printf("\n");
    }

    printf("\nFirst %dx%d block of expected:\n", check_rows, check_cols);
    for (int i = 0; i < check_rows; i++) {
      printf("Row %d: ", i);
      for (int j = 0; j < check_cols; j++) {
        printf("%3d ", C_expected[i * N + j]);
      }
      printf("\n");
    }
  }

  // Cleanup
  free_bsr_matrix(bsr_A);
  free(A_dense);
  free(B_dense);
  free(C_result);
  free(C_expected);

  return (errors == 0) ? 0 : 1;
}
