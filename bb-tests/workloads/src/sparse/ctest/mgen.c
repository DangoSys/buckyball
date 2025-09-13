#include "mgen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Generate sparse matrix with specified non-zero count
void generate_sparse_matrix(elem_t *matrix, int rows, int cols,
                            int non_zero_count) {
  // Initialize matrix to zero
  int min_val = 1;
  int max_val = 3;
  memset(matrix, 0, rows * cols * sizeof(elem_t));

  srand(time(NULL) + min_val); // Use different seeds

  int placed = 0;
  while (placed < non_zero_count) {
    int row = rand() % rows;
    int col = rand() % cols;

    if (matrix[row * cols + col] == 0) {
      matrix[row * cols + col] = min_val + (rand() % (max_val - min_val + 1));
      placed++;
    }
  }
}

// Generate dense matrix
void generate_dense_matrix(elem_t *matrix, int rows, int cols) {
  int min_val = 1;
  int max_val = 3;
  srand(time(NULL) + min_val + 1000); // Different seed

  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = min_val + (rand() % (max_val - min_val + 1));
  }
}

// Compute expected result (CPU implementation)
void compute_expected_result(elem_t *A, elem_t *B, elem_t *C, int M, int K,
                             int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int sum = 0;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// Convert dense matrix to BSR format with CSC blocks
BSRMatrix *convert_to_bsr(elem_t *matrix, int rows, int cols, int block_size) {
  BSRMatrix *bsr = malloc(sizeof(BSRMatrix));
  if (!bsr)
    return NULL;

  int block_rows = (rows + block_size - 1) / block_size;
  int block_cols = (cols + block_size - 1) / block_size;

  bsr->num_block_rows = block_rows;
  bsr->num_block_cols = block_cols;
  bsr->block_size = block_size;
  bsr->matrix_rows = rows;
  bsr->matrix_cols = cols;

  // First pass: count non-empty blocks
  int num_blocks = 0;
  for (int br = 0; br < block_rows; br++) {
    for (int bc = 0; bc < block_cols; bc++) {
      // Check if block is non-empty
      bool has_nonzero = false;
      for (int i = 0; i < block_size && !has_nonzero; i++) {
        for (int j = 0; j < block_size && !has_nonzero; j++) {
          int row = br * block_size + i;
          int col = bc * block_size + j;
          if (row < rows && col < cols && matrix[row * cols + col] != 0) {
            has_nonzero = true;
          }
        }
      }
      if (has_nonzero) {
        num_blocks++;
      }
    }
  }

  bsr->num_blocks = num_blocks;
  bsr->blocks = malloc(num_blocks * sizeof(CSCBlock));
  if (!bsr->blocks) {
    free(bsr);
    return NULL;
  }

  // Second pass: fill blocks
  int block_idx = 0;
  for (int br = 0; br < block_rows; br++) {
    for (int bc = 0; bc < block_cols; bc++) {
      // Extract block
      elem_t block_data[block_size * block_size];
      bool has_nonzero = false;

      for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
          int row = br * block_size + i;
          int col = bc * block_size + j;
          if (row < rows && col < cols) {
            block_data[i * block_size + j] = matrix[row * cols + col];
            if (matrix[row * cols + col] != 0) {
              has_nonzero = true;
            }
          } else {
            block_data[i * block_size + j] = 0;
          }
        }
      }

      if (has_nonzero) {
        // Convert block to CSC format
        CSCBlock *csc_block = &bsr->blocks[block_idx];
        csc_block->rows = block_size;
        csc_block->cols = block_size;

        // Count non-zeros and allocate
        int nnz = 0;
        for (int i = 0; i < block_size * block_size; i++) {
          if (block_data[i] != 0)
            nnz++;
        }

        csc_block->nnz = nnz;
        csc_block->col_ptr = malloc((block_size + 1) * sizeof(int));
        csc_block->row_indices = malloc(nnz * sizeof(int));
        csc_block->values = malloc(nnz * sizeof(elem_t));

        if (!csc_block->col_ptr || !csc_block->row_indices ||
            !csc_block->values) {
          // Cleanup on failure
          free_bsr_matrix(bsr);
          return NULL;
        }

        // Fill CSC data
        int val_idx = 0;
        csc_block->col_ptr[0] = 0;

        for (int col = 0; col < block_size; col++) {
          for (int row = 0; row < block_size; row++) {
            elem_t val = block_data[row * block_size + col];
            if (val != 0) {
              csc_block->row_indices[val_idx] = row;
              csc_block->values[val_idx] = val;
              val_idx++;
            }
          }
          csc_block->col_ptr[col + 1] = val_idx;
        }

        block_idx++;
      }
    }
  }

  return bsr;
}

// Free BSR matrix memory
void free_bsr_matrix(BSRMatrix *bsr) {
  if (!bsr)
    return;

  if (bsr->blocks) {
    for (int i = 0; i < bsr->num_blocks; i++) {
      free(bsr->blocks[i].col_ptr);
      free(bsr->blocks[i].row_indices);
      free(bsr->blocks[i].values);
    }
    free(bsr->blocks);
  }
  free(bsr);
}

// Print matrix for debugging
void print_matrix(const char *name, elem_t *matrix, int rows, int cols) {
  printf("%s (%dx%d):\n", name, rows, cols);
  for (int i = 0; i < rows && i < 8; i++) {
    printf("  ");
    for (int j = 0; j < cols && j < 8; j++) {
      printf("%3d ", matrix[i * cols + j]);
    }
    if (cols > 8)
      printf("...");
    printf("\n");
  }
  if (rows > 8)
    printf("  ...\n");
  printf("\n");
}

// Compare matrices
int compare_matrices(elem_t *a, elem_t *b, int rows, int cols) {
  int errors = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (a[i * cols + j] != b[i * cols + j]) {
        errors++;
      }
    }
  }
  return errors;
}

// Print BSR matrix for debugging
void print_bsr_matrix(const char *name, BSRMatrix *bsr) {
  printf("%s BSR Matrix:\n", name);
  printf("  Matrix size: %dx%d\n", bsr->matrix_rows, bsr->matrix_cols);
  printf("  Block size: %d\n", bsr->block_size);
  printf("  Block grid: %dx%d\n", bsr->num_block_rows, bsr->num_block_cols);
  printf("  Non-zero blocks: %d\n", bsr->num_blocks);

  for (int i = 0; i < bsr->num_blocks && i < 3; i++) {
    CSCBlock *block = &bsr->blocks[i];
    printf("  Block %d: %dx%d, nnz=%d\n", i, block->rows, block->cols,
           block->nnz);
    printf("  col_ptr: ");
    for (int j = 0; j <= block->cols && j < 8; j++) {
      printf("%d ", block->col_ptr[j]);
    }
    printf("\n  row_indices: ");
    for (int j = 0; j < block->nnz && j < 8; j++) {
      printf("%d ", block->row_indices[j]);
    }
    printf("\n  values: ");
    for (int j = 0; j < block->nnz && j < 8; j++) {
      printf("%d ", block->values[j]);
    }
    printf("\n");
  }
}
