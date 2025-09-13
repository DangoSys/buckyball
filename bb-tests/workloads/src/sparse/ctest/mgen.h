#ifndef MGEN_H
#define MGEN_H

#include "buckyball.h"
#include <stdbool.h>

// CSC format for a single block
typedef struct {
  int *col_ptr;     // Column pointers (size: block_cols + 1)
  int *row_indices; // Row indices of non-zero elements
  elem_t *values;   // Values of non-zero elements
  int nnz;          // Number of non-zero elements in this block
  int rows;         // Block height
  int cols;         // Block width
} CSCBlock;

// BSR format with CSC blocks
typedef struct {
  CSCBlock *blocks;   // Array of CSC blocks
  int num_blocks;     // Total number of non-zero blocks
  int num_block_rows; // Number of block rows
  int num_block_cols; // Number of block columns
  int block_size;     // Block size (assuming square blocks)
  int matrix_rows;    // Original matrix rows
  int matrix_cols;    // Original matrix columns
} BSRMatrix;

// Function to generate a sparse matrix with specified dimensions and number of
// non-zero elements
void generate_sparse_matrix(elem_t *matrix, int rows, int cols,
                            int non_zero_count);

// Function to generate a dense matrix with specified dimensions
void generate_dense_matrix(elem_t *matrix, int rows, int cols);

// Function to compute expected result (CPU implementation)
void compute_expected_result(elem_t *A, elem_t *B, elem_t *C, int M, int K,
                             int N);

// Function to convert dense/sparse matrix to BSR format with CSC blocks
BSRMatrix *convert_to_bsr(elem_t *matrix, int rows, int cols, int block_size);

// Function to free BSR matrix memory
void free_bsr_matrix(BSRMatrix *bsr_matrix);

// Function to print BSR matrix (for debugging)
void print_bsr_matrix(const char *name, BSRMatrix *bsr_matrix);

// Utility function to print matrix (for debugging)
void print_matrix(const char *name, elem_t *matrix, int rows, int cols);

// Utility function to compare matrices
int compare_matrices(elem_t *a, elem_t *b, int rows, int cols);

#endif // MGEN_H
