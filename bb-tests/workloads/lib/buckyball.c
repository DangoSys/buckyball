#include "buckyball.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_u8_random_matrix(elem_t *matrix, int rows, int cols, int seed) {
  srand(seed);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = rand() % 128;
  }
}

void init_u32_random_matrix(result_t *matrix, int rows, int cols, int seed) {
  srand(seed);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = rand() % 256;
  }
}
int compare_u8_matrices(elem_t *a, elem_t *b, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    if (a[i] != b[i]) {
      printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
      return 0;
    }
  }
  return 1;
}
int compare_u32_matrices(result_t *a, result_t *b, int rows, int cols) {
  for (int i = 0; i <= rows * cols - 1; i++) {
    if (a[i] != b[i]) {
      printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
      return 0;
    }
  }
  return 1;
}

int compare_i8_matrices(elem_t *a, elem_t *b, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    if (a[i] != b[i]) {
      printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
      return 0;
    }
  }
  return 1;
}
int compare_i32_matrices(result_t *a, result_t *b, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    if (a[i] != b[i]) {
      printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
      return 0;
    }
  }
  return 1;
}

void clear_u32_matrix(result_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(result_t));
}
void clear_u8_matrix(elem_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(elem_t));
}
void clear_i8_matrix(elem_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(elem_t));
}

// Transpose matrix
void transpose_u8_matrix(elem_t *src, elem_t *dst, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      dst[j * rows + i] = src[i * cols + j];
    }
  }
}

result_t gemmini_in_shift(result_t v, int shift) {
  if (shift <= 0)
    return v;
  if (shift >= 32)
    return 0;

  uint32_t s = (uint32_t)shift;
  uint32_t bits = (uint32_t)v;
  uint32_t point_five = (bits >> (s - 1)) & 1;
  uint32_t zeros = (s <= 1) ? 0 : ((bits & ((1u << (s - 1)) - 1)) != 0);
  uint32_t ones_digit = (bits >> s) & 1;
  uint32_t r = point_five & (zeros | ones_digit);
  return (v >> s) + (result_t)r;
}

// CPU matrix multiplication (used to generate expected results)
void cpu_matmul(elem_t *a, elem_t *b, result_t *c, int rows, int cols,
                int inner) {
  clear_u32_matrix(c, rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      for (int k = 0; k < inner; k++) {
        c[i * cols + j] += a[i * inner + k] * b[k * cols + j];
      }
    }
  }
}
