#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read cycle counter (rdcycle) helper. Works on RV64 with a single rdcycle.
   On RV32 we read low/high and detect rollover to produce a 64-bit value. */
unsigned long long read_rdcycle(void) {
#if defined(__riscv_xlen) && __riscv_xlen == 64
  unsigned long long cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
#else
  unsigned int lo1, hi, lo2;
  /* Loop until two consecutive low reads are equal to avoid rollover window */
  asm volatile("1: rdcycle %0\n"
               "   rdcycleh %1\n"
               "   rdcycle %2\n"
               "   bne %0, %2, 1b\n"
               : "=&r"(lo1), "=&r"(hi), "=&r"(lo2));
  return ((unsigned long long)hi << 32) | lo1;
#endif
}

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
      // print_matrix("Expected", b, 1, cols);
      // print_matrix("Actual", a, 1, cols);
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

void init_i8_random_matrix(elem_t *matrix, int rows, int cols, int seed) {
  srand(seed);
  for (int i = 0; i < rows * cols; i++) {
    /* produce values in range -128 .. 127 */
    matrix[i] = (elem_t)((rand() % 256) - 128);
  }
}
void init_i32_random_matrix(result_t *matrix, int rows, int cols, int seed) {
  srand(seed);
  for (int i = 0; i < rows * cols; i++) {
    /* produce values in a reasonable 16-bit signed range -32768 .. 32767 */
    matrix[i] = (result_t)((rand() % 65536) - 32768);
  }
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

void print_u32_matrix(const char *name, result_t *matrix, int rows, int cols) {
  printf("Matrix %s:\n", name);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%4d ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}
void print_u8_matrix(const char *name, elem_t *matrix, int rows, int cols) {
  printf("Matrix %s:\n", name);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%4d ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}
// Signed print variants
void print_i32_matrix(const char *name, result_t *matrix, int rows, int cols) {
  printf("Matrix %s:\n", name);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%4d ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void print_i8_matrix(const char *name, elem_t *matrix, int rows, int cols) {
  printf("Matrix %s:\n", name);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      /* cast to int to avoid printing as char */
      printf("%4d ", (int)matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}
void clear_u32_matrix(result_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(result_t));
}
void clear_u8_matrix(elem_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(elem_t));
}
void clear_i32_matrix(result_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(result_t));
}
void clear_i8_matrix(elem_t *matrix, int rows, int cols) {
  memset(matrix, 0, rows * cols * sizeof(elem_t));
}

void init_ones_matrix(elem_t *matrix, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = 1;
  }
}

void init_identity_matrix(elem_t *matrix, int size) {
  clear_u8_matrix(matrix, size, size);
  for (int i = 0; i < size; i++) {
    matrix[i * size + i] = 1;
  }
}

void init_row_vector(elem_t *matrix, int cols, elem_t value) {
  clear_u8_matrix(matrix, DIM, DIM);
  for (int j = 0; j < cols; j++) {
    matrix[j] = value;
  }
}

void init_col_vector(elem_t *matrix, int rows, elem_t value) {
  clear_u8_matrix(matrix, DIM, DIM);
  for (int i = 0; i < rows; i++) {
    matrix[i * DIM] = value;
  }
}

void init_random_matrix(elem_t *matrix, int rows, int cols, int seed) {
  srand(seed);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i * cols + j] = (rand() % 5); // 0-4的随机数
    }
  }
}

void init_bbfp_random_matrix(elem_t *matrix, int rows, int cols, int seed) {
  srand(seed);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i * cols + j] = (rand() % 16);
    }
  }
}

void init_sequence_matrix(elem_t *matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i * cols + j] = i + j;
    }
  }
}
// 初始化列对齐的随机矩阵和原始矩阵
void init_col_aligned_random_matrix(elem_t *aligned_matrix, elem_t *matrix,
                                    int align, int rows, int cols, int seed) {
  srand(seed);
  int aligned_cols = (cols + align - 1) / align * align;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < aligned_cols; j++) {
      aligned_matrix[i * aligned_cols + j] = (j < cols) ? (rand() % 128) : 0;
    }
  }
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i * cols + j] = aligned_matrix[i * aligned_cols + j];
    }
  }
}
// 转置矩阵
void transpose_u8_matrix(elem_t *src, elem_t *dst, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      dst[j * rows + i] = src[i * cols + j];
    }
  }
}
void transpose_u32_matrix(result_t *src, result_t *dst, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      dst[j * rows + i] = src[i * cols + j];
    }
  }
}

// CPU矩阵乘法（用于生成期望结果）
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

unsigned long long read_cycle(void) {
  unsigned long long c;
  asm volatile("csrr %0, cycle" : "=r"(c));
  return c;
}
