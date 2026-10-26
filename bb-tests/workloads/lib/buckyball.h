#ifndef BUCKYBALL_H
#define BUCKYBALL_H

#include <stdint.h>

// Data type for matrix elements
typedef int8_t elem_t;
typedef int32_t result_t;

static inline void multithread(int target_hart_id) {
  int hart_id;
  asm volatile("csrr %0, mhartid" : "=r"(hart_id));

  if (hart_id != target_hart_id) {
    while (1) {
      asm volatile("wfi"); // Wait for interrupt
    }
  }
  // If hart_id == target_hart_id, continue execution
}

// Utility functions
void init_u8_random_matrix(elem_t *matrix, int rows, int cols, int seed);
void init_u32_random_matrix(result_t *matrix, int rows, int cols, int seed);

int compare_u8_matrices(elem_t *a, elem_t *b, int rows, int cols);
int compare_u32_matrices(result_t *a, result_t *b, int rows, int cols);
int compare_i8_matrices(elem_t *a, elem_t *b, int rows, int cols);
int compare_i32_matrices(result_t *a, result_t *b, int rows, int cols);
void clear_u32_matrix(result_t *matrix, int rows, int cols);
void clear_u8_matrix(elem_t *matrix, int rows, int cols);
void clear_i8_matrix(elem_t *matrix, int rows, int cols);
/* Matrix operation functions */
void transpose_u8_matrix(elem_t *src, elem_t *dst, int rows, int cols);
void cpu_matmul(elem_t *a, elem_t *b, result_t *c, int rows, int cols,
                int inner);
result_t gemmini_in_shift(result_t v, int shift);

#endif
