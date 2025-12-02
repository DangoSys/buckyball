#ifndef BUCKYBALL_H
#define BUCKYBALL_H

#include <stdint.h>

// String macros (from xcustom.h)
// #define STR1(x) #x
// #ifndef STR
// #define STR(x) STR1(x)
// #endif

#define CAT_(A, B) A##B
#define CAT(A, B) CAT_(A, B)

// Data type for matrix elements
typedef int8_t elem_t;
typedef int32_t result_t;

#define MULTICORE_INIT(hart_id)                                                \
  __attribute__((constructor)) static void _multicore_init() {                 \
    multicore(hart_id);                                                        \
  }

static inline void multicore(int target_hart_id) {
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

/* Read CPU cycle counter (64-bit). Implemented in buckyball.c for global use */
unsigned long long read_rdcycle(void);

void print_u32_matrix(const char *name, result_t *matrix, int rows, int cols);
void print_u8_matrix(const char *name, elem_t *matrix, int rows, int cols);
void print_i32_matrix(const char *name, result_t *matrix, int rows, int cols);
void print_i8_matrix(const char *name, elem_t *matrix, int rows, int cols);

void init_u8_random_matrix(elem_t *matrix, int rows, int cols, int seed);
void init_u32_random_matrix(result_t *matrix, int rows, int cols, int seed);
void init_i8_random_matrix(elem_t *matrix, int rows, int cols, int seed);
void init_i32_random_matrix(result_t *matrix, int rows, int cols, int seed);

int compare_u8_matrices(elem_t *a, elem_t *b, int rows, int cols);
int compare_u32_matrices(result_t *a, result_t *b, int rows, int cols);
int compare_i8_matrices(elem_t *a, elem_t *b, int rows, int cols);
int compare_i32_matrices(result_t *a, result_t *b, int rows, int cols);
int compare_u32_matrices_with_tolerance(result_t *a, result_t *b, int rows,
                                        int cols, double tolerance);
void clear_u32_matrix(result_t *matrix, int rows, int cols);
void clear_u8_matrix(elem_t *matrix, int rows, int cols);
void clear_i32_matrix(result_t *matrix, int rows, int cols);
void clear_i8_matrix(elem_t *matrix, int rows, int cols);

void init_ones_matrix(elem_t *matrix, int rows, int cols);
void init_identity_matrix(elem_t *matrix, int size);
void init_row_vector(elem_t *matrix, int cols, elem_t value);
void init_col_vector(elem_t *matrix, int rows, elem_t value);
void init_random_matrix(elem_t *matrix, int rows, int cols, int seed);
void init_bbfp_random_matrix(elem_t *matrix, int rows, int cols, int seed);
void init_sequence_matrix(elem_t *matrix, int rows, int cols);
void init_col_aligned_random_matrix(elem_t *aligned_matrix, elem_t *matrix,
                                    int align, int rows, int cols, int seed);
/* Matrix operation functions */
void transpose_u8_matrix(elem_t *src, elem_t *dst, int rows, int cols);
void transpose_u32_matrix(result_t *src, result_t *dst, int rows, int cols);
void cpu_matmul(elem_t *a, elem_t *b, result_t *c, int rows, int cols,
                int inner);
void cpu_relu(elem_t *a, elem_t *matrix, int rows, int cols);
void cpu_transfer(elem_t *src, elem_t *dst, int rows, int cols);

unsigned long long read_cycle(void);
#endif
