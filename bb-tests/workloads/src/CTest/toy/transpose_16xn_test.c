#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16

static elem_t input_matrix[DIM * 64] __attribute__((aligned(64)));
static elem_t output_matrix[64 * DIM] __attribute__((aligned(64)));
static elem_t expected_matrix[64 * DIM] __attribute__((aligned(64)));

void hw_transpose(elem_t *a, elem_t *b, int size) {
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(op2_bank_id, 1, 1);

  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_transpose(op1_bank_id, op2_bank_id, size, 0);
  bb_mvout((uintptr_t)b, op2_bank_id, size, 1);
  bb_fence();
}

int run_test(const char *test_name, int cols) {
  int size = cols; // iter = number of rows in scratchpad = cols for 16xN

  // Initialize input: A[i][j] = i + j
  init_sequence_matrix(input_matrix, DIM, cols);

  // Compute expected transpose on CPU
  transpose_u8_matrix(input_matrix, expected_matrix, DIM, cols);

  // Run hardware transpose
  hw_transpose(input_matrix, output_matrix, size);

  // Compare results
  if (compare_u8_matrices(output_matrix, expected_matrix, cols, DIM)) {
    printf("Test %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test %s FAILED\n", test_name);
    return 0;
  }
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int all_passed = 1;

  all_passed &= run_test("transpose 16x16", 16);
  all_passed &= run_test("transpose 16x32", 32);

  if (all_passed) {
    printf("All transpose 16xn tests PASSED\n");
    return 0;
  } else {
    printf("Some transpose 16xn tests FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
