#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test matrices
static elem_t input_matrix_a[DIM * 1024] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * 1024] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));
static elem_t a_transposed[DIM * 1024] __attribute__((aligned(64)));

int acc_mvin_mvout_pressure_test() {
  for (int i = 0; i < 4; i++) {
    init_u32_random_matrix(expected_matrix, DIM, DIM, i * 10 + i);

    uint32_t wr_addr = spad_addr(4, i);
    bb_mvin((uintptr_t)expected_matrix, wr_addr, DIM << 2, 1);
    init_u32_random_matrix(expected_matrix, DIM, DIM, i * 10 + i);
    clear_u32_matrix(output_matrix, DIM, DIM);

    wr_addr = spad_addr(4, i);
    bb_mvout((uintptr_t)output_matrix, wr_addr, DIM << 2);
    bb_fence();
    if (!compare_u32_matrices(output_matrix, expected_matrix, DIM, DIM)) {
      printf("Test ACC mvin/mvout pressure %d FAILED\n", i);
      return 0;
    } else {
      printf("Test ACC mvin/mvout pressure %d PASSED\n", i);
    }
  }
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = acc_mvin_mvout_pressure_test();
  if (passed) {
    printf("ACC mvin/mvout pressure test PASSED\n");
  } else {
    printf("ACC mvin/mvout pressure test FAILED\n");
  }
#ifdef MULTICORE
  exit(0);
#endif
}
