#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

// Test matrices
static elem_t input_matrix_a[DIM * 1024] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * 1024] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));
static elem_t a_transposed[DIM * 1024] __attribute__((aligned(64)));

int acc_mvin_mvout_pressure_test() {
  for (int i = 0; i < 4; i++) {
    init_u32_random_matrix(expected_matrix, DIM, DIM, i * 10 + i);

    uint32_t acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);
    bb_mvin((uintptr_t)expected_matrix, acc_bank_id, DIM << 2, 1);
    init_u32_random_matrix(expected_matrix, DIM, DIM, i * 10 + i);
    clear_u32_matrix(output_matrix, DIM, DIM);

    acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);
    bb_mvout((uintptr_t)output_matrix, acc_bank_id, DIM << 2, 1);
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
