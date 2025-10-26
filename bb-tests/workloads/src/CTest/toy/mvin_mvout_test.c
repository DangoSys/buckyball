#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test matrices
static elem_t input_matrix_a[DIM * 1024] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * 1024] __attribute__((aligned(64)));
static elem_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static elem_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));
static elem_t a_transposed[DIM * 1024] __attribute__((aligned(64)));

int alternately_mvin_mvout_pressure_test() {
  for (int i = 0; i < 4; i++) {
    init_u8_random_matrix(expected_matrix, DIM, DIM, i * 10 + i);
    uint32_t wr_addr = spad_addr(0, i);
    bb_mvin((uintptr_t)expected_matrix, wr_addr, DIM, 1);
    clear_u32_matrix(output_matrix, DIM, DIM);
    bb_mvout((uintptr_t)output_matrix, wr_addr, DIM, 1);
    bb_fence();
    if (!compare_u8_matrices(output_matrix, expected_matrix, DIM, DIM)) {
      printf("Test mvin/mvout pressure %d FAILED\n", i);
      return 0;
    } else {
      printf("Test mvin/mvout pressure %d PASSED\n", i);
    }
  }
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = alternately_mvin_mvout_pressure_test();
  if (passed) {
    printf("Alternately mvin/mvout pressure test PASSED\n");
  } else {
    printf("Alternately mvin/mvout pressure test FAILED\n");
  }
#ifdef MULTICORE
  exit(0);
#endif
}
