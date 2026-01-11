#include "../toy/buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM (BANK_WIDTH / sizeof(elem_t) / 8)
#define DIM2 1024

// Test matrices
static elem_t input_matrix[DIM * DIM2] __attribute__((aligned(16)));
static elem_t output_matrix[DIM * DIM2] __attribute__((aligned(16)));

int mvin_mvout_simple_test() {
  for (int i = 0; i < 4; i++) {
    init_u8_random_matrix(input_matrix, DIM, DIM2, 111);
    bb_mvin((uintptr_t)input_matrix, 0, DIM2, 1);
    clear_u8_matrix(output_matrix, DIM, DIM2);
    bb_fence();
    bb_mvout((uintptr_t)output_matrix, 0, DIM2, 1);
    bb_fence();
    if (!compare_u8_matrices(output_matrix, input_matrix, DIM, DIM2)) {
      printf("Test mvin/mvout simple %d FAILED\n", i);
      return 0;
    } else {
      printf("Test mvin/mvout simple %d PASSED\n", i);
    }
  }
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = mvin_mvout_simple_test();
  if (passed) {
    printf("mvin/mvout simple test PASSED\n");
  } else {
    printf("mvin/mvout simple test FAILED\n");
  }
#ifdef MULTICORE
  exit(0);
#endif
}
