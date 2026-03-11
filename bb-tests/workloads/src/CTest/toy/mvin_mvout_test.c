#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 16

// Test matrices
static elem_t input_matrix[DIM * DIM] __attribute__((aligned(128)));
static elem_t output_matrix[DIM * DIM] __attribute__((aligned(128)));

int mvin_mvout_simple_test() {
  uint32_t bank_id = 0;
  bb_mem_alloc_private(bank_id, 1, 1);

  for (int i = 0; i < 1; i++) {
    init_u8_random_matrix(input_matrix, DIM, DIM, 111);
    bb_mvin((uintptr_t)input_matrix, bank_id, DIM, 1);
    clear_u8_matrix(output_matrix, DIM, DIM);
    bb_mvout((uintptr_t)output_matrix, bank_id, DIM, 1);
    if (!compare_u8_matrices(output_matrix, input_matrix, DIM, DIM)) {
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
