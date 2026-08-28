#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ACC bank row is 4 physical banks x 16B = 64B = 16 x int32.
#define DIM 16

static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

int acc_mvin_mvout_pressure_test() {
  for (int i = 0; i < 4; i++) {
    init_u32_random_matrix(expected_matrix, DIM, DIM, i * 10 + i);
    clear_u32_matrix(output_matrix, DIM, DIM);

    uint32_t acc_bank_id = 2;
    bb_mem_alloc(acc_bank_id, 1, 4);
    bb_mvin((uintptr_t)expected_matrix, acc_bank_id, DIM, 1);
    bb_mvout((uintptr_t)output_matrix, acc_bank_id, DIM, 1);
    bb_fence();
    if (!compare_u32_matrices(output_matrix, expected_matrix, DIM, DIM)) {
      printf("Test ACC mvin/mvout pressure %d FAILED\n", i);
      return 0;
    }
    printf("Test ACC mvin/mvout pressure %d PASSED\n", i);
    bb_mem_release(acc_bank_id);
  }

  // Same-vbank realloc without release must free prior physical banks
  // (bemu/RTL).
  {
    uint32_t acc_bank_id = 2;
    for (int i = 0; i < 8; i++) {
      bb_mem_alloc(acc_bank_id, 1, 4);
    }
    bb_mem_release(acc_bank_id);
    printf("Test ACC same-vbank realloc without release PASSED\n");
  }
  return 1;
}

int main() {
  int passed = acc_mvin_mvout_pressure_test();
  if (passed) {
    printf("ACC mvin/mvout pressure test PASSED\n");
  } else {
    printf("ACC mvin/mvout pressure test FAILED\n");
  }
  return passed ? 0 : 1;
}
