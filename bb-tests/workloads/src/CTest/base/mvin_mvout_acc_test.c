#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <params.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// One ACC row spans four configured physical bank rows.
enum { ROWS = 16, COLS = (4 * BANK_WIDTH / 32) };

static result_t output_matrix[ROWS * COLS] __attribute__((aligned(64)));
static result_t expected_matrix[ROWS * COLS] __attribute__((aligned(64)));

int acc_mvin_mvout_pressure_test() {
  for (int i = 0; i < 4; i++) {
    init_u32_random_matrix(expected_matrix, ROWS, COLS, i * 10 + i);
    clear_u32_matrix(output_matrix, ROWS, COLS);

    uint32_t acc_bank_id = 2;
    bb_mem_alloc(acc_bank_id, 1, 4);
    bb_mvin((uintptr_t)expected_matrix, acc_bank_id, ROWS, 1);
    bb_mvout((uintptr_t)output_matrix, acc_bank_id, ROWS, 1);
    if (!compare_u32_matrices(output_matrix, expected_matrix, ROWS, COLS)) {
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
