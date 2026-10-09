#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <params.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROWS = 16, COLS = (BANK_WIDTH / 8) };

// Test matrices
static elem_t input_matrix[ROWS * COLS] __attribute__((aligned(128)));
static elem_t output_matrix[ROWS * COLS] __attribute__((aligned(128)));

int mvin_mvout_simple_test() {
  uint32_t bank_id = 0;
  bb_mem_alloc(bank_id, 1, 1);

  for (int i = 0; i < 1; i++) {
    init_u8_random_matrix(input_matrix, ROWS, COLS, 111);
    clear_u8_matrix(output_matrix, ROWS, COLS);
    bb_mvin((uintptr_t)input_matrix, bank_id, ROWS, 1);
    bb_mvout((uintptr_t)output_matrix, bank_id, ROWS, 1);
    if (!compare_u8_matrices(output_matrix, input_matrix, ROWS, COLS)) {
      printf("Test mvin/mvout simple %d FAILED\n", i);
      return 0;
    } else {
      printf("Test mvin/mvout simple %d PASSED\n", i);
    }
  }
  return 1;
}

int main() {
  int passed = mvin_mvout_simple_test();
  if (passed) {
    printf("mvin/mvout simple test PASSED\n");
  } else {
    printf("mvin/mvout simple test FAILED\n");
  }
  return passed ? 0 : 1;
}
