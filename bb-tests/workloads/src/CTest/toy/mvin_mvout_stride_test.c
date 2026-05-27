#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define ROW_BYTES 16
#define ROWS 4
#define IN_STRIDE 3
#define OUT_STRIDE 5

static elem_t input_matrix[ROWS * IN_STRIDE * ROW_BYTES]
    __attribute__((aligned(128)));
static elem_t output_matrix[ROWS * OUT_STRIDE * ROW_BYTES]
    __attribute__((aligned(128)));
static elem_t expected_matrix[ROWS * OUT_STRIDE * ROW_BYTES]
    __attribute__((aligned(128)));

static void init_inputs(void) {
  for (int i = 0; i < ROWS * IN_STRIDE * ROW_BYTES; i++) {
    input_matrix[i] = (elem_t)((i * 7 + 3) & 0x7f);
  }

  for (int i = 0; i < ROWS * OUT_STRIDE * ROW_BYTES; i++) {
    output_matrix[i] = (elem_t)0xa5;
    expected_matrix[i] = (elem_t)0xa5;
  }

  for (int r = 0; r < ROWS; r++) {
    for (int b = 0; b < ROW_BYTES; b++) {
      expected_matrix[r * OUT_STRIDE * ROW_BYTES + b] =
          input_matrix[r * IN_STRIDE * ROW_BYTES + b];
    }
  }
}

static int compare_output(void) {
  for (int i = 0; i < ROWS * OUT_STRIDE * ROW_BYTES; i++) {
    if (output_matrix[i] != expected_matrix[i]) {
      printf("Mismatch at byte %d: expected %d, got %d\n", i,
             expected_matrix[i], output_matrix[i]);
      return 0;
    }
  }
  return 1;
}

int mvin_mvout_stride_test(void) {
  uint32_t bank_id = 0;

  init_inputs();
  bb_mem_alloc(bank_id, 1, 1);
  bb_mvin((uintptr_t)input_matrix, bank_id, ROWS, IN_STRIDE);
  bb_mvout((uintptr_t)output_matrix, bank_id, ROWS, OUT_STRIDE);
  bb_fence();

  if (!compare_output()) {
    printf("mvin/mvout stride test FAILED\n");
    return 0;
  }

  printf("mvin/mvout stride test PASSED\n");
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = mvin_mvout_stride_test();
#ifdef MULTICORE
  exit(0);
#endif
  return passed ? 0 : 1;
}
