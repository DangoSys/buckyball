#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * 1024] __attribute__((aligned(64)));

void hw_transpose(const char *test_name, elem_t *a, elem_t *b, int size) {
  // spad0: operand A, offset 0
  uint32_t op1_bank_id = 0;
  // spad1: operand B, offset 0
  uint32_t op2_bank_id = 1;

  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_fence();
  bb_transpose(op1_bank_id, op2_bank_id, size, 0);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  hw_transpose(test_name, a, b, size);
  return 1;
}

int test_transpose() {
  init_sequence_matrix(input_matrix_a, DIM, DIM);
  return run_test("Im2col", input_matrix_a, output_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_transpose();
  if (passed) {
    printf("Transpose test PASSED\n");
    return 0;
  } else {
    printf("Transpose test FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
