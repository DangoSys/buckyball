#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * 1024] __attribute__((aligned(64)));

void hw_transpose(const char *test_name, elem_t *a, elem_t *b, int size) {
  uint32_t op1_addr = spad_addr(0, 0); // spad0: 操作数A, 偏移0
  uint32_t op2_addr = spad_addr(1, 0); // spad1: 操作数B, 偏移0

  bb_mvin((uintptr_t)a, op1_addr, size);
  bb_fence();
  bb_transpose(op1_addr, op2_addr, size);
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
  } else {
    printf("Transpose test FAILED\n");
  }
#ifdef MULTICORE
  exit(0);
#endif
}
