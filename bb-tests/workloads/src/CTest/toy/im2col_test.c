#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

static elem_t input_matrix_a[DIM * 64] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * 1024] __attribute__((aligned(64)));

void hw_im2col(const char *test_name, elem_t *a, elem_t *b, int size) {
  // spad0: operand A, offset 0
  uint32_t op1_bank_id = 0;
  // spad1: operand B, offset 0
  uint32_t op2_bank_id = 1;
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);
  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_fence();
  uint64_t krow = 4;
  uint64_t kcol = 1;
  uint64_t inrow = 16;
  uint64_t incol = 16;
  uint64_t startrow = 1;
  uint64_t startcol = 1;
  // bb_im2col(op1_bank_id, op2_bank_id, krow, kcol, inrow, incol, startrow,
  // startcol);
  bb_im2col(op1_bank_id, op2_bank_id, krow, kcol, inrow, incol, startrow,
            startcol);
  bb_fence();
  bb_mvout((uintptr_t)b, op2_bank_id, size, 1);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  hw_im2col(test_name, a, b, size);
  return 1;
}

int test_im2col() {
  init_sequence_matrix(input_matrix_a, DIM, 32);
  return run_test("Im2col", input_matrix_a, output_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_im2col();
  if (passed) {
    printf("Im2col test PASSED\n");
  } else {
    printf("Im2col test FAILED\n");
  }
#ifdef MULTICORE
  exit(0);
#endif
}
