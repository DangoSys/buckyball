#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * DIM] __attribute__((aligned(64)));
static elem_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));
// static elem_t probe_matrix[DIM * DIM] __attribute__((aligned(64)));
// Used to verify content in SPAD after MVIN


// bb_transfer(op1_addr, wr_addr, iter) wrapper in bbhw implementation
// (func7=TRANSFER_FUNC7).

void hw_transfer(const char *test_name, elem_t *a, elem_t *b, int size) {
  // Source operand in spad bank 0, write target in spad bank 1
  uint32_t op1_addr = spad_addr(0, 0);
  uint32_t wr_addr = spad_addr(1, 0);

  // Move input into scratchpad bank0, starting at offset 0, iterate size times
  // row-wise
  bb_mvin((uintptr_t)a, op1_addr, size, 1);
  bb_fence();
  // Call Transfer instruction
  bb_transfer(op1_addr, wr_addr, size);
  bb_fence();
  bb_mvout((uintptr_t)b, wr_addr, size, 1);
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  clear_i8_matrix(output_matrix_b, size, size);
  cpu_transfer(a, expected_matrix, size, size);
  hw_transfer(test_name, a, output_matrix_b, size);
  if (!compare_i8_matrices(expected_matrix, output_matrix_b, size, size)) {
    printf("%s: Output matrix does not match expected result!\n", test_name);
    return 0;
  }
  printf("%s: Output matrix match expected result!\n", test_name);
  return 1;
}

int test_transfer(int seed) {
  init_i8_random_matrix(input_matrix_a, DIM, DIM, seed);
  return run_test("Transfer", input_matrix_a, output_matrix_b, DIM);
  // TransferBall test code, need to comment out the code block above
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_transfer(5);
  if (passed) {
    printf("Transfer test PASSED!\n");
  } else {
    printf("Transfer test FAILED!\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
