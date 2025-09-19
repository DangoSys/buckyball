#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

void hw_matmul(const char *test_name, elem_t *a, elem_t *b, result_t *c,
               int size) {
  uint32_t op1_addr = spad_addr(0, 0); // spad0: 操作数A, 偏移0
  uint32_t op2_addr = spad_addr(1, 0); // spad1: 操作数B, 偏移0
  uint32_t wr_addr = spad_addr(2, 0);  // acc0: 写入累加器, 偏移0

  printf("op1_addr: %d\n", op1_addr);
  printf("op2_addr: %d\n", op2_addr);
  printf("wr_addr: %d\n", wr_addr);

  bb_mvin((uintptr_t)a, op1_addr, size);
  bb_mvin((uintptr_t)b, op2_addr, size);
  bb_mvin((uintptr_t)c, wr_addr, size << 2);
  bb_fence();
  bb_bbfp_mul(op1_addr, op2_addr, wr_addr, size);
  bb_fence();
  bb_mvout((uintptr_t)c, wr_addr, size << 2);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  clear_u32_matrix(output_matrix, DIM, DIM);
  cpu_matmul(b, a, expected_matrix, size, size, size);
  hw_matmul(test_name, a, b, output_matrix, size);
  if (compare_u32_matrices(output_matrix, expected_matrix, size, size)) {
    printf("Test %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test %s FAILED\n", test_name);
    return 0;
  }
}

int test_random1() {
  init_bbfp_random_matrix(input_matrix_a, DIM, DIM, 111);
  init_bbfp_random_matrix(input_matrix_b, DIM, DIM, 222);
  return run_test("Random matrices 2", input_matrix_a, input_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_random1();
  if (passed) {
    printf("bbfp_matmul_random2 test PASSED\n");
  } else {
    printf("bbfp_matmul_random2 test FAILED\n");
  }
#ifdef MULTICORE
  exit(0);
#endif
}
