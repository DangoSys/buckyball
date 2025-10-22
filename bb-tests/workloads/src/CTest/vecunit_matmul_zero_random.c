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
  static elem_t a_transposed[DIM * DIM] __attribute__((aligned(64)));
  transpose_u8_matrix(a, a_transposed, size, size);
  uint32_t op1_addr = spad_addr(0, 0); // spad0: 操作数A, 偏移0
  uint32_t op2_addr = spad_addr(1, 0); // spad1: 操作数B, 偏移0
  uint32_t wr_addr = spad_addr(2, 0);  // acc0: 写入累加器, 偏移0

  bb_mvin((uintptr_t)a_transposed, op1_addr, size, 1);
  bb_mvin((uintptr_t)b, op2_addr, size, 1);

  bb_fence();
  bb_mul_warp16(op1_addr, op2_addr, wr_addr, size);
  bb_fence();
  bb_mvout((uintptr_t)c, wr_addr, size << 2);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  clear_u32_matrix(output_matrix, DIM, DIM);
  cpu_matmul(a, b, expected_matrix, size, size, size);
  hw_matmul(test_name, a, b, output_matrix, size);
  if (compare_u32_matrices(output_matrix, expected_matrix, size, size)) {
    printf("Test %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test %s FAILED\n", test_name);
    return 0;
  }
}

int test_zero_random() {
  clear_u8_matrix(input_matrix_a, DIM, DIM);
  init_random_matrix(input_matrix_b, DIM, DIM, 555);
  return run_test("Zero × Random", input_matrix_a, input_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_zero_random();
  if (passed) {
    printf("vecunit_matmul_zero_random test PASSED\n");
    return 0;
  } else {
    printf("vecunit_matmul_zero_random test FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
