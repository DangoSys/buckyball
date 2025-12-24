#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

static elem_t input_matrix_a[DIM * 64] __attribute__((aligned(16)));
static elem_t input_matrix_b[64 * DIM] __attribute__((aligned(16)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

void hw_matmul(const char *test_name, elem_t *a, elem_t *b, result_t *c,
               int size) {
  static elem_t a_transposed[64 * DIM] __attribute__((aligned(16)));
  transpose_u8_matrix(a, a_transposed, DIM, 64);
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  bb_mvin((uintptr_t)a_transposed, op1_bank_id, size, 1);
  bb_mvin((uintptr_t)b, op2_bank_id, size, 1);
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);
  bb_fence();
  bb_mul_warp16(op1_bank_id, op2_bank_id, acc_bank_id, size, 0);
  bb_fence();
  bb_mvout((uintptr_t)c, acc_bank_id, DIM << 2, 1);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  clear_u32_matrix(output_matrix, DIM, DIM);
  cpu_matmul(a, b, expected_matrix, DIM, DIM, size);
  hw_matmul(test_name, a, b, output_matrix, size);
  if (compare_u32_matrices(output_matrix, expected_matrix, DIM, DIM)) {
    printf("Test %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test %s FAILED\n", test_name);
    return 0;
  }
}

int test_random1() {
  init_u8_random_matrix(input_matrix_a, DIM, 64, 111);
  init_u8_random_matrix(input_matrix_b, 64, DIM, 221);
  return run_test("Random matrices 1", input_matrix_a, input_matrix_b, 64);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_random1();
  if (passed) {
    printf("vecunit_matmul_16xn_random1 test PASSED\n");
    return 0;
  } else {
    printf("vecunit_matmul_16xn_random1 test FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
