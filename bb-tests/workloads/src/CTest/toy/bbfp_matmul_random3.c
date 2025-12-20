#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

void hw_matmul(const char *test_name, elem_t *a, elem_t *b, result_t *c,
               int size) {
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);
  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_mvin((uintptr_t)b, op2_bank_id, size, 1);
  bb_fence();
  bb_bbfp_mul(op1_bank_id, op2_bank_id, acc_bank_id, size);
  bb_fence();
  bb_mvout((uintptr_t)c, op2_bank_id, size << 2, 1);
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
  init_bbfp_random_matrix(input_matrix_a, DIM, DIM, 333);
  init_bbfp_random_matrix(input_matrix_b, DIM, DIM, 444);
  return run_test("Random matrices 3", input_matrix_a, input_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_random1();
  if (passed) {
    printf("bbfp_matmul_random3 test PASSED\n");
  } else {
    printf("bbfp_matmul_random3 test FAILED\n");
  }
#ifdef MULTICORE
  exit(0);
#endif
}
