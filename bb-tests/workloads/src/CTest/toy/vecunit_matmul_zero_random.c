#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

void hw_matmul(const char *test_name, elem_t *a, elem_t *b, result_t *c,
               int size) {
  // spad0: original A
  uint32_t op1_bank_id = 0;
  // spad1: operand B
  uint32_t op2_bank_id = 1;
  // acc0: write to accumulator
  int acc_bank_id = 2; // virtual bank id
  // spad3: transposed A
  uint32_t a_transposed_bank_id = 3;

  bb_mem_alloc_private(op1_bank_id, 1, 1);
  bb_mem_alloc_private(op2_bank_id, 1, 1);
  bb_mem_alloc_private(acc_bank_id, 1, 4);
  bb_mem_alloc_private(a_transposed_bank_id, 1, 1);

  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_mvin((uintptr_t)b, op2_bank_id, size, 1);
  bb_transpose(op1_bank_id, a_transposed_bank_id, size, 0);

  bb_mul_warp16(a_transposed_bank_id, op2_bank_id, acc_bank_id, size, 0);
  bb_mvout((uintptr_t)c, acc_bank_id, size, 1);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
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
