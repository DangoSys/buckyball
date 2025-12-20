#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

// Column count n for 16xn matrix multiplication
#define MATMUL_COL 50
// 16-byte alignment of n
#define MAX_ALIGNED_MATMUL_COL 64
static elem_t aligned_input_matrix_a[DIM * MAX_ALIGNED_MATMUL_COL]
    __attribute__((aligned(16)));
static elem_t input_matrix_a[DIM * MATMUL_COL] __attribute__((aligned(16)));
static elem_t input_matrix_b[MATMUL_COL * DIM] __attribute__((aligned(16)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

void hw_matmul(const char *test_name, elem_t *a, elem_t *b, result_t *c,
               int size) {
  // spad0: operand A, offset 0
  uint32_t op1_bank_id = 0;
  // spad1: operand B, offset 0
  uint32_t op2_bank_id = 1;
  // acc0: write to accumulator, offset 0
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);
  uint32_t col_stride = (size + DIM - 1) / DIM;
  for (int i = 0; i < col_stride; i++) {
    bb_mvin((uintptr_t)a + i * DIM, op2_bank_id + size + i * DIM, DIM,
            col_stride);
  }
  bb_mvin((uintptr_t)b, op2_bank_id, size, 1);
  bb_fence();
  bb_transpose(op2_bank_id + size, op1_bank_id, size, 0);
  bb_fence();
  bb_mul_warp16(op1_bank_id, op2_bank_id, acc_bank_id, size, 0);
  bb_fence();
  bb_mvout((uintptr_t)c, acc_bank_id, DIM << 2, 1);
  bb_fence();
}

int run_test(const char *test_name, elem_t *aligned_a, elem_t *a, elem_t *b,
             int size) {
  clear_u32_matrix(output_matrix, DIM, DIM);
  hw_matmul(test_name, aligned_a, b, output_matrix, size);
  cpu_matmul(a, b, expected_matrix, DIM, DIM, size);
  if (compare_u32_matrices(output_matrix, expected_matrix, DIM, DIM)) {
    printf("Test %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test %s FAILED\n", test_name);
    return 0;
  }
}

int test_transpose_matmul() {
  init_col_aligned_random_matrix(aligned_input_matrix_a, input_matrix_a, DIM,
                                 DIM, MATMUL_COL, 111);
  init_u8_random_matrix(input_matrix_b, MATMUL_COL, DIM, 222);
  return run_test("Transpose Matmul", aligned_input_matrix_a, input_matrix_a,
                  input_matrix_b, MATMUL_COL);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_transpose_matmul();
  if (passed) {
    printf("Transpose Matmul test PASSED\n");
    return 0;
  } else {
    printf("Transpose Matmul test FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
