#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>

#define MATMUL_COL 50             // 16xn矩阵乘法的列数n
#define MAX_ALIGNED_MATMUL_COL 64 // n的16字节对齐
static elem_t aligned_input_matrix_a[DIM * MAX_ALIGNED_MATMUL_COL]
    __attribute__((aligned(16)));
static elem_t input_matrix_a[DIM * MATMUL_COL] __attribute__((aligned(16)));
static elem_t input_matrix_b[MATMUL_COL * DIM] __attribute__((aligned(16)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

void hw_matmul(const char *test_name, elem_t *a, elem_t *b, result_t *c,
               int size) {
  uint32_t op1_addr = spad_addr(0, 0); // spad0: 操作数A, 偏移0
  uint32_t op2_addr = spad_addr(1, 0); // spad1: 操作数B, 偏移0
  uint32_t wr_addr = spad_addr(4, 0);  // acc0: 写入累加器, 偏移0
  uint32_t col_stride = (size + DIM - 1) / DIM;
  for (int i = 0; i < col_stride; i++) {
    bb_mvin((uintptr_t)a + i * DIM, op2_addr + size + i * DIM, DIM, col_stride);
  }
  bb_mvin((uintptr_t)b, op2_addr, size, 1);
  bb_mvin((uintptr_t)c, wr_addr, DIM << 2, 1);
  bb_fence();
  bb_transpose(op2_addr + size, op1_addr, size, 0);
  bb_fence();
  bb_mul_warp16(op1_addr, op2_addr, wr_addr, size, 0);
  bb_fence();
  bb_mvout((uintptr_t)c, wr_addr, DIM << 2, 1);
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

int test_aligned_matmul() {
  init_col_aligned_random_matrix(aligned_input_matrix_a, input_matrix_a, DIM,
                                 DIM, MATMUL_COL, 111);
  init_u8_random_matrix(input_matrix_b, MATMUL_COL, DIM, 222);
  return run_test("Aligned Matmul", aligned_input_matrix_a, input_matrix_a,
                  input_matrix_b, MATMUL_COL);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_aligned_matmul();
  if (passed) {
    printf("Aligned Matmul test PASSED\n");
    return 0;
  } else {
    printf("Aligned Matmul test FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
