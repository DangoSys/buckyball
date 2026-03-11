#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64))) = {
    // ---- Cycle 1 ----
    // Row1
    -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    // Row2
    -17, -16, -15, -14, -13, -12, -11, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    // Row3
    -27, -26, -25, -24, -23, -22, -21, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    // Row4
    -37, -36, -35, -34, -33, -32, -31, 30, 31, 32, 33, 34, 35, 36, 37, 38,

    // ---- Cycle 2 ----
    -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -17, -16, -15, -14,
    -13, -12, -11, 10, 11, 12, 13, 14, 15, 16, 17, 18, -27, -26, -25, -24, -23,
    -22, -21, 20, 21, 22, 23, 24, 25, 26, 27, 28, -37, -36, -35, -34, -33, -32,
    -31, 30, 31, 32, 33, 34, 35, 36, 37, 38,

    // ---- Cycle 3 ----
    -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -17, -16, -15, -14,
    -13, -12, -11, 10, 11, 12, 13, 14, 15, 16, 17, 18, -27, -26, -25, -24, -23,
    -22, -21, 20, 21, 22, 23, 24, 25, 26, 27, 28, -37, -36, -35, -34, -33, -32,
    -31, 30, 31, 32, 33, 34, 35, 36, 37, 38,

    // ---- Cycle 4 ----
    -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -17, -16, -15, -14,
    -13, -12, -11, 10, 11, 12, 13, 14, 15, 16, 17, 18, -27, -26, -25, -24, -23,
    -22, -21, 20, 21, 22, 23, 24, 25, 26, 27, 28, -37, -36, -35, -34, -33, -32,
    -31, 30, 31, 32, 33, 34, 35, 36, 37, 38};

static elem_t expected_matrix[DIM * DIM] __attribute__((aligned(64))) = {
    // Row 1 of A^T (col 1 of A)
    -7, -17, -27, -37, -7, -17, -27, -37, -7, -17, -27, -37, -7, -17, -27, -37,
    // Row 2
    -6, -16, -26, -36, -6, -16, -26, -36, -6, -16, -26, -36, -6, -16, -26, -36,
    // Row 3
    -5, -15, -25, -35, -5, -15, -25, -35, -5, -15, -25, -35, -5, -15, -25, -35,
    // Row 4
    -4, -14, -24, -34, -4, -14, -24, -34, -4, -14, -24, -34, -4, -14, -24, -34,
    // Row 5
    -3, -13, -23, -33, -3, -13, -23, -33, -3, -13, -23, -33, -3, -13, -23, -33,
    // Row 6
    -2, -12, -22, -32, -2, -12, -22, -32, -2, -12, -22, -32, -2, -12, -22, -32,
    // Row 7
    -1, -11, -21, -31, -1, -11, -21, -31, -1, -11, -21, -31, -1, -11, -21, -31,
    // Row 8
    0, 10, 20, 30, 0, 10, 20, 30, 0, 10, 20, 30, 0, 10, 20, 30,
    // Row 9
    1, 11, 21, 31, 1, 11, 21, 31, 1, 11, 21, 31, 1, 11, 21, 31,
    // Row 10
    2, 12, 22, 32, 2, 12, 22, 32, 2, 12, 22, 32, 2, 12, 22, 32,
    // Row 11
    3, 13, 23, 33, 3, 13, 23, 33, 3, 13, 23, 33, 3, 13, 23, 33,
    // Row 12
    4, 14, 24, 34, 4, 14, 24, 34, 4, 14, 24, 34, 4, 14, 24, 34,
    // Row 13
    5, 15, 25, 35, 5, 15, 25, 35, 5, 15, 25, 35, 5, 15, 25, 35,
    // Row 14
    6, 16, 26, 36, 6, 16, 26, 36, 6, 16, 26, 36, 6, 16, 26, 36,
    // Row 15
    7, 17, 27, 37, 7, 17, 27, 37, 7, 17, 27, 37, 7, 17, 27, 37,
    // Row 16
    8, 18, 28, 38, 8, 18, 28, 38, 8, 18, 28, 38, 8, 18, 28, 38};
static elem_t output_matrix_b[DIM * 1024] __attribute__((aligned(64)));

void hw_transpose(const char *test_name, elem_t *a, elem_t *b, int size) {
  // spad0: operand A, offset 0
  uint32_t op1_bank_id = 0;
  // spad1: operand B, offset 0
  uint32_t op2_bank_id = 1;

  bb_mem_alloc_private(op1_bank_id, 1, 1);
  bb_mem_alloc_private(op2_bank_id, 1, 1);

  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_transpose(op1_bank_id, op2_bank_id, size, 0);
  bb_mvout((uintptr_t)b, op2_bank_id, size, 1);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  hw_transpose(test_name, a, b, size);
  if (compare_i8_matrices(output_matrix_b, expected_matrix, size, size)) {
    printf("%s compare test PASSED\n", test_name);
    return 1;
  } else {
    printf("%s compare test FAILED\n", test_name);
    return 0;
  }
}

int test_transpose() {
  // init_sequence_matrix(input_matrix_a, DIM, DIM);
  return run_test("transpose", input_matrix_a, output_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_transpose();
  if (passed) {
    printf("Transpose test PASSED\n");
    return 0;
  } else {
    printf("Transpose test FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
