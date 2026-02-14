#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16 // 强制 16x16

// =======================
// 固定输入矩阵（4行周期）
// =======================
static elem_t input_matrix[DIM * DIM] __attribute__((aligned(64))) = {
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

// =======================
// 直接写死 ReLU 结果
// =======================
static elem_t expected_matrix[DIM * DIM] __attribute__((aligned(64))) = {
    // ---- Cycle 1 ----
    0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 0, 0, 0, 0, 0, 0, 0, 30, 31, 32, 33, 34, 35, 36, 37, 38,

    // ---- Cycle 2 ----
    0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 0, 0, 0, 0, 0, 0, 0, 30, 31, 32, 33, 34, 35, 36, 37, 38,

    // ---- Cycle 3 ----
    0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 0, 0, 0, 0, 0, 0, 0, 30, 31, 32, 33, 34, 35, 36, 37, 38,

    // ---- Cycle 4 ----
    0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 0, 0, 0, 0, 0, 0, 0, 30, 31, 32, 33, 34, 35, 36, 37, 38};

static elem_t output_matrix[DIM * DIM] __attribute__((aligned(64)));

// =======================
// HW ReLU Flow（保持不变）
// =======================
void hw_relu(const char *test_name, elem_t *a, elem_t *b, int size) {
  uint32_t op1_bank_id = 0;
  uint32_t wr_bank_id = 1;

  bb_vbank_config(op1_bank_id, 0, 1);
  bb_vbank_config(wr_bank_id, 0, 1);

  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_fence();

  bb_relu(op1_bank_id, wr_bank_id, size);
  bb_fence();

  bb_mvout((uintptr_t)b, wr_bank_id, size, 1);
}

// =======================
// 测试函数（去掉 CPU 计算）
// =======================
int run_test(const char *test_name, elem_t *a, int size) {
  clear_i8_matrix(output_matrix, size, size);

  hw_relu(test_name, a, output_matrix, size);

  if (compare_i8_matrices(output_matrix, expected_matrix, size, size)) {
    printf("%s compare test PASSED\n", test_name);
    return 1;
  } else {
    printf("%s compare test FAILED\n", test_name);
    return 0;
  }
}

int test_relu(int seed) { return run_test("ReLU", input_matrix, DIM); }

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = test_relu(5);
  if (passed) {
    printf("ReLU test PASSED!!!\n");
  } else {
    printf("ReLU test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
