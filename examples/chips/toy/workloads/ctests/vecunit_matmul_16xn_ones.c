#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 16

static elem_t input_matrix_a[DIM * 32] __attribute__((aligned(64)));
static elem_t input_matrix_b[32 * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

void hw_matmul(const char *test_name, elem_t *a, elem_t *b, result_t *c,
               int size) {
  (void)test_name;
  if (size % DIM != 0) {
    printf("K=%d not multiple of %d\n", size, DIM);
    exit(1);
  }

  uint32_t op1 = 0;
  uint32_t op2 = 1;
  uint32_t acc = 2;
  uint32_t a_t = 3;
  static result_t zero[DIM * DIM] __attribute__((aligned(64))) = {0};
  static elem_t tile_a[DIM * DIM] __attribute__((aligned(64)));

  bb_mem_alloc(op1, 1, 1);
  bb_mem_alloc(op2, 1, 1);
  bb_mem_alloc(acc, 1, 4);
  bb_mem_alloc(a_t, 1, 1);
  bb_mvin((uintptr_t)zero, acc, DIM, 1);

  for (int k0 = 0; k0 < size; k0 += DIM) {
    for (int r = 0; r < DIM; r++)
      memcpy(&tile_a[r * DIM], &a[r * size + k0], (size_t)DIM);
    bb_mvin((uintptr_t)tile_a, op1, DIM, 1);
    bb_mvin((uintptr_t)(b + k0 * DIM), op2, DIM, 1);
    bb_transpose(op1, a_t, DIM, 8);
    bb_mul_warp16(a_t, op2, acc, DIM, 0);
  }

  bb_mvout((uintptr_t)c, acc, DIM, 1);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  hw_matmul(test_name, a, b, output_matrix, size);
  cpu_matmul(a, b, expected_matrix, DIM, DIM, size);
  if (compare_u32_matrices(output_matrix, expected_matrix, DIM, DIM)) {
    printf("Test %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test %s FAILED\n", test_name);
    return 0;
  }
}

int test_ones_16x32() {
  init_sequence_matrix(input_matrix_a, DIM, 32);
  init_sequence_matrix(input_matrix_b, 32, DIM);
  return run_test("All-ones matrices", input_matrix_a, input_matrix_b, 32);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_ones_16x32();
  if (passed) {
    printf("vecunit_matmul_16xn_ones test PASSED\n");
    return 0;
  } else {
    printf("vecunit_matmul_16xn_ones test FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
