#include "smatmul_test_common.h"

#define ROWS 32
#define K 32

static elem_t a[ROWS * K] __attribute__((aligned(64)));
static elem_t b[K * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t packed_a[ROWS * K] __attribute__((aligned(64)));
static elem_t packed_b[K * MATRIX_TILE] __attribute__((aligned(64)));
static result_t packed_c[ROWS * MATRIX_TILE] __attribute__((aligned(64)));
static result_t expected[ROWS * MATRIX_TILE] __attribute__((aligned(64)));
static result_t actual[ROWS * MATRIX_TILE] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < ROWS * K; ++i)
    a[i] = (i * 11 % 29) - 14;
  for (int i = 0; i < K * MATRIX_TILE; ++i)
    b[i] = (i * 3 % 31) - 15;
  cpu_matmul(a, b, expected, ROWS, MATRIX_TILE, K);
  matrix_pack_a(a, packed_a, ROWS, K);
  matrix_pack_b(b, packed_b, K);
  matrix_hw(packed_a, packed_b, packed_c, ROWS, K);
  matrix_unpack_c(packed_c, actual, ROWS);
  if (!compare_u32_matrices(actual, expected, ROWS, MATRIX_TILE))
    return 1;
  printf("smatmul_32x16x32 PASSED\n");
  return 0;
}
