#include "smatmul_test_common.h"

#define ROWS 16
#define COLUMNS 512
#define K 16

static elem_t a[ROWS * K] __attribute__((aligned(64)));
static elem_t b[K * COLUMNS] __attribute__((aligned(64)));
static elem_t packed_a[ROWS * K] __attribute__((aligned(64)));
static elem_t packed_b[COLUMNS * MATRIX_TILE] __attribute__((aligned(64)));
static result_t packed_c[ROWS * COLUMNS] __attribute__((aligned(64)));
static result_t expected[ROWS * COLUMNS] __attribute__((aligned(64)));
static result_t actual[ROWS * COLUMNS] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < ROWS * K; ++i)
    a[i] = (i * 7 % 23) - 11;
  for (int i = 0; i < K * COLUMNS; ++i)
    b[i] = (i * 5 % 19) - 9;
  cpu_matmul(a, b, expected, ROWS, COLUMNS, K);
  matrix_pack_a(a, packed_a, ROWS, K);
  matrix_pack_b_ws(b, packed_b, K, COLUMNS);
  matrix_hw_ws(packed_a, packed_b, packed_c, ROWS, COLUMNS, K);
  matrix_unpack_c_ws(packed_c, actual, ROWS, COLUMNS);
  if (!compare_u32_matrices(actual, expected, ROWS, COLUMNS))
    return 1;
  printf("smatmul_ws_16x512 PASSED\n");
  return 0;
}
