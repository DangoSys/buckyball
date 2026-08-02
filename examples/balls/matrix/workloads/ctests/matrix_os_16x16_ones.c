#include "matrix_test_common.h"
#include <stdlib.h>

#define DIM 16

static elem_t a[DIM * DIM] __attribute__((aligned(64)));
static elem_t b[DIM * DIM] __attribute__((aligned(64)));
static elem_t pa[DIM * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb[DIM * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc[DIM * MATRIX_ACC_LANES] __attribute__((aligned(64)));
static result_t out[DIM * DIM] __attribute__((aligned(64)));
static result_t exp_[DIM * DIM] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_ones_matrix(a, DIM, DIM);
  init_ones_matrix(b, DIM, DIM);
  cpu_matmul(a, b, exp_, DIM, DIM, DIM);
  matrix_pack_a(a, pa, DIM, DIM);
  matrix_pack_b(b, pb, DIM, DIM);
  clear_u32_matrix(pc, DIM, MATRIX_ACC_LANES);
  matrix_hw_mnk(pa, pb, pc, DIM, DIM, DIM, 0);
  matrix_unpack_c(pc, out, DIM, DIM);
  if (!compare_u32_matrices(out, exp_, DIM, DIM)) {
    printf("matrix_os_16x16_ones FAILED\n");
    return 1;
  }
  printf("matrix_os_16x16_ones PASSED\n");
  return 0;
}
