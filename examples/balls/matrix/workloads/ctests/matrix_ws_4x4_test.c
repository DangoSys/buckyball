#include "matrix_test_common.h"

#define M 4
#define N 4
#define K 4

static elem_t a[M * K] __attribute__((aligned(64))) = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
};
static elem_t b[K * N] __attribute__((aligned(64))) = {
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
};
static elem_t pa[16 * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb[16 * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc[M * MATRIX_ACC_LANES] __attribute__((aligned(64)));
static result_t out[M * N] __attribute__((aligned(64)));
static result_t exp_[M * N] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  cpu_matmul(a, b, exp_, M, N, K);
  matrix_pack_a(a, pa, M, K);
  matrix_pack_b(b, pb, K, N);
  clear_u32_matrix(pc, M, MATRIX_ACC_LANES);
  matrix_hw_mnk(pa, pb, pc, M, N, K, 1);
  matrix_unpack_c(pc, out, M, N);
  if (!compare_u32_matrices(out, exp_, M, N)) {
    printf("matrix_ws_4x4 FAILED\n");
    return 1;
  }
  printf("matrix_ws_4x4 PASSED\n");
  return 0;
}
