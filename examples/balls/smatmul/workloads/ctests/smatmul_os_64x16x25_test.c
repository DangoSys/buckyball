#include "smatmul_test_common.h"

#define M 64
#define N 16
#define K 25

static elem_t a[M * K] __attribute__((aligned(64)));
static elem_t b[K * N] __attribute__((aligned(64)));
static elem_t pa[128 * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb[32 * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc[M * MATRIX_ACC_LANES] __attribute__((aligned(64)));
static result_t out[M * N] __attribute__((aligned(64)));
static result_t exp_[M * N] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_ones_matrix(a, M, K);
  init_ones_matrix(b, K, N);
  cpu_matmul(a, b, exp_, M, N, K);
  matrix_pack_a(a, pa, M, K);
  matrix_pack_b(b, pb, K, N);
  clear_u32_matrix(pc, M, MATRIX_ACC_LANES);
  printf("smatmul_os_64x16x25 start\n");
  matrix_hw_ws(pa, pb, pc, M, N, K);
  matrix_unpack_c(pc, out, M, N);
  if (!compare_u32_matrices(out, exp_, M, N)) {
    printf("smatmul_os_64x16x25 FAILED\n");
    return 1;
  }
  printf("smatmul_os_64x16x25 PASSED\n");
  return 0;
}
