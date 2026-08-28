#include "smatmul_test_common.h"

#define M 48
#define N 16
#define K 16

static elem_t a[M * K] __attribute__((aligned(64)));
static elem_t b[K * N] __attribute__((aligned(64)));
static elem_t pa[48 * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb[16 * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc[M * MATRIX_ACC_LANES] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_ones_matrix(a, M, K);
  init_ones_matrix(b, K, N);
  matrix_pack_a(a, pa, M, K);
  matrix_pack_b(b, pb, K, N);
  clear_u32_matrix(pc, M, MATRIX_ACC_LANES);
  printf("smatmul_os_48x16 start\n");
  matrix_hw_ws(pa, pb, pc, M, N, K);
  printf("smatmul_os_48x16 DONE\n");
  return 0;
}
