#include "matrix_test_common.h"

#define M 64
#define N 16
#define K 25

static elem_t a[M * K] __attribute__((aligned(64)));
static elem_t b[K * N] __attribute__((aligned(64)));
static elem_t pa[128 * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb[32 * MATRIX_TILE] __attribute__((aligned(64)));
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
  printf("matrix_os_64x16x25_slim start\n");
  matrix_hw_mnk(pa, pb, pc, M, N, K, 0);
  printf("matrix_os_64x16x25_slim DONE\n");
  return 0;
}
