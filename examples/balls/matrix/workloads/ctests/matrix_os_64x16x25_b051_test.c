#include "matrix_test_common.h"

#define M 64
#define N 16
#define K 25

static elem_t a[M * K] __attribute__((aligned(64)));
static elem_t b[K * N] __attribute__((aligned(64)));
static elem_t pa[128 * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb[32 * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc[M * MATRIX_ACC_LANES] __attribute__((aligned(64)));

static void matrix_hw_banks(const elem_t *packed_a, const elem_t *packed_b,
                            result_t *packed_c, int m, int n, int k,
                            uint32_t op1, uint32_t op2, uint32_t wr) {
  int a_rows = matrix_a_rows(m, k);
  int b_rows = matrix_b_rows(n, k);
  int c_blocks = matrix_c_blocks(m, n);

  bb_mem_alloc(op1, 1, 1);
  bb_mem_alloc(op2, 1, 1);
  bb_mem_alloc(wr, 1, 4);
  bb_mvin((uintptr_t)packed_a, op1, a_rows, 1);
  bb_mvin((uintptr_t)packed_b, op2, b_rows, 1);
  bb_matrix_mnk_mode(op1, op2, wr, m, n, k, BB_MATRIX_MODE_OS);
  bb_mvout((uintptr_t)packed_c, wr, c_blocks, 1);
  bb_fence();
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_ones_matrix(a, M, K);
  init_ones_matrix(b, K, N);
  matrix_pack_a(a, pa, M, K);
  matrix_pack_b(b, pb, K, N);
  clear_u32_matrix(pc, M, MATRIX_ACC_LANES);
  printf("matrix_os_64x16x25_b051 start\n");
  matrix_hw_banks(pa, pb, pc, M, N, K, 0, 5, 1);
  printf("matrix_os_64x16x25_b051 DONE\n");
  return 0;
}
