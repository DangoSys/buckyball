#include "smatmul_test_common.h"
#include <params.h>

#define M BANK_LINES
#define N MATRIX_TILE
#define K MATRIX_TILE
#define SEED_A 0x51
#define SEED_B 0xA3

_Static_assert(M >= BANK_LINES && K == MATRIX_TILE,
               "bank OS test must cover a full bank of A rows");

static elem_t a[M * K] __attribute__((aligned(64)));
static elem_t b[K * N] __attribute__((aligned(64)));
static elem_t pa[BANK_LINES * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb[MATRIX_TILE * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc[BANK_LINES * MATRIX_ACC_LANES] __attribute__((aligned(64)));
static result_t out[M * N] __attribute__((aligned(64)));
static result_t exp_[M * N] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_u8_random_matrix(a, M, K, SEED_A);
  init_u8_random_matrix(b, K, N, SEED_B);
  cpu_matmul(a, b, exp_, M, N, K);
  matrix_pack_a(a, pa, M, K);
  matrix_pack_b(b, pb, K, N);
  clear_u32_matrix(pc, matrix_c_blocks(M, N), MATRIX_ACC_LANES);
  matrix_hw_ws(pa, pb, pc, M, N, K);
  matrix_unpack_c(pc, out, M, N);
  if (!compare_u32_matrices(out, exp_, M, N)) {
    printf("smatmul_bank_os_1024x16x16 FAILED\n");
    return 1;
  }
  printf("smatmul_bank_os_1024x16x16 PASSED\n");
  return 0;
}
