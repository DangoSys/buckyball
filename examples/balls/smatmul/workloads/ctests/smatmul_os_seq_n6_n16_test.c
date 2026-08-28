#include "smatmul_test_common.h"
#include <isa/smatmul.h>

#define M1 48
#define N1 6
#define K1 25
#define M2 64
#define N2 16
#define K2 25

static elem_t a1[M1 * K1] __attribute__((aligned(64)));
static elem_t b1[K1 * N1] __attribute__((aligned(64)));
static elem_t pa1[96 * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb1[32 * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc1[M1 * MATRIX_ACC_LANES] __attribute__((aligned(64)));

static elem_t a2[M2 * K2] __attribute__((aligned(64)));
static elem_t b2[K2 * N2] __attribute__((aligned(64)));
static elem_t pa2[128 * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb2[32 * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc2[M2 * MATRIX_ACC_LANES] __attribute__((aligned(64)));

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
  bb_smatmul_ws(op1, op2, wr, m, n, k);
  bb_mvout((uintptr_t)packed_c, wr, c_blocks, 1);
  bb_fence();
  bb_mem_release(op1);
  bb_mem_release(op2);
  bb_mem_release(wr);
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_ones_matrix(a1, M1, K1);
  init_ones_matrix(b1, K1, N1);
  matrix_pack_a(a1, pa1, M1, K1);
  matrix_pack_b(b1, pb1, K1, N1);
  clear_u32_matrix(pc1, M1, MATRIX_ACC_LANES);

  init_ones_matrix(a2, M2, K2);
  init_ones_matrix(b2, K2, N2);
  matrix_pack_a(a2, pa2, M2, K2);
  matrix_pack_b(b2, pb2, K2, N2);
  clear_u32_matrix(pc2, M2, MATRIX_ACC_LANES);

  printf("smatmul_os_seq_n6_n16 start\n");
  matrix_hw_banks(pa1, pb1, pc1, M1, N1, K1, 0, 5, 1);
  printf("smatmul_os_seq_n6_n16 after_n6\n");
  matrix_hw_banks(pa2, pb2, pc2, M2, N2, K2, 0, 5, 1);
  printf("smatmul_os_seq_n6_n16 DONE\n");
  return 0;
}
