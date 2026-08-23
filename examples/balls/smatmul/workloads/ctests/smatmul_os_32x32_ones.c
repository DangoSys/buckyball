#include "smatmul_test_common.h"

#define DIM 32
#define PANEL 16

static elem_t a[DIM * DIM] __attribute__((aligned(64)));
static elem_t b[DIM * DIM] __attribute__((aligned(64)));
static elem_t pa[64 * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t bpanel[DIM * PANEL] __attribute__((aligned(64)));
static elem_t pb[32 * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc[DIM * MATRIX_ACC_LANES] __attribute__((aligned(64)));
static result_t panel_out[DIM * PANEL] __attribute__((aligned(64)));
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
  for (int n0 = 0; n0 < DIM; n0 += PANEL) {
    for (int r = 0; r < DIM; ++r)
      for (int c = 0; c < PANEL; ++c)
        bpanel[r * PANEL + c] = b[r * DIM + n0 + c];
    matrix_pack_b(bpanel, pb, DIM, PANEL);
    clear_u32_matrix(pc, DIM, MATRIX_ACC_LANES);
    matrix_hw_ws(pa, pb, pc, DIM, PANEL, DIM);
    matrix_unpack_c(pc, panel_out, DIM, PANEL);
    for (int r = 0; r < DIM; ++r)
      for (int c = 0; c < PANEL; ++c)
        out[r * DIM + n0 + c] = panel_out[r * PANEL + c];
  }
  if (!compare_u32_matrices(out, exp_, DIM, DIM)) {
    printf("smatmul_os_32x32_ones FAILED\n");
    return 1;
  }
  printf("smatmul_os_32x32_ones PASSED\n");
  return 0;
}
