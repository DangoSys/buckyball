#include "matrix_test_common.h"

#define DIM 4

static elem_t a0[DIM * DIM] __attribute__((aligned(64))) = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
};
static elem_t b0[DIM * DIM] __attribute__((aligned(64))) = {
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
};
static elem_t a1[DIM * DIM] __attribute__((aligned(64))) = {
    2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2,
};
static elem_t b1[DIM * DIM] __attribute__((aligned(64))) = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};
static elem_t pa[16 * MATRIX_TILE] __attribute__((aligned(64)));
static elem_t pb[16 * MATRIX_TILE] __attribute__((aligned(64)));
static result_t pc[DIM * MATRIX_ACC_LANES] __attribute__((aligned(64)));
static result_t out[DIM * DIM] __attribute__((aligned(64)));
static result_t exp_[DIM * DIM] __attribute__((aligned(64)));

static int run_one(const char *name, elem_t *a, elem_t *b, int ws) {
  cpu_matmul(a, b, exp_, DIM, DIM, DIM);
  matrix_pack_a(a, pa, DIM, DIM);
  matrix_pack_b(b, pb, DIM, DIM);
  clear_u32_matrix(pc, DIM, MATRIX_ACC_LANES);
  matrix_hw_mnk(pa, pb, pc, DIM, DIM, DIM, ws);
  matrix_unpack_c(pc, out, DIM, DIM);
  if (!compare_u32_matrices(out, exp_, DIM, DIM)) {
    printf("%s FAILED\n", name);
    return 0;
  }
  printf("%s PASSED\n", name);
  return 1;
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int ok = run_one("mode_switch_os", a0, b0, 0);
  ok = run_one("mode_switch_ws", a1, b1, 1) && ok;
  ok = run_one("mode_switch_os2", a0, b1, 0) && ok;
  return ok ? 0 : 1;
}
