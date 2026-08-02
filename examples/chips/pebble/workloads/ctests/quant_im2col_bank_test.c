#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <bbhw/mem/params.h>
#include <stdio.h>

/* Same bank-fill shape as conv_im2col_bank; different seed. */
enum { ITER = 34, K = 3, STRIDE = 1, PAD = 0, LANES = 16, SEED = 0xC6 };
enum {
  OUT_DIM = (ITER + 2 * PAD - K) / STRIDE + 1,
  WINDOWS = OUT_DIM * OUT_DIM,
  KERNEL = K * K,
  M_TILES = (WINDOWS + LANES - 1) / LANES,
  K_TILES = (KERNEL + LANES - 1) / LANES,
  OUT_ROWS = M_TILES * K_TILES * LANES,
};
_Static_assert(OUT_ROWS == BANK_LINES, "bank test must fill one bank");

static elem_t in[ITER * ITER] __attribute__((aligned(64)));
static elem_t out[OUT_ROWS * LANES] __attribute__((aligned(64)));
static elem_t exp[OUT_ROWS * LANES] __attribute__((aligned(64)));

static void build_expected(void) {
  int w = 0;
  clear_i8_matrix(exp, OUT_ROWS, LANES);
  for (int orow = 0; orow < OUT_DIM; orow++) {
    for (int ocol = 0; ocol < OUT_DIM; ocol++) {
      for (int krow = 0; krow < K; krow++) {
        for (int kcol = 0; kcol < K; kcol++) {
          int ki = krow * K + kcol;
          int bank_row =
              ((w / LANES) * K_TILES + ki / LANES) * LANES + (w % LANES);
          int row = orow * STRIDE + krow - PAD;
          int col = ocol * STRIDE + kcol - PAD;
          exp[bank_row * LANES + ki % LANES] =
              (row < 0 || row >= ITER || col < 0 || col >= ITER)
                  ? 0
                  : in[row * ITER + col];
        }
      }
      w++;
    }
  }
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_i8_random_matrix(in, ITER, ITER, SEED);
  clear_i8_matrix(out, OUT_ROWS, LANES);
  build_expected();
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)in, 0, (ITER * ITER + LANES - 1) / LANES, 1);
  bb_im2col(0, 1, ITER, K, STRIDE, PAD);
  bb_mvout((uintptr_t)out, 1, OUT_ROWS, 1);
  bb_fence();
  int ok = compare_i8_matrices(out, exp, OUT_ROWS, LANES);
  printf("quant_im2col_bank %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
