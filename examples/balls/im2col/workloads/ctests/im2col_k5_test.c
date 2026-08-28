#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/im2col.h>
#include <stdio.h>

enum { ITER = 7, K = 5, STRIDE = 1, PAD = 0, LANES = 16 };
enum {
  OUT_DIM = (ITER + 2 * PAD - K) / STRIDE + 1,
  WINDOWS = OUT_DIM * OUT_DIM,
  KERNEL = K * K,
  M_TILES = (WINDOWS + LANES - 1) / LANES,
  K_TILES = (KERNEL + LANES - 1) / LANES,
  OUT_ROWS = M_TILES * K_TILES * LANES,
};

static elem_t in[ITER * ITER] __attribute__((aligned(64))) = {
    2,   -4,  6,  0,  8,   -10, 12, -1,  3,  5,   7,  0,   11, -13, 14, 0,   16,
    -18, 20,  22, 24, -15, 17,  19, 0,   21, -23, 25, 26,  28, 0,   30, -32, 34,
    36,  -27, 29, 31, 33,  0,   35, -37, 38, 0,   40, -42, 44, 46,  48,
};
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
  clear_i8_matrix(out, OUT_ROWS, LANES);
  build_expected();

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)in, 0, (ITER * ITER + LANES - 1) / LANES, 1);
  bb_im2col(0, 1, ITER, K, STRIDE, PAD);
  bb_mvout((uintptr_t)out, 1, OUT_ROWS, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);

  if (compare_i8_matrices(out, exp, OUT_ROWS, LANES)) {
    printf("im2col k5 7x7 PASSED\n");
    return 0;
  }
  printf("im2col k5 7x7 FAILED\n");
  return 1;
}
