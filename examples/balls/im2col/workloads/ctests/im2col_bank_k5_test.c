#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <string.h>

/* 27x27 k5 -> 1088 rows: fills bank0 (1024) and spills 64 rows to bank1. */
enum {
  ITER = 27,
  K = 5,
  STRIDE = 1,
  PAD = 0,
  LANES = 16,
  N_GROUPS = 2,
  SEED = 0x55,
};
enum {
  OUT_DIM = (ITER + 2 * PAD - K) / STRIDE + 1,
  WINDOWS = OUT_DIM * OUT_DIM,
  KERNEL = K * K,
  M_TILES = (WINDOWS + LANES - 1) / LANES,
  K_TILES = (KERNEL + LANES - 1) / LANES,
  OUT_ROWS = M_TILES * K_TILES * LANES,
  /* mvout(groups>1): depth is rows per group; host gets depth*groups beats. */
  MV_ROWS = BANK_LINES,
  MV_BEATS = MV_ROWS * N_GROUPS,
};
_Static_assert(OUT_ROWS > BANK_LINES, "k5 bank test must spill a full bank");

static elem_t in[ITER * ITER] __attribute__((aligned(64)));
static elem_t mv[MV_BEATS * LANES] __attribute__((aligned(64)));
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

static void deinterleave_mvout(void) {
  for (int i = 0; i < MV_ROWS; i++) {
    memcpy(&out[i * LANES], &mv[(i * N_GROUPS) * LANES], LANES);
    if (BANK_LINES + i < OUT_ROWS) {
      memcpy(&out[(BANK_LINES + i) * LANES], &mv[(i * N_GROUPS + 1) * LANES],
             LANES);
    }
  }
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_i8_random_matrix(in, ITER, ITER, SEED);
  clear_i8_matrix(out, OUT_ROWS, LANES);
  clear_i8_matrix(mv, MV_BEATS, LANES);
  build_expected();

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, BANK_LINES, N_GROUPS);
  bb_mvin((uintptr_t)in, 0, (ITER * ITER + LANES - 1) / LANES, 1);
  bb_im2col(0, 1, ITER, K, STRIDE, PAD);
  bb_mvout((uintptr_t)mv, 1, MV_ROWS, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);
  deinterleave_mvout();

  if (compare_i8_matrices(out, exp, OUT_ROWS, LANES)) {
    printf("im2col bank k5 27x27 PASSED\n");
    return 0;
  }
  printf("im2col bank k5 27x27 FAILED\n");
  return 1;
}
