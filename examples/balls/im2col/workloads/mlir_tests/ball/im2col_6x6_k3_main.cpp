#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { ITER = 6, K = 3, STRIDE = 1, PAD = 0, LANES = 16 };
enum {
  OUT_DIM = (ITER + 2 * PAD - K) / STRIDE + 1,
  WINDOWS = OUT_DIM * OUT_DIM,
  KERNEL = K * K,
  M_TILES = (WINDOWS + LANES - 1) / LANES,
  K_TILES = (KERNEL + LANES - 1) / LANES,
  OUT_ROWS = M_TILES * K_TILES * LANES,
};

static void fail(void) {
#ifdef BAREMETAL
  volatile uint32_t *sim_exit = (volatile uint32_t *)0x60000000;
  *sim_exit = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

static int8_t src(int row, int col) {
  if (row < 0 || row >= ITER || col < 0 || col >= ITER)
    return 0;
  return (int8_t)(row + col);
}

#ifdef __cplusplus
extern "C"
#endif
    void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t stride0,
                      int64_t stride1) {
  (void)allocated;
  if (size0 != OUT_ROWS || size1 != LANES || stride0 != LANES || stride1 != 1) {
    printf("FAILED: im2col ball_op unexpected shape "
           "(size=%ldx%ld stride=%ldx%ld)\n",
           (long)size0, (long)size1, (long)stride0, (long)stride1);
    fail();
  }

  int8_t *out = aligned + offset;
  int w = 0;
  for (int orow = 0; orow < OUT_DIM; orow++) {
    for (int ocol = 0; ocol < OUT_DIM; ocol++) {
      for (int krow = 0; krow < K; krow++) {
        for (int kcol = 0; kcol < K; kcol++) {
          int ki = krow * K + kcol;
          int bank_row =
              ((w / LANES) * K_TILES + ki / LANES) * LANES + (w % LANES);
          int8_t exp =
              src(orow * STRIDE + krow - PAD, ocol * STRIDE + kcol - PAD);
          int8_t got = out[bank_row * stride0 + (ki % LANES) * stride1];
          if (got != exp) {
            printf("FAILED: im2col ball_op w=%d ki=%d "
                   "(expected %d, got %d)\n",
                   w, ki, (int)exp, (int)got);
            fail();
          }
        }
      }
      w++;
    }
  }

  printf("PASSED: im2col ball_op 6x6 k3\n");
}
