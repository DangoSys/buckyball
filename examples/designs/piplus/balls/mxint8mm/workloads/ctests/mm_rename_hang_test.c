#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8mm.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* mm_linear_sequence_test without operand setup, the shortest run of its
 * instruction sequence: on the dense hart the second mxint8mm, behind a block
 * move out of bank 7 and a renaming move in to it, never stored its last
 * output tile. A and B are whatever DDR holds; the two tiles read the same A,
 * so their C must still be equal byte for byte. */
#define M 43
#define N 48
#define K 608
#define ROWS (MXINT8_ROWS(K))
#define PITCH (N / 4)
#define STRIDE 288
#define TILES 2

static uint8_t a[M * ROWS * 16]
    __attribute__((section(".noinit"), aligned(64)));
static uint8_t b[N * ROWS * 16]
    __attribute__((section(".noinit"), aligned(64)));
static float bias[N] __attribute__((aligned(64)));
static uint8_t out[TILES][M * STRIDE * 16]
    __attribute__((section(".noinit"), aligned(4096)));

static void run(void *unused) {
  (void)unused;
  for (unsigned bank = 1; bank <= 7; ++bank)
    bb_mem_alloc(bank, 1, 1);
  for (unsigned t = 0; t < TILES; ++t)
    bb_mvin_group((uintptr_t)a, 1 + t, 0, M * ROWS, 1);
  bb_mvin_group((uintptr_t)b, 6, 0, N * ROWS, 1);
  for (unsigned t = 0; t < TILES; ++t) {
    bb_mvin_group((uintptr_t)bias, 7, 0, PITCH, 1);
    bb_mxint8mm(1 + t, 6, 7, M, N, K, 0, PITCH, PITCH, 0, 1, 0);
    bb_mvout_blocks((uintptr_t)out[t], 7, 0, PITCH, PITCH, M * 8, STRIDE, 3);
    bb_mvout_blocks((uintptr_t)out[t] + 128, 7, 0, PITCH + 8, PITCH, M * 4,
                    STRIDE, 2);
  }
  for (unsigned bank = 1; bank <= 7; ++bank)
    bb_mem_release(bank);
}

int main(void) {
  bb_thread_run(run, 0);
  for (unsigned t = 1; t < TILES; ++t)
    for (unsigned m = 0; m < M; ++m)
      if (memcmp(&out[t][m * STRIDE * 16], &out[0][m * STRIDE * 16],
                 PITCH * 16) != 0) {
        printf("mm rename hang: tile %u row %u differs from tile 0\n", t, m);
        return 1;
      }
  printf("mm rename hang PASSED\n");
  return 0;
}
