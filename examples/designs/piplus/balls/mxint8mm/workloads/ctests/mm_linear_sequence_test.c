#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8mm.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* The instruction sequence of a pi0 vision Linear on a dense thread, as the
 * compiler emits it: A tiles (43 rows, K 608) in banks 1.., B (48 rows)
 * in bank 6, the bias in bank 7; per tile the bias moves in again, mxint8mm
 * writes C behind it in bank 7 and two block move outs take C to DDR rows
 * 4608 bytes apart. Two tiles of the same A (pi0 hung on the second): the
 * outputs must be equal byte for byte. */
#define M 43
#define N 48
#define K 608
#define ROWS (MXINT8_ROWS(K))
#define PITCH (N / 4)
#define STRIDE 288
#define TILES 2

static uint8_t a[M * ROWS * 16] __attribute__((aligned(64)));
static uint8_t b[N * ROWS * 16] __attribute__((aligned(64)));
static float bias[N] __attribute__((aligned(64)));
/* Outside .bss: zeroing 400 KB at the slow start of a dense hart took a
 * million cycles. Only the C bytes are compared. */
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
  /* Codes of small magnitude, written a word (8 codes) at a time: the CPU of
   * a dense hart is slow, a byte loop over A and B took longer than the run. */
  for (unsigned m = 0; m < M; ++m) {
    uint64_t *row = (uint64_t *)&a[m * ROWS * 16];
    for (unsigned w = 0; w < K / 8; ++w)
      row[w] = UINT64_C(0x0403fe0201fffd05) ^
               (UINT64_C(0x0101010101010101) * ((m + w) & 3));
    memset(&a[m * ROWS * 16 + K], 126, K / 32);
  }
  for (unsigned n = 0; n < N; ++n) {
    uint64_t *row = (uint64_t *)&b[n * ROWS * 16];
    for (unsigned w = 0; w < K / 8; ++w)
      row[w] = UINT64_C(0xfd0302fc0105ff02) ^
               (UINT64_C(0x0101010101010101) * ((n * 3 + w) & 3));
    memset(&b[n * ROWS * 16 + K], 127, K / 32);
  }
  for (unsigned n = 0; n < N; ++n)
    bias[n] = (float)n - 20.0f;
  bb_thread_run(run, 0);
  for (unsigned t = 1; t < TILES; ++t)
    for (unsigned m = 0; m < M; ++m)
      if (memcmp(&out[t][m * STRIDE * 16], &out[0][m * STRIDE * 16],
                 PITCH * 16) != 0) {
        printf("mm linear sequence: tile %u row %u differs from tile 0\n", t,
               m);
        return 1;
      }
  printf("mm linear sequence PASSED\n");
  return 0;
}
