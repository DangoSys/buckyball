#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// 6x6 input, k3, stride 1: 16 windows; nine K tiles (one per kernel
// position, all sixteen channels) in M tiles of eight rows.
enum { SIDE = 6, K = 3, OUT = 4, TILE_ROWS = 8, TILES = 9, LINES = 144 };

static void fail(void) {
#ifdef BAREMETAL
  *(volatile uint32_t *)0x60000000 = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

extern "C" void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t row_stride,
                             int64_t lane_stride) {
  (void)allocated;
  if (rows != LINES || lanes != 16 || row_stride != 16 || lane_stride != 1)
    fail();
  int8_t *out = aligned + offset;
  for (int w = 0; w < OUT * OUT; ++w)
    for (int t = 0; t < TILES; ++t)
      for (int ch = 0; ch < 16; ++ch) {
        int y = w / OUT + t / K, x = w % OUT + t % K;
        int line = ((w / TILE_ROWS) * TILES + t) * TILE_ROWS + w % TILE_ROWS;
        int8_t expected = (int8_t)(y + x + ch);
        int8_t actual = out[line * row_stride + ch];
        if (actual != expected) {
          printf("FAILED: im2col 6x6 k3 w=%d t=%d ch=%d expected=%d "
                 "actual=%d\n",
                 w, t, ch, expected, actual);
          fail();
        }
      }
  printf("PASSED: im2col 6x6 k3\n");
}
