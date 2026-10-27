#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// 8x8 map, two channel blocks, k3 padding 1: K tile t = (ky * 3 + kx) * 2 +
// block; output group j holds tiles 6j..6j+5, line ((w / 8) * 6 + t % 6) * 8
// + w % 8. A whole-bank DMA puts the three groups side by side in a row.
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != 384 || lanes != 48 || stride != 48 || lane_stride != 1)
    exit(1);
  for (int w = 0; w < 64; ++w)
    for (int t = 0; t < 18; ++t)
      for (int ch = 0; ch < 16; ++ch) {
        int position = t / 2, block = t % 2;
        int y = w / 8 + position / 3 - 1, x = w % 8 + position % 3 - 1;
        int expected = 0;
        if (y >= 0 && y < 8 && x >= 0 && x < 8)
          expected = y * 8 + x + (block * 16 + ch) * 3;
        int line = ((w / 8) * 6 + t % 6) * 8 + w % 8;
        int lane = (t / 6) * 16 + ch;
        if (data[offset + line * stride + lane] != (int8_t)expected) {
          printf("im2col two blocks mismatch w=%d t=%d ch=%d\n", w, t, ch);
          exit(1);
        }
      }
  printf("im2col k3 two blocks PASS\n");
}
