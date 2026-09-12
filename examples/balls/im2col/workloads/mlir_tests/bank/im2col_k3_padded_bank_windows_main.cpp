#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != BANK_LINES || lanes != BANK_WIDTH / 8 || stride != lanes ||
      lane_stride != 1)
    exit(1);
  for (int window = 0; window < BANK_LINES; ++window) {
    int y = window / BANK_ISQRT;
    int x = window % BANK_ISQRT;
    for (int ky = 0; ky < 3; ++ky)
      for (int kx = 0; kx < 3; ++kx) {
        int source_y = y + ky - 1;
        int source_x = x + kx - 1;
        int expected = 0;
        if (source_y >= 0 && source_y < BANK_ISQRT && source_x >= 0 &&
            source_x < BANK_ISQRT)
          expected = source_y * BANK_ISQRT + source_x;
        int lane = ky * 3 + kx;
        if (data[offset + window * stride + lane] != (int8_t)expected) {
          printf("im2col K3 bank mismatch window=%d lane=%d\n", window, lane);
          exit(1);
        }
      }
  }
  printf("im2col K3 padded bank PASS rows=%d\n", BANK_LINES);
}
