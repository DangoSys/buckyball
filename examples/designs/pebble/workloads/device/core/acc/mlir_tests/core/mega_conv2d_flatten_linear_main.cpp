#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(void) {
#ifdef BAREMETAL
  *(volatile uint32_t *)0x60000000 = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

static int clamp8(int value) {
  return value < -128 ? -128 : (value > 127 ? 127 : value);
}

extern "C" void check_result(float *allocated, float *aligned, int64_t offset,
                             int64_t rows, int64_t columns, int64_t row_stride,
                             int64_t column_stride) {
  (void)allocated;
  if (rows != 1 || columns != 16 || row_stride != 16 || column_stride != 1)
    fail();
  int conv[8][8][16];
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      for (int c = 0; c < 16; ++c) {
        int sum = c;
        for (int ky = 0; ky < 3; ++ky)
          for (int kx = 0; kx < 3; ++kx)
            for (int ic = 0; ic < 2; ++ic) {
              int iy = y + ky - 1, ix = x + kx - 1;
              if (iy >= 0 && iy < 8 && ix >= 0 && ix < 8)
                sum += ((3 * iy + 2 * ix + ic) % 7 - 3) *
                       ((3 * ky + kx + ic + 3 * c) % 5 - 2);
            }
        conv[y][x][c] = sum < 0 ? 0 : clamp8(sum);
      }
  // The pooled map in channel-major order.
  int flat[256];
  for (int c = 0; c < 16; ++c)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x) {
        int maximum = -128;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            if (conv[2 * y + dy][2 * x + dx][c] > maximum)
              maximum = conv[2 * y + dy][2 * x + dx][c];
        flat[c * 16 + y * 4 + x] = maximum;
      }
  float *output = aligned + offset;
  for (int n = 0; n < 16; ++n) {
    int sum = n;
    for (int k = 0; k < 256; ++k)
      sum += flat[k] * ((k + n) % 5 - 2);
    if (output[n * column_stride] != (float)sum) {
      printf("FAILED: mega_conv2d_flatten_linear n=%d exp=%d got=%f\n", n, sum,
             output[n * column_stride]);
      fail();
    }
  }
  printf("PASSED: mega_conv2d_flatten_linear resident map to Linear\n");
}
