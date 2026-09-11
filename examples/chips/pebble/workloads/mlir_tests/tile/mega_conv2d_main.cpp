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

extern "C" void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                             int64_t n, int64_t channels, int64_t height,
                             int64_t width, int64_t n_stride,
                             int64_t channel_stride, int64_t height_stride,
                             int64_t width_stride) {
  (void)allocated;
  if (n != 1 || channels != 16 || height != 2 || width != 2 || n_stride != 64 ||
      channel_stride != 4 || height_stride != 2 || width_stride != 1)
    fail();
  int8_t *output = aligned + offset;
  int first[4][4][16];
  int second[4][4][16];
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
      for (int c = 0; c < 16; ++c) {
        int sum = c - 4;
        for (int ky = 0; ky < 3; ++ky)
          for (int kx = 0; kx < 3; ++kx)
            for (int ic = 0; ic < 2; ++ic) {
              int iy = y + ky - 1, ix = x + kx - 1;
              if (iy >= 0 && iy < 4 && ix >= 0 && ix < 4)
                sum += ((3 * iy + 2 * ix + ic) % 7 - 3) *
                       ((3 * ky + kx + ic + 3 * c) % 5 - 2);
            }
        first[y][x][c] = sum < 0 ? 0 : (sum > 127 ? 127 : sum);
      }
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
      for (int c = 0; c < 16; ++c) {
        int sum = 0;
        for (int ky = 0; ky < 3; ++ky)
          for (int kx = 0; kx < 3; ++kx) {
            int iy = y + ky - 1, ix = x + kx - 1;
            if (iy >= 0 && iy < 4 && ix >= 0 && ix < 4)
              sum += first[iy][ix][c] * ((ky + 2 * kx + c) % 5 - 2);
          }
        second[y][x][c] = sum < -128 ? -128 : (sum > 127 ? 127 : sum);
      }
  for (int channel = 0; channel < 16; ++channel) {
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        int expected = -128;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            if (second[2 * y + dy][2 * x + dx][channel] > expected)
              expected = second[2 * y + dy][2 * x + dx][channel];
        int actual = output[channel * channel_stride + y * height_stride + x];
        if (actual != expected) {
          printf("FAILED: mega_conv2d c=%d y=%d x=%d exp=%d got=%d\n",
                 channel, y, x, expected, actual);
          fail();
        }
      }
    }
  }
  printf("PASSED: mega_conv2d Conv-to-Depthwise-to-MaxPool resident chain\n");
}
