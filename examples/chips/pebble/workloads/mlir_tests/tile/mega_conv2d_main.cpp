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
  for (int channel = 0; channel < 16; ++channel) {
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        int actual = output[channel * channel_stride + y * height_stride +
                            x * width_stride];
        if (actual != 127) {
          printf("FAILED: mega_conv2d c=%d y=%d x=%d expected=127 actual=%d\n",
                 channel, y, x, actual);
          fail();
        }
      }
    }
  }
  printf("PASSED: mega_conv2d Conv-to-Depthwise-to-MaxPool resident chain\n");
}
