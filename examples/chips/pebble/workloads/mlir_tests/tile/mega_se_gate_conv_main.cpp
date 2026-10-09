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
                             int64_t n, int64_t height, int64_t width,
                             int64_t channels, int64_t n_stride,
                             int64_t height_stride, int64_t width_stride,
                             int64_t channel_stride) {
  (void)allocated;
  if (n != 1 || height != 6 || width != 6 || channels != 40 ||
      n_stride != 1440 || height_stride != 240 || width_stride != 40 ||
      channel_stride != 1)
    fail();
  const int8_t *output = aligned + offset;
  for (int y = 0; y < 6; ++y)
    for (int x = 0; x < 6; ++x)
      for (int c = 0; c < 40; ++c)
        if (output[y * height_stride + x * width_stride + c] != 2)
          fail();
  printf("PASSED: mega_se_gate_conv natural resident chain\n");
}
