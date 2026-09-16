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
  static const int8_t expected[24] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0};
  if (n != 1 || height != 1 || width != 1 || channels != 24 || n_stride != 24 ||
      height_stride != 24 || width_stride != 24 || channel_stride != 1)
    fail();
  const int8_t *output = aligned + offset;
  for (int i = 0; i < 24; ++i)
    if (output[i] != expected[i]) {
      printf("FAILED: mega_global_avg_conv_exact i=%d got=%d expected=%d\n", i,
             output[i], expected[i]);
      fail();
    }
  printf("PASSED: mega_global_avg_conv_exact\n");
}
