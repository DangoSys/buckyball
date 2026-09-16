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
  if (n != 1 || height != 14 || width != 14 || channels != 40 ||
      n_stride != 7840 || height_stride != 560 || width_stride != 40 ||
      channel_stride != 1)
    fail();
  const int8_t *output = aligned + offset;
  uint64_t hash = 1469598103934665603ULL;
  for (int y = 0; y < 14; ++y)
    for (int x = 0; x < 14; ++x)
      for (int c = 0; c < 40; ++c) {
        hash ^= (uint8_t)output[y * height_stride + x * width_stride + c];
        hash *= 1099511628211ULL;
      }
  printf("mega_se_gate_conv hash=%08x%08x\n", (uint32_t)(hash >> 32),
         (uint32_t)hash);
  if (hash != 0x2069d0879c6c7f83ULL) {
    printf("FAILED: expected=2069d0879c6c7f83\n");
    fail();
  }
  printf("PASSED: mega_se_gate_conv natural resident chain\n");
}
