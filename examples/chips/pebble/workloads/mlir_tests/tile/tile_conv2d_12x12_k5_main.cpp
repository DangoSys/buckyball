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

#ifdef __cplusplus
extern "C"
#endif
    void check_result(float *allocated, float *aligned, int64_t offset,
                      int64_t n, int64_t h, int64_t w, int64_t c, int64_t sn,
                      int64_t sh, int64_t sw, int64_t sc) {
  (void)allocated;
  if (n != 1 || h != 8 || w != 8 || c != 1 || sn != 64 || sh != 8 || sw != 1 ||
      sc != 1)
    fail();
  // quant scale=1: each window sums 5*5 ones -> 25
  for (int i = 0; i < 64; ++i) {
    if (aligned[offset + i] != 25.0f)
      fail();
  }
  printf("PASSED: tile.tile_conv2d 12x12 k5\n");
}
