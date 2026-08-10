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
                      int64_t m, int64_t n, int64_t sm, int64_t sn) {
  (void)allocated;
  (void)sm;
  (void)sn;
  if (m != 1 || n != 1024)
    fail();
  for (int i = 0; i < 1024; ++i) {
    if (aligned[offset + i] != 576.0f)
      fail();
  }
  printf("PASSED: tile.tile_matmul 1x576x1024 f32\n");
}
