#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(void) {
#ifdef BAREMETAL
  volatile uint32_t *sim_exit = (volatile uint32_t *)0x60000000;
  *sim_exit = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

#ifdef __cplusplus
extern "C"
#endif
    void check_result(int32_t *allocated, int32_t *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t stride0,
                      int64_t stride1) {
  (void)allocated;

  if (size0 != 16 || size1 != 16 || stride0 != 16 || stride1 != 1) {
    printf("FAILED: buckyball matmul 16x64 unexpected memref shape "
           "(size=%dx%d stride=%dx%d)\n",
           (int)size0, (int)size1, (int)stride0, (int)stride1);
    fail();
  }

  int32_t *c = aligned + offset;
  const int32_t expected = 0x42800000;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t got = c[i * stride0 + j * stride1];
      if (got != expected) {
        printf("FAILED: buckyball matmul 16x64 c[%d][%d] "
               "(expected 0x%08x, got 0x%08x)\n",
               i, j, expected, got);
        fail();
      }
    }
  }

  printf("PASSED: buckyball matmul 16x64 @ 64x16\n");
}
