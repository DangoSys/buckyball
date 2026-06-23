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

  if (size0 != 1 || size1 != 10 || stride0 != 10 || stride1 != 1) {
    printf("FAILED: tile matmul 1x84 unexpected memref shape "
           "(size=%dx%d stride=%dx%d)\n",
           (int)size0, (int)size1, (int)stride0, (int)stride1);
    fail();
  }

  int32_t *c = aligned + offset;
  const int32_t expected = 0x42a80000;
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < 10; ++j) {
      int32_t got = c[i * stride0 + j * stride1];
      if (got != expected) {
        printf("FAILED: tile matmul 1x84 c[%d][%d] "
               "(expected 0x%08x, got 0x%08x)\n",
               i, j, expected, got);
        fail();
      }
    }
  }

  printf("PASSED: tile matmul 1x84 @ 84x10\n");
}
