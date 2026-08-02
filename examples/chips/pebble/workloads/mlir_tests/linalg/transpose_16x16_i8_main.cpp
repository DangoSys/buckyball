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
    void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t stride0,
                      int64_t stride1) {
  (void)allocated;
  if (size0 != 16 || size1 != 16 || stride0 != 16 || stride1 != 1)
    fail();
  int8_t *out = aligned + offset;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      if (out[i * stride0 + j * stride1] != (int8_t)(i + j))
        fail();
  printf("PASSED: linalg.transpose 16x16 i8\n");
}
