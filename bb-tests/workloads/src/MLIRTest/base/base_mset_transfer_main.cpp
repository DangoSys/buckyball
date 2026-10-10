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
    void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                      int64_t size0, int64_t size1, int64_t stride0,
                      int64_t stride1) {
  (void)allocated;
  if (size0 != 16 || size1 != 32 || stride0 != 32 || stride1 != 1) {
    fail();
  }
  int8_t *out = aligned + offset;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int value = i * 16 + j;
      if (out[i * stride0 + j] != (int8_t)value ||
          out[i * stride0 + 16 + j] != (int8_t)(127 - value)) {
        fail();
      }
    }
  }
  printf("PASSED: base mset transfer preserves group order\n");
}
