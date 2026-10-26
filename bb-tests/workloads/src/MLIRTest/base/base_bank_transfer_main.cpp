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
  if (size0 != 2 || size1 != 48 || stride0 != 48 || stride1 != 1) {
    fail();
  }
  int8_t *out = aligned + offset;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 48; ++j) {
      int8_t expected =
          j < 16 ? (int8_t)(i * 16 + j) : (int8_t)(i * 32 + j - 16 + 64);
      if (out[i * stride0 + j * stride1] != expected) {
        fail();
      }
    }
  }
  printf("PASSED: base bank_transfer appends groups\n");
}
