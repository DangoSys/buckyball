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
    void check_result(int32_t *allocated, int32_t *aligned, int64_t offset,
                      int64_t rows, int64_t columns, int64_t rowStride,
                      int64_t columnStride) {
  (void)allocated;
  if (rows != 513 || columns != 17 || rowStride != 17 || columnStride != 1)
    fail();
  for (int row = 0; row < 513; ++row) {
    for (int column = 0; column < 17; ++column) {
      int32_t value = (row + column) % 37 - 18;
      int32_t expected = value < 0 ? 0 : value;
      if (aligned[offset + row * rowStride + column * columnStride] != expected)
        fail();
    }
  }
  printf("PASSED: linalg.relu 513x17 i32\n");
}
