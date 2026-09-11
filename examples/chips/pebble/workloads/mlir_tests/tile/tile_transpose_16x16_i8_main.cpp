#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(void) {
#ifdef BAREMETAL
  *(volatile uint32_t *)0x60000000 = 1;
  while (1) {}
#else
  exit(1);
#endif
}

extern "C" void check_result(int8_t *, int8_t *out, int64_t offset,
                             int64_t rows, int64_t columns,
                             int64_t row_stride, int64_t column_stride) {
  if (rows != 16 || columns != 16 || row_stride != 16 || column_stride != 1)
    fail();
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      if (out[offset + i * row_stride + j] != (int8_t)(3 * j + 5 * i))
        fail();
  printf("PASSED: tile.tile_transpose 16x16 ordered input\n");
}
