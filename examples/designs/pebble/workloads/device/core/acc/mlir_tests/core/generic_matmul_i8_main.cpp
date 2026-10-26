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

extern "C" void check_result(int32_t *, int32_t *output, int64_t offset,
                             int64_t rows, int64_t columns, int64_t row_stride,
                             int64_t column_stride) {
  if (rows != 1 || columns != 32 || row_stride != 32 || column_stride != 1)
    fail();
  for (int column = 0; column < 32; ++column) {
    int32_t expected = column - 16;
    for (int k = 0; k < 32; ++k)
      expected += (k % 7 - 3) * ((3 * k + 5 * column) % 9 - 4);
    int32_t actual = output[offset + column * column_stride];
    if (actual != expected) {
      printf("FAILED: generic matmul column=%d expected=%d actual=%d\n", column,
             expected, actual);
      fail();
    }
  }
  printf("PASSED: generic matmul INT8 signed inputs and nonzero INT32 "
         "accumulator\n");
}
