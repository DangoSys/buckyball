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

extern "C" void check_result(float *allocated, float *aligned, int64_t offset,
                             int64_t rows, int64_t columns, int64_t row_stride,
                             int64_t column_stride) {
  (void)allocated;
  if (rows != 1 || columns != 16 || row_stride != 16 || column_stride != 1)
    fail();
  for (int column = 0; column < 16; ++column) {
    float actual = aligned[offset + column * column_stride];
    if (actual != 16.0f) {
      printf("FAILED: mega_matmul column=%d expected=16 actual=%f\n", column,
             actual);
      fail();
    }
  }
  printf("PASSED: mega_matmul INT8 to FP32\n");
}
