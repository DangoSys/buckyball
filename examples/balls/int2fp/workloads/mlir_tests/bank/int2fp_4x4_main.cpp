#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const float expected[16] = {
    0.125f, 0.25f,  0.375f, -0.125f, -0.25f, 0.0f, 0.5f, 0.625f,
    1.25f,  -1.25f, 0.875f, 12.5f,   -12.5f, 1.0f, 2.0f, -1.0f,
};

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
  if (rows != 4 || columns != 4 || row_stride != 4 || column_stride != 1)
    fail();
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      int index = row * 4 + column;
      float actual =
          aligned[offset + row * row_stride + column * column_stride];
      if (actual != expected[index]) {
        printf("FAILED: int32_to_fp32 index=%d expected=%f actual=%f\n", index,
               expected[index], actual);
        fail();
      }
    }
  }
  printf("PASSED: int32_to_fp32 scaled conversion\n");
}
