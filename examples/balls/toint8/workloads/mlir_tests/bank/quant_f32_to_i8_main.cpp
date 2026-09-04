#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const int8_t expected[16] = {
    -128, -128, -2, -2, 0, 0, 0, 2, 2, 127, 127, 127, 1, 3, 5, 7,
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

extern "C" void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                             int64_t rows, int64_t columns, int64_t row_stride,
                             int64_t column_stride) {
  (void)allocated;
  if (rows != 1 || columns != 16 || row_stride != 16 || column_stride != 1)
    fail();
  for (int i = 0; i < 16; ++i) {
    int8_t actual = aligned[offset + i];
    if (actual != expected[i]) {
      printf("FAILED: quant_f32_to_i8 index=%d expected=%d actual=%d\n", i,
             expected[i], actual);
      fail();
    }
  }
  printf("PASSED: quant_f32_to_i8\n");
}
