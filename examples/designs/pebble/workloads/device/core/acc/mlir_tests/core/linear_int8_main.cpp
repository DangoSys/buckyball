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

extern "C" void check_result(float *, float *output, int64_t offset,
                             int64_t rows, int64_t columns, int64_t row_stride,
                             int64_t column_stride) {
  if (rows != 1 || columns != 32 || row_stride != 32 || column_stride != 1)
    fail();
  const float input[32] = {
      -300, -255, -5, -3, -1, 0,    1,    3,   5,    254,  255,
      300,  2,    6,  10, 14, -257, -253, -7,  -9,   -11,  -13,
      -15,  7,    9,  11, 13, 15,   253,  257, -254, -256,
  };
  int8_t codes[32];
  for (int k = 0; k < 32; ++k) {
    float scaled = input[k] * 0.5f;
    int rounded = (int)scaled;
    float fraction = scaled - rounded;
    if (fraction > 0.5f || (fraction == 0.5f && rounded % 2 != 0))
      ++rounded;
    if (fraction < -0.5f || (fraction == -0.5f && rounded % 2 != 0))
      --rounded;
    if (rounded < -128)
      rounded = -128;
    if (rounded > 127)
      rounded = 127;
    codes[k] = (int8_t)rounded;
  }
  for (int column = 0; column < 32; ++column) {
    int32_t accumulator = column - 16;
    for (int k = 0; k < 32; ++k) {
      int8_t weight = (7 * k + 3 * column) % 13 - 6;
      accumulator += codes[k] * weight;
    }
    float step = 2.0f * ((column + 1) / 128.0f);
    float expected = (float)accumulator * step;
    float actual = output[offset + column * column_stride];
    // The scales are binary fractions; these results are exact.
    if (actual != expected) {
      printf("FAILED: linear_int8 column=%d expected=%f actual=%f\n", column,
             expected, actual);
      fail();
    }
  }
  printf("PASSED: linear_int8 1x32x32 rounding, saturation, channel scales, "
         "bias\n");
}
