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

static int8_t quantize(float value) {
  int rounded = (int)value;
  float fraction = value - rounded;
  if (fraction > 0.5f || (fraction == 0.5f && rounded % 2 != 0))
    ++rounded;
  if (fraction < -0.5f || (fraction == -0.5f && rounded % 2 != 0))
    --rounded;
  if (rounded < -128)
    rounded = -128;
  if (rounded > 127)
    rounded = 127;
  return (int8_t)rounded;
}

extern "C" void check_result(float *, float *output, int64_t offset,
                             int64_t rows, int64_t columns, int64_t row_stride,
                             int64_t column_stride) {
  if (rows != 1 || columns != 16 || row_stride != 16 || column_stride != 1)
    fail();
  int8_t input[32], intermediate[32];
  for (int k = 0; k < 32; ++k)
    input[k] = quantize((float)((k % 7 - 3) * 99) * 0.5f);
  for (int column = 0; column < 32; ++column) {
    int32_t accumulator = column - 16;
    for (int k = 0; k < 32; ++k)
      accumulator += input[k] * ((3 * k + 5 * column) % 9 - 4);
    float scale = 2.0f * ((column + 1) / 64.0f) * 0.5f;
    int8_t requantized = quantize((float)accumulator * scale);
    intermediate[column] = requantized < 0 ? 0 : requantized;
  }
  for (int column = 0; column < 16; ++column) {
    int32_t accumulator = column - 8;
    for (int k = 0; k < 32; ++k)
      accumulator += intermediate[k] * ((5 * k + 3 * column) % 7 - 3);
    float expected = (float)accumulator * (2.0f * ((column + 1) / 128.0f));
    float actual = output[offset + column * column_stride];
    // Binary fraction scales keep this small integer reference exact.
    if (actual != expected) {
      printf("FAILED: linear_chain_i8 column=%d expected=%f actual=%f\n",
             column, expected, actual);
      fail();
    }
  }
  printf(
      "PASSED: linear_chain_i8 two Linear layers with ReLU and INT8 rescale\n");
}
