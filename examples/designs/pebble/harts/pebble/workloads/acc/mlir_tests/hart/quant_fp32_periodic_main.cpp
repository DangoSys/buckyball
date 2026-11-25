#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void init_input(float *, float *data, int64_t offset, int64_t,
                           int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           int64_t) {
  for (int i = 0; i < 96; ++i)
    data[offset + i] = i == 0 ? 1024.0f : i == 1 ? -1024.0f : (i - 48) * 0.5f;
}
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset, int64_t n,
                             int64_t h, int64_t w, int64_t c, int64_t,
                             int64_t sh, int64_t sw, int64_t sc) {
  if (n != 1 || h != 2 || w != 16 || c != 3)
    abort();
  for (int i = 0; i < 96; ++i) {
    int value = (i - 48) * (i % 3 == 0 ? 2 : i % 3 == 1 ? 1 : 8);
    int magnitude = abs(value), quotient = magnitude / 8,
        remainder = magnitude % 8;
    int rounded =
        quotient + (remainder > 4 || (remainder == 4 && (quotient & 1)));
    int expected = i == 0      ? 127
                   : i == 1    ? -128
                   : value < 0 ? -rounded
                               : rounded;
    if (data[offset + (i / 48) * sh + (i % 48 / 3) * sw + (i % 3) * sc] !=
        expected)
      abort();
  }
  puts("PASSED: RGB periodic FP32 quantization, ties and saturation");
}
