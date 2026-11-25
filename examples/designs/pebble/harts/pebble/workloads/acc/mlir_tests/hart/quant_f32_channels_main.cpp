#include <array>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static constexpr auto reference = [] {
  std::array<int8_t, 2 * 2 * 131 * 19> values{};
  for (int scalar = 0; scalar < 2; ++scalar)
    for (int b = 0; b < 2; ++b)
      for (int r = 0; r < 131; ++r)
        for (int c = 0; c < 19; ++c) {
          int value = (b * 7 + r * 3 + c) % 29 - 14;
          int expected;
          if (scalar || c % 3 == 0) {
            int magnitude = (value < 0 ? -value : value),
                rounded = magnitude / 4;
            if (magnitude % 4 > 2 || (magnitude % 4 == 2 && rounded % 2))
              ++rounded;
            expected = value < 0 ? -rounded : rounded;
          } else {
            expected = c % 3 == 1 ? value : value * 16;
          }
          if (expected < -128)
            expected = -128;
          if (expected > 127)
            expected = 127;

          values[((scalar * 2 + b) * 131 + r) * 19 + c] = expected;
        }
  return values;
}();

extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t batches, int64_t rows, int64_t columns,
                             int64_t bs, int64_t rs, int64_t cs,
                             int32_t scalar) {
  printf("Checking quantization mode %d, shape %ld x %ld x %ld\n", scalar,
         batches, rows, columns);
  if (batches != 2 || rows != 131 || columns != 19)
    exit(1);
  for (int b = 0; b < 2; ++b)
    for (int r = 0; r < 131; ++r)
      for (int c = 0; c < 19; ++c) {
        int expected = reference[((scalar * 2 + b) * 131 + r) * 19 + c];
        int actual = data[offset + b * bs + r * rs + c * cs];
        if (actual != expected) {
          printf("Mismatch mode %d at [%d,%d,%d]: %d, expected %d\n", scalar, b,
                 r, c, actual, expected);
          exit(1);
        }
      }
  printf(
      "PASSED FP32 quantization: channel and row tails, RNE and saturation\n");
}
