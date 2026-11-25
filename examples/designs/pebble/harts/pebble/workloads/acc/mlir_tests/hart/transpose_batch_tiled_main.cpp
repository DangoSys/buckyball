#include <array>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static constexpr auto reference = [] {
  std::array<int8_t, 2 * 208 * 5> values{};
  for (int b = 0; b < 2; ++b)
    for (int r = 0; r < 208; ++r)
      for (int c = 0; c < 5; ++c)
        values[(b * 208 + r) * 5 + c] = (b * 13 + c * 5 + r * 3) % 255 - 128;
  return values;
}();

extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t batches, int64_t rows, int64_t columns,
                             int64_t bs, int64_t rs, int64_t cs) {
  if (batches != 2 || rows != 208 || columns != 5)
    exit(1);
  for (int b = 0; b < 2; ++b)
    for (int r = 0; r < 208; ++r)
      for (int c = 0; c < 5; ++c) {
        int actual = data[offset + b * bs + r * rs + c * cs];
        int expected = reference[(b * 208 + r) * 5 + c];
        if (actual != expected) {
          printf("Transpose mismatch [%d,%d,%d]: %d, expected %d\n", b, r, c,
                 actual, expected);
          exit(1);
        }
      }
  printf("PASSED tiled transpose: batch, cropped strides and partial rows\n");
}
