#include <array>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static constexpr auto reference = [] {
  std::array<int32_t, 2 * 17 * 32> values{};
  for (int b = 0; b < 2; ++b)
    for (int m = 0; m < 17; ++m)
      for (int n = 0; n < 32; ++n)
        for (int k = 0; k < 33; ++k)
          values[(b * 17 + m) * 32 + n] +=
              ((b * 7 + m * 3 + k) % 19 - 9) * ((b * 5 + k * 2 + n) % 23 - 11);
  return values;
}();

extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t batches, int64_t rows, int64_t columns,
                             int64_t bs, int64_t rs, int64_t cs) {
  if (batches != 2 || rows != 17 || columns != 32)
    abort();
  for (int b = 0; b < 2; ++b)
    for (int m = 0; m < 17; ++m)
      for (int n = 0; n < 32; ++n) {
        int32_t expected = reference[(b * 17 + m) * 32 + n];
        int32_t actual = data[offset + b * bs + m * rs + n * cs];
        if (actual != expected) {
          printf("Mismatch at [%d,%d,%d]: %d, expected %d\n", b, m, n, actual,
                 expected);
          exit(1);
        }
      }
  printf("PASSED batch matmul: signed values, batch isolation and M/K tails\n");
}
