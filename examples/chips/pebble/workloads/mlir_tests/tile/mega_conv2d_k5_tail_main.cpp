#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct Expected {
  int values[16 * 25]{};
};
constexpr Expected reference() {
  Expected result{};
  for (int o = 0; o < 16; ++o)
    for (int y = 0; y < 5; ++y)
      for (int x = 0; x < 5; ++x) {
        int expected = 0;
        for (int ky = 0; ky < 5; ++ky)
          for (int kx = 0; kx < 5; ++kx)
            for (int c = 0; c < 2; ++c) {
              int iy = y + ky - 2, ix = x + kx - 2;
              if (iy >= 0 && iy < 5 && ix >= 0 && ix < 5)
                expected += ((iy * 3 + ix * 2 + c) % 7 - 3) *
                            ((ky * 5 + kx + c + o) % 3 - 1);
            }
        result.values[(o * 5 + y) * 5 + x] = expected;
      }
  return result;
}
constexpr auto expected_values = reference();

extern "C" void check_result(float *, float *data, int64_t offset, int64_t n,
                             int64_t channels, int64_t height, int64_t width,
                             int64_t ns, int64_t cs, int64_t hs, int64_t ws) {
  if (n != 1 || channels != 16 || height != 5 || width != 5)
    abort();
  for (int o = 0; o < 16; ++o)
    for (int y = 0; y < 5; ++y)
      for (int x = 0; x < 5; ++x) {
        int expected = expected_values.values[(o * 5 + y) * 5 + x];
        float actual = data[offset + o * cs + y * hs + x * ws];
        if (actual != expected) {
          printf("FAIL k5 tail o=%d y=%d x=%d actual=%f expected=%d\n", o, y, x,
                 actual, expected);
          abort();
        }
      }
  printf("PASS k5 Conv: 4x4 tiles, tail, 8x16 packing and FP32 row copies\n");
}
