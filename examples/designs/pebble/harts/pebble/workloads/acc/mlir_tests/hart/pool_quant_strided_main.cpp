#include <array>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static constexpr auto expected = [] {
  std::array<int8_t, 9 * 9 * 16> data{};
  for (int y = 0; y < 9; ++y)
    for (int x = 0; x < 9; ++x)
      for (int c = 0; c < 16; ++c) {
        int maximum = 0;
        for (int ky = 0; ky < 3; ++ky)
          for (int kx = 0; kx < 3; ++kx) {
            int value =
                c == 0 ? 4096
                : c == 1
                    ? -1000
                    : ((2 * y + ky) * 7 + (2 * x + kx + 1) * 11 + c * 3) % 37 -
                          19;
            if (value > maximum)
              maximum = value;
          }
        int scaled = maximum * (c + 1), rounded = scaled / 16;
        int remainder = scaled % 16;
        if (remainder > 8 || (remainder == 8 && rounded % 2))
          ++rounded;
        data[(y * 9 + x) * 16 + c] = rounded > 127 ? 127 : rounded;
      }
  return data;
}();

extern "C" void check_result(int8_t *, int8_t *data, int64_t offset, int64_t n,
                             int64_t h, int64_t w, int64_t c, int64_t ns,
                             int64_t hs, int64_t ws, int64_t cs) {
  if (n != 1 || h != 9 || w != 9 || c != 16)
    abort();
  for (int y = 0; y < 9; ++y)
    for (int x = 0; x < 9; ++x)
      for (int channel = 0; channel < 16; ++channel)
        if (data[offset + y * hs + x * ws + channel * cs] !=
            expected[(y * 9 + x) * 16 + channel]) {
          printf("FAILED pool quant: y=%d x=%d channel=%d\n", y, x, channel);
          abort();
        }
  printf("PASSED pool quant: crop, overlap, tails, ReLU, rounding and "
         "saturation\n");
}
