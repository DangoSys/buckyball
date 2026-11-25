#include <array>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static constexpr auto expected = [] {
  std::array<int32_t, 3 * 3 * 16> data{};
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x)
      for (int c = 0; c < 16; ++c)
        for (int ky = 0; ky < 3; ++ky)
          for (int kx = 0; kx < 3; ++kx) {
            int iy = y * 2 + ky - 1, ix = x * 2 + kx - 1;
            if (iy < 0 || iy >= 5 || ix < 0 || ix >= 5)
              continue;
            int32_t value = c == 0   ? INT_MAX
                            : c == 1 ? INT_MAX - 1
                                     : (iy * 7 + (ix + 1) * 11 + c * 3) % 37;
            int index = (y * 3 + x) * 16 + c;
            if (value > data[index])
              data[index] = value;
          }
  return data;
}();
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t n, int64_t h, int64_t w, int64_t c,
                             int64_t ns, int64_t hs, int64_t ws, int64_t cs) {
  if (n != 1 || h != 3 || w != 3 || c != 16)
    abort();
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x)
      for (int channel = 0; channel < 16; ++channel)
        if (data[offset + y * hs + x * ws + channel * cs] !=
            expected[(y * 3 + x) * 16 + channel]) {
          printf("FAILED I32 maxpool: y=%d x=%d channel=%d\n", y, x, channel);
          abort();
        }
  printf(
      "PASSED I32 maxpool: cropped strides, border branches and INT32_MAX\n");
}
