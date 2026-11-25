#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset, int64_t n,
                             int64_t h, int64_t w, int64_t c, int64_t ns,
                             int64_t hs, int64_t ws, int64_t cs) {
  if (n != 1 || h != 4 || w != 4 || c != 16)
    abort();
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
      for (int channel = 0; channel < 16; ++channel) {
        int peak_y = y + 2 < 4 ? y + 2 : 3;
        int peak_x = x + 2 < 4 ? x + 2 : 3;
        int expected = peak_y * 4 + peak_x - 40 - channel;
        if (data[offset + y * hs + x * ws + channel * cs] != expected)
          abort();
      }
  puts("PASSED: signed INT8 pooling with border padding");
}
