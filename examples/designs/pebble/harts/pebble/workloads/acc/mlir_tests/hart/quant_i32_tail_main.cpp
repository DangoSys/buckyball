#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset, int64_t n,
                             int64_t h, int64_t w, int64_t c, int64_t ns,
                             int64_t hs, int64_t ws, int64_t cs) {
  static const int8_t expected[20] = {0,  2,   0,    -2,  2,    4, -2,
                                      -4, 127, -128, 127, -128, 0, 1,
                                      2,  -1,  -2,   3,   4,    -3};
  if (n != 1 || h != 1 || w != 2 || c != 10)
    abort();
  for (int x = 0; x < 2; ++x)
    for (int channel = 0; channel < 10; ++channel)
      if (data[offset + x * ws + channel * cs] != expected[x * 10 + channel])
        abort();
  printf("PASSED: I32 quantization tail, ties and saturation\n");
}
