#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t n, int64_t h, int64_t w, int64_t c,
                             int64_t ns, int64_t hs, int64_t ws, int64_t cs) {
  static const int32_t expected[16] = {-2, 0,  2,  3,  4,  6,  8,  9,
                                       10, 12, 14, 15, 16, 18, 20, 21};
  if (n != 1 || h != 1 || w != 1 || c != 16)
    abort();
  for (int i = 0; i < 16; ++i)
    if (data[offset + i * cs] != expected[i])
      abort();
  printf("PASSED: single-pixel convolution uses matrix scheduling\n");
}
