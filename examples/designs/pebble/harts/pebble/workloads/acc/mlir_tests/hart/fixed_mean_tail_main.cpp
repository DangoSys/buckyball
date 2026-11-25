#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t n, int64_t h, int64_t w, int64_t c,
                             int64_t ns, int64_t hs, int64_t ws, int64_t cs) {
  static const int32_t expected[2][10] = {
      {INT32_MAX, INT32_MIN, 0, 2, 0, -2, 17, -18, 19, -20},
      {INT32_MAX, INT32_MIN, 0, 2, 0, -2, 117, -118, 119, -120}};
  if (n != 2 || h != 1 || w != 1 || c != 10)
    abort();
  for (int batch = 0; batch < 2; ++batch)
    for (int channel = 0; channel < 10; ++channel)
      if (data[offset + batch * ns + channel * cs] !=
          expected[batch][channel]) {
        printf("FAILED fixed mean: batch=%d channel=%d\n", batch, channel);
        abort();
      }
  printf(
      "PASSED fixed mean: strided input, channel tail, INT64 sums and ties\n");
}
