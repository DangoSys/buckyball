#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t round_div(int64_t value, int64_t divisor) {
  uint64_t n = value < 0 ? -value : value;
  uint64_t q = n / divisor, r = n % divisor;
  q += r * 2 > (uint64_t)divisor || (r * 2 == (uint64_t)divisor && (q & 1));
  return value < 0 ? -(int64_t)q : (int64_t)q;
}
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t n, int64_t h, int64_t w, int64_t c,
                             int64_t ns, int64_t hs, int64_t ws, int64_t cs) {
  if (n != 1 || h != 2 || w != 3 || c != 24)
    abort();
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 3; ++x)
      for (int channel = 0; channel < 24; ++channel) {
        int i = y * 120 + (x + 1) * 24 + channel;
        int64_t q = round_div(((i - 120) * 1024 + i % 3) * 3, 2);
        q = round_div(q * (q + 3 * (1 << 24)), 6 * (1 << 24));
        q = round_div(q, 2);
        if (data[offset + y * hs + x * ws + channel * cs] != q) {
          printf("FAILED fixed tail: y=%d x=%d c=%d\n", y, x, channel);
          abort();
        }
      }
  printf("PASSED fixed strided crop and channel tail\n");
}
