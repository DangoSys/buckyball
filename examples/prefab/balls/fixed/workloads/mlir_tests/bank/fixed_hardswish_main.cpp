#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int32_t *, int32_t *values, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  const int64_t q = INT64_C(1) << 24;
  const int64_t input[16] = {0,         q / 2,      q,         -q / 2,
                             -q,        -3 * q / 2, 2 * q,     -2 * q,
                             3 * q,     -3 * q,     4 * q,     -4 * q,
                             INT32_MAX, INT32_MIN,  3 * q - 1, -3 * q + 1};
  if (rows != 4 || columns != 4)
    abort();
  for (int i = 0; i < 16; ++i) {
    int64_t x = input[i], expected;
    if (x <= -3 * q)
      expected = 0;
    else if (x >= 3 * q)
      expected = x;
    else {
      int64_t product = x * (x + 3 * q);
      uint64_t n = product < 0 ? -product : product;
      uint64_t result = n / (6 * q), remainder = n % (6 * q);
      result +=
          remainder * 2 > 6 * q || (remainder * 2 == 6 * q && (result & 1));
      expected = product < 0 ? -(int64_t)result : (int64_t)result;
    }
    if (values[offset + i / 4 * rs + i % 4 * cs] != expected)
      abort();
  }
  printf("PASSED: fixed hardswish MLIR\n");
}
