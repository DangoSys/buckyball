#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_lookup(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  const int32_t expected[8] = {1 << 30, 1 << 29, 1 << 28, 1 << 27, 0, 0, 0, 0};
  if (rows != 1 || columns != 8)
    abort();
  for (int i = 0; i < 8; ++i)
    if (data[offset + i * cs] != expected[i])
      abort();
}
extern "C" void check_softmax(int8_t *, int8_t *data, int64_t offset,
                              int64_t rows, int64_t columns, int64_t rs,
                              int64_t cs) {
  const int8_t expected[8] = {32, 16, 8, 4, 32, 16, 8, 4};
  if (rows != 1 || columns != 8)
    abort();
  for (int i = 0; i < 8; ++i)
    if (data[offset + i * cs] != expected[i])
      abort();
}
extern "C" void check_periodic(int32_t *, int32_t *data, int64_t offset,
                               int64_t rows, int64_t columns, int64_t rs,
                               int64_t cs) {
  if (rows != 2 || columns != 32)
    abort();
  for (int row = 0; row < 2; ++row)
    for (int i = 0; i < 32; ++i)
      if (data[offset + row * rs + i * cs] != (i % 2 ? 4 : 2))
        abort();
}
extern "C" void check_quant(int32_t *, int32_t *data, int64_t offset,
                            int64_t rows, int64_t columns, int64_t rs,
                            int64_t cs) {
  const int32_t expected[8] = {0, 2, 0, -2, INT32_MAX, INT32_MIN, 512, -512};
  if (rows != 1 || columns != 8)
    abort();
  for (int i = 0; i < 8; ++i)
    if (data[offset + i * cs] != expected[i])
      abort();
  puts("PASSED: nonlinear and periodic fixed lowering");
}
