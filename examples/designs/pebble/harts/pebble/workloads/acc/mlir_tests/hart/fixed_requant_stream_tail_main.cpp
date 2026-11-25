#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void init(int32_t *, int32_t *multipliers, int64_t mo, int64_t ms,
                     int64_t mt, int32_t *, int32_t *shifts, int64_t so,
                     int64_t ss, int64_t st) {
  if (ms != 2064 || ss != 2064)
    exit(1);
  for (int i = 0; i < 2064; ++i) {
    multipliers[mo + i * mt] = i / 1024 + 1;
    shifts[so + i * st] = 1;
  }
}
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  if (rows != 1 || columns != 2064)
    exit(1);
  for (int i = 0; i < 2064; ++i)
    if (data[offset + i * cs] != (i < 1024 ? 2 : i < 2048 ? 3 : 4))
      exit(1);
  printf("PASS: streamed requantization with changing coefficients and a final "
         "partial tile\n");
}
