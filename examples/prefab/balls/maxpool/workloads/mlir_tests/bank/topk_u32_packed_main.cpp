#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(uint32_t *, uint32_t *data, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs) {
  const uint32_t expected[16] = {
      UINT32_MAX, 3,  UINT32_MAX, 8,  0x80000000u, 4,  0x80000000u, 9,
      9,          18, 8,          17, 7,           16, 0,           0};
  if (rows != 4 || columns != 4)
    exit(1);
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      if (data[offset + r * rs + c * cs] != expected[r * 4 + c]) {
        printf("topk_u32_packed FAIL row=%d lane=%d\n", r, c);
        exit(1);
      }
  printf("topk_u32_packed PASS\n");
}
