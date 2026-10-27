#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static_assert(THREAD_MEM_BANK_ENTRIES == 1024);
extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != 16 || lanes != 16 || stride != 16 || lane_stride != 1)
    exit(1);
  for (int row = 0; row < 16; ++row)
    for (int column = 0; column < 16; ++column) {
      int32_t expected = 7;
      for (int k = 0; k < 2048; ++k)
        expected +=
            ((3 * row + 5 * k) % 7 - 3) * ((2 * k + 3 * column) % 5 - 2);
      // in_shift by two, rounding to nearest even.
      const uint32_t bits = (uint32_t)expected;
      const int rounded =
          ((bits >> 1) & 1) && ((bits & 1) || ((bits >> 2) & 1));
      expected = (expected >> 2) + rounded;
      if (data[offset + row * stride + column] != expected) {
        printf("gemmini accumulated mismatch row=%d column=%d expected=%d "
               "actual=%d\n",
               row, column, expected, data[offset + row * stride + column]);
        exit(1);
      }
    }
  printf("gemmini matmul_i8_tile read_destination over two groups PASS\n");
}
