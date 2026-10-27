#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int32_t *, int32_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != 16 || lanes != 16 || stride != 16 || lane_stride != 1)
    exit(1);
  for (int row = 0; row < 16; ++row)
    for (int column = 0; column < 16; ++column) {
      int32_t expected = -5;
      for (int k = 0; k < THREAD_MEM_BANK_ENTRIES; ++k)
        expected +=
            ((3 * k + 5 * row) % 7 - 3) * ((2 * k + 3 * column) % 5 - 2);
      if (data[offset + row * stride + column] != expected) {
        printf("matmul_at_i8 bank-depth mismatch row=%d column=%d\n", row,
               column);
        exit(1);
      }
    }
  printf("matmul_at_i8 bank-depth PASS K=%d\n", THREAD_MEM_BANK_ENTRIES);
}
