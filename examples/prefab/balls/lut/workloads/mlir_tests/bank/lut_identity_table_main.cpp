#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != THREAD_MEM_BANK_ENTRIES || lanes != THREAD_MEM_BANK_WIDTH / 8 ||
      stride != lanes || lane_stride != 1)
    exit(1);
  for (int i = 0; i < rows * lanes; ++i)
    if (data[offset + i] != (int8_t)i) {
      printf("lut bank mismatch index=%d\n", i);
      exit(1);
    }
  printf("lut bank PASS rows=%d\n", THREAD_MEM_BANK_ENTRIES);
}
