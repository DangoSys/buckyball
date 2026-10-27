#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(uint16_t *, uint16_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != THREAD_MEM_BANK_ENTRIES || lanes != 8 || stride != 8 ||
      lane_stride != 1)
    exit(1);
  for (int row = 0; row < rows; ++row)
    for (int lane = 0; lane < lanes; ++lane)
      if (data[offset + row * stride + lane] !=
          (uint16_t)(0x4400 | ((row ^ lane) & 1) << 15)) {
        printf("RedMulE bank rows mismatch row=%d lane=%d\n", row, lane);
        exit(1);
      }
  puts("RedMulE BankOp full bank rows PASS");
}
