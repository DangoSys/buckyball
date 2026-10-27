#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != THREAD_MEM_BANK_ENTRIES || lanes != 16 || stride != 16 ||
      lane_stride != 1)
    exit(1);
  for (int block = 0; block < THREAD_MEM_BANK_ENTRIES / 2; ++block)
    for (int element = 0; element < 32; ++element) {
      int value = (block + element) % 16 - 8;
      int shift = block % 8 - 2;
      int expected = shift >= 0 ? value << shift : value >> -shift;
      if (expected > 127)
        expected = 127;
      if (expected < -128)
        expected = -128;
      if (data[offset + block * 32 + element] != expected) {
        printf("mxfp4_i8 bank mismatch block=%d element=%d got=%d "
               "expected=%d\n",
               block, element, data[offset + block * 32 + element], expected);
        exit(1);
      }
    }
  printf("mxfp4_i8 bank PASS blocks=%d\n", THREAD_MEM_BANK_ENTRIES / 2);
}
