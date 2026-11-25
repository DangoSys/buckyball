#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t size, int64_t stride) {
  if (size != 320 || stride != 1)
    abort();
  for (unsigned i = 0; i < 320; ++i) {
    unsigned at = i >= 5 ? i - 5 : 320;
    uint8_t expected = at / 96 < 3 && at % 96 < 64
                           ? (uint8_t)(3 + at / 96 * 80 + at % 96)
                           : 90;
    if ((uint8_t)data[offset + i] != expected)
      abort();
  }
  printf("PASSED: blocked DMA bank lowering\n");
}
