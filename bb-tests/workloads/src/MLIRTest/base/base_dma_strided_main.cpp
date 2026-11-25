#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                             int64_t size, int64_t stride) {
  for (int i = 0; i < 128; ++i) {
    int at = i >= 3 ? i - 3 : 128;
    int8_t expected =
        at / 23 < 4 && at % 23 < 7 ? 3 + at / 23 * 20 + at % 23 : 90;
    if (aligned[offset + i * stride] != expected) {
      printf("Strided DMA MLIR mismatch: byte=%d\n", i);
      exit(1);
    }
  }
  printf("PASSED: strided DMA MLIR\n");
}
