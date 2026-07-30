#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>

#define ROWS 2
#define LANES 16
#define SCALE_0_5 0x3F000000U

static int32_t int32_input[ROWS * LANES] __attribute__((aligned(64))) = {
    -1000, -257, -255, -5, -3, -1, 0, 1, 3, 5, 127, 253, 255, 257, 1000, 2,
    -999,  -511, -259, -9, -7, -3, 2, 4, 6, 9, 125, 251, 254, 258, 511,  999,
};

static int8_t expected_int8[ROWS * LANES] __attribute__((aligned(64))) = {
    -128, -128, -128, -2, -2, 0,  0, 0, 2, 2, 64, 126, 127, 127, 127, 1,
    -128, -128, -128, -4, -4, -2, 1, 2, 3, 4, 62, 126, 127, 127, 127, 127,
};

static int8_t output_int8[ROWS * LANES] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  const uint32_t src_bank = 0;
  const uint32_t dst_bank = 1;

  bb_mem_alloc(src_bank, ROWS, 4);
  bb_mem_alloc(dst_bank, ROWS, 1);
  bb_mvin((uintptr_t)int32_input, src_bank, ROWS, 1);
  bb_int32_to_int8(src_bank, dst_bank, ROWS, SCALE_0_5);
  bb_mvout((uintptr_t)output_int8, dst_bank, ROWS, 1);
  bb_fence();

  int passed = 1;
  for (int i = 0; i < ROWS * LANES; ++i) {
    if (output_int8[i] != expected_int8[i]) {
      printf("MISMATCH at [%d]: got %d, expected %d\n", i, output_int8[i],
             expected_int8[i]);
      passed = 0;
    }
  }

  printf("Int2Fp INT32-to-INT8 test %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
