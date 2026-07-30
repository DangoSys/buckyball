#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ROWS 2
#define LANES 16
#define SCALE_0_25 0x3E800000U

static int8_t int8_input[ROWS * LANES] __attribute__((aligned(64))) = {
    -128, -127, -31, -15, -7, -3, -1, 0, 1, 2, 3,  5,  7,  15, 31, 127,
    -64,  -33,  -17, -9,  -5, -2, 0,  4, 6, 8, 10, 12, 14, 16, 32, 64,
};

static float expected_fp32[ROWS * LANES] __attribute__((aligned(64))) = {
    -32.0f, -31.75f, -7.75f, -3.75f, -1.75f, -0.75f, -0.25f, 0.0f,
    0.25f,  0.5f,    0.75f,  1.25f,  1.75f,  3.75f,  7.75f,  31.75f,
    -16.0f, -8.25f,  -4.25f, -2.25f, -1.25f, -0.5f,  0.0f,   1.0f,
    1.5f,   2.0f,    2.5f,   3.0f,   3.5f,   4.0f,   8.0f,   16.0f,
};

static float output_fp32[ROWS * LANES] __attribute__((aligned(64)));

static uint32_t fp32_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  const uint32_t src_bank = 0;
  const uint32_t dst_bank = 1;

  bb_mem_alloc(src_bank, ROWS, 1);
  bb_mem_alloc(dst_bank, ROWS, 4);
  bb_mvin((uintptr_t)int8_input, src_bank, ROWS, 1);
  bb_int2fp(src_bank, dst_bank, ROWS, SCALE_0_25);
  bb_mvout((uintptr_t)output_fp32, dst_bank, ROWS, 1);
  bb_fence();

  int passed = 1;
  for (int i = 0; i < ROWS * LANES; ++i) {
    const uint32_t actual = fp32_bits(output_fp32[i]);
    const uint32_t expected = fp32_bits(expected_fp32[i]);
    if (actual != expected) {
      printf("MISMATCH at [%d]: got 0x%08X, expected 0x%08X\n", i, actual,
             expected);
      passed = 0;
    }
  }

  printf("Int2Fp INT8-to-FP32 test %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
