#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#define ROWS 2
#define LANES 4
#define SCALE_0_25 0x3E800000U

static int32_t int32_input[ROWS * LANES] __attribute__((aligned(64))) = {
    1, -1, 2, -2, 3, -3, 5, -5,
};

static uint32_t expected_fp32[ROWS * LANES] __attribute__((aligned(64))) = {
    0x3E800000, 0xBE800000, 0x3F000000, 0xBF000000,
    0x3F400000, 0xBF400000, 0x3FA00000, 0xBFA00000,
};

static uint32_t output_fp32[ROWS * LANES] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  const uint32_t src_bank = 0;
  const uint32_t dst_bank = 1;

  bb_mem_alloc(src_bank, ROWS, 1);
  bb_mem_alloc(dst_bank, ROWS, 1);
  bb_mvin((uintptr_t)int32_input, src_bank, ROWS, 1);
  bb_int2fp(src_bank, dst_bank, ROWS, SCALE_0_25);
  bb_mvout((uintptr_t)output_fp32, dst_bank, ROWS, 1);
  bb_fence();

  int passed = 1;
  for (int i = 0; i < ROWS * LANES; ++i) {
    if (output_fp32[i] != expected_fp32[i]) {
      printf("MISMATCH at [%d]: got 0x%08X, expected 0x%08X\n", i,
             output_fp32[i], expected_fp32[i]);
      passed = 0;
    }
  }

  printf("Int2Fp INT32-to-FP32 test %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
