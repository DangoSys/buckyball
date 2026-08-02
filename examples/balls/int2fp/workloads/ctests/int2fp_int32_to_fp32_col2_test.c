#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>

#define ROWS 2
#define LANES 8
#define SCALE_1_0 0x3F800000U

static int32_t int32_input[ROWS * LANES] __attribute__((aligned(64))) = {
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8,
};

static uint32_t expected_fp32[ROWS * LANES] __attribute__((aligned(64))) = {
    0x3F800000, 0x40000000, 0x40400000, 0xBF800000, 0xC0000000, 0x00000000,
    0x40800000, 0x40A00000, 0x41200000, 0xC1200000, 0x40E00000, 0x42C80000,
    0xC2C80000, 0x41000000, 0x41800000, 0xC1000000,
};

static uint32_t output_fp32[ROWS * LANES] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  const uint32_t src_bank = 0;
  const uint32_t dst_bank = 1;

  bb_mem_alloc(src_bank, ROWS, 2);
  bb_mem_alloc(dst_bank, ROWS, 2);
  bb_mvin((uintptr_t)int32_input, src_bank, ROWS, 1);
  bb_int2fp(src_bank, dst_bank, ROWS, SCALE_1_0);
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

  printf("Int2Fp INT32-to-FP32 col2 test %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
