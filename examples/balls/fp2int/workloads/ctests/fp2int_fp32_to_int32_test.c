#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>

#define ROWS 4
#define LANES 4
#define N (ROWS * LANES)
#define SCALE_1_0 0x3F800000U

static uint32_t fp32_input[N] __attribute__((aligned(64))) = {
    0x3F800000, 0x40000000, 0x40400000, 0xBF800000, 0xC0000000, 0x00000000,
    0x40800000, 0x40A00000, 0x41200000, 0xC1200000, 0x3F000000, 0x42C80000,
    0xC2C80000, 0x40E00000, 0x41000000, 0xC1000000,
};

static int32_t expected[N] __attribute__((aligned(64))) = {
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 0, 100, -100, 7, 8, -8,
};

static int32_t output[N] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  const uint32_t src = 0;
  const uint32_t dst = 1;
  bb_mem_alloc(src, ROWS, 1);
  bb_mem_alloc(dst, ROWS, 1);
  bb_mvin((uintptr_t)fp32_input, src, ROWS, 1);
  bb_fp2int(src, dst, ROWS, SCALE_1_0);
  bb_mvout((uintptr_t)output, dst, ROWS, 1);
  bb_fence();

  int passed = 1;
  for (int i = 0; i < N; i++) {
    if (output[i] != expected[i]) {
      printf("MISMATCH [%d]: got %d expected %d\n", i, output[i], expected[i]);
      passed = 0;
    }
  }
  printf("Fp2Int FP32-to-INT32 %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
