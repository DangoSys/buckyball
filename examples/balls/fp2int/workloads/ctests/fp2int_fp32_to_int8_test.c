#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>

#define ROWS 2
#define LANES 16
#define SCALE_2_0 0x40000000U

static float fp32_input[ROWS * LANES] __attribute__((aligned(64))) = {
    0.125f,  -0.125f,  0.25f,  -0.25f,  0.75f,   -0.75f,  1.25f,  -1.25f,
    1.75f,   -1.75f,   63.25f, 63.75f,  -63.75f, -64.75f, 0.0f,   -0.0f,
    2.25f,   -2.25f,   2.75f,  -2.75f,  3.25f,   -3.25f,  3.75f,  -3.75f,
    10.125f, -10.125f, 20.25f, -20.25f, 0.375f,  -0.375f, 64.25f, -65.25f,
};

static int8_t expected_int8[ROWS * LANES] __attribute__((aligned(64))) = {
    0, 0,  0, 0,  2, -2, 2, -2, 4,  -4,  126, 127, -128, -128, 0,   0,
    4, -4, 6, -6, 6, -6, 8, -8, 20, -20, 40,  -40, 1,    -1,   127, -128,
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
  bb_mvin((uintptr_t)fp32_input, src_bank, ROWS, 1);
  bb_fp2int(src_bank, dst_bank, ROWS, SCALE_2_0);
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

  printf("Fp2Int FP32-to-INT8 test %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
