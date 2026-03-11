#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16

// INT32 input data (4 elements per SRAM word, 16 words)
static int32_t int32_input[DIM * 4] __attribute__((aligned(64))) = {
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8,
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8,
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8,
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 7, 100, -100, 8, 16, -8,
};

// Expected FP32 output as bit patterns: int32_val * 1.0 = float(int32_val)
// 1.0=0x3F800000, 2.0=0x40000000, 3.0=0x40400000, -1.0=0xBF800000
// -2.0=0xC0000000, 0.0=0x00000000, 4.0=0x40800000, 5.0=0x40A00000
// 10.0=0x41200000, -10.0=0xC1200000, 7.0=0x40E00000, 100.0=0x42C80000
// -100.0=0xC2C80000, 8.0=0x41000000, 16.0=0x41800000, -8.0=0xC1000000
static uint32_t expected_fp32[DIM * 4] __attribute__((aligned(64))) = {
    0x3F800000, 0x40000000, 0x40400000, 0xBF800000, 0xC0000000, 0x00000000,
    0x40800000, 0x40A00000, 0x41200000, 0xC1200000, 0x40E00000, 0x42C80000,
    0xC2C80000, 0x41000000, 0x41800000, 0xC1000000, 0x3F800000, 0x40000000,
    0x40400000, 0xBF800000, 0xC0000000, 0x00000000, 0x40800000, 0x40A00000,
    0x41200000, 0xC1200000, 0x40E00000, 0x42C80000, 0xC2C80000, 0x41000000,
    0x41800000, 0xC1000000, 0x3F800000, 0x40000000, 0x40400000, 0xBF800000,
    0xC0000000, 0x00000000, 0x40800000, 0x40A00000, 0x41200000, 0xC1200000,
    0x40E00000, 0x42C80000, 0xC2C80000, 0x41000000, 0x41800000, 0xC1000000,
    0x3F800000, 0x40000000, 0x40400000, 0xBF800000, 0xC0000000, 0x00000000,
    0x40800000, 0x40A00000, 0x41200000, 0xC1200000, 0x40E00000, 0x42C80000,
    0xC2C80000, 0x41000000, 0x41800000, 0xC1000000,
};

static uint32_t output_fp32[DIM * 4] __attribute__((aligned(64)));

// FP32 bit pattern for scale = 1.0
#define SCALE_1_0 0x3F800000U

void hw_dequant(int32_t *input, uint32_t *output, int num_words) {
  uint32_t op1_bank_id = 0;
  uint32_t wr_bank_id = 1;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(wr_bank_id, 1, 1);

  bb_mvin((uintptr_t)input, op1_bank_id, num_words, 1);

  bb_dequant(op1_bank_id, wr_bank_id, num_words, SCALE_1_0);

  bb_mvout((uintptr_t)output, wr_bank_id, num_words, 1);
  bb_fence();
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  for (int i = 0; i < DIM * 4; i++) {
    output_fp32[i] = 0;
  }

  hw_dequant(int32_input, output_fp32, DIM);

  int passed = 1;
  for (int i = 0; i < DIM * 4; i++) {
    if (output_fp32[i] != expected_fp32[i]) {
      printf("MISMATCH at [%d]: got 0x%08X, expected 0x%08X\n", i,
             output_fp32[i], expected_fp32[i]);
      passed = 0;
    }
  }

  if (passed) {
    printf("Dequant test PASSED\n");
  } else {
    printf("Dequant test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
