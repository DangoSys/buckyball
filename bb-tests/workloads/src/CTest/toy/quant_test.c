#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16

// FP32 bit patterns for input data (4 elements per SRAM word, 16 words = 64
// FP32 values) Using scale = 1.0 so INT32 result = round(fp32_val * 1.0) =
// round(fp32_val) Values: 1.0, 2.0, 3.0, -1.0, -2.0, 0.0, 4.0, 5.0, 10.0,
// -10.0, 0.5, 100.0, -100.0, 7.0, 8.0, -8.0 Repeated 4 times to fill 16 words
// (64 elements)

static uint32_t fp32_input[DIM * 4] __attribute__((aligned(64))) = {
    // Word 0-3 (row 0): 1.0, 2.0, 3.0, -1.0 | -2.0, 0.0, 4.0, 5.0 | 10.0,
    // -10.0, 0.5, 100.0 | -100.0, 7.0, 8.0, -8.0
    0x3F800000,
    0x40000000,
    0x40400000,
    0xBF800000,
    0xC0000000,
    0x00000000,
    0x40800000,
    0x40A00000,
    0x41200000,
    0xC1200000,
    0x3F000000,
    0x42C80000,
    0xC2C80000,
    0x40E00000,
    0x41000000,
    0xC1000000,
    // Word 4-7 (row 1): same pattern
    0x3F800000,
    0x40000000,
    0x40400000,
    0xBF800000,
    0xC0000000,
    0x00000000,
    0x40800000,
    0x40A00000,
    0x41200000,
    0xC1200000,
    0x3F000000,
    0x42C80000,
    0xC2C80000,
    0x40E00000,
    0x41000000,
    0xC1000000,
    // Word 8-11 (row 2): same pattern
    0x3F800000,
    0x40000000,
    0x40400000,
    0xBF800000,
    0xC0000000,
    0x00000000,
    0x40800000,
    0x40A00000,
    0x41200000,
    0xC1200000,
    0x3F000000,
    0x42C80000,
    0xC2C80000,
    0x40E00000,
    0x41000000,
    0xC1000000,
    // Word 12-15 (row 3): same pattern
    0x3F800000,
    0x40000000,
    0x40400000,
    0xBF800000,
    0xC0000000,
    0x00000000,
    0x40800000,
    0x40A00000,
    0x41200000,
    0xC1200000,
    0x3F000000,
    0x42C80000,
    0xC2C80000,
    0x40E00000,
    0x41000000,
    0xC1000000,
};

// Expected INT32 output: round(fp32 * 1.0)
// 1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 1, 100, -100, 7, 8, -8 (0.5 rounds to 1)
static int32_t expected_int32[DIM * 4] __attribute__((aligned(64))) = {
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 1, 100, -100, 7, 8, -8,
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 1, 100, -100, 7, 8, -8,
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 1, 100, -100, 7, 8, -8,
    1, 2, 3, -1, -2, 0, 4, 5, 10, -10, 1, 100, -100, 7, 8, -8,
};

static int32_t output_int32[DIM * 4] __attribute__((aligned(64)));

// FP32 bit pattern for scale = 1.0
#define SCALE_1_0 0x3F800000U

void hw_quant(uint32_t *input, int32_t *output, int num_words) {
  uint32_t op1_bank_id = 0;
  uint32_t wr_bank_id = 1;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(wr_bank_id, 1, 1);

  bb_mvin((uintptr_t)input, op1_bank_id, num_words, 1);

  bb_quant(op1_bank_id, wr_bank_id, num_words, SCALE_1_0);

  bb_mvout((uintptr_t)output, wr_bank_id, num_words, 1);
  bb_fence();
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  for (int i = 0; i < DIM * 4; i++) {
    output_int32[i] = 0;
  }

  hw_quant(fp32_input, output_int32, DIM);

  int passed = 1;
  for (int i = 0; i < DIM * 4; i++) {
    if (output_int32[i] != expected_int32[i]) {
      printf("MISMATCH at [%d]: got %d, expected %d\n", i, output_int32[i],
             expected_int32[i]);
      passed = 0;
    }
  }

  if (passed) {
    printf("Quant test PASSED\n");
  } else {
    printf("Quant test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
