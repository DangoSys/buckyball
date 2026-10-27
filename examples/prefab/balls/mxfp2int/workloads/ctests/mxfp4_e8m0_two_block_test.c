#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/mxfp2int.h>
#include <stdint.h>
#include <stdio.h>

#define NUM_BLOCKS 2
#define FP4_PER_BLOCK 32
#define BYTES_PER_BLOCK 16
#define E8M0_BIAS 127

static uint8_t input_mxfp4[NUM_BLOCKS * BYTES_PER_BLOCK]
    __attribute__((aligned(64)));
static uint8_t scales_e8m0[16] __attribute__((aligned(64)));
static int8_t output_int8[NUM_BLOCKS * FP4_PER_BLOCK]
    __attribute__((aligned(64)));

static uint8_t gen_fp4(int block, int elem) {
  return (uint8_t)((elem + block) & 0x0F);
}

static int8_t golden(int block, int elem) {
  int val = (int)gen_fp4(block, elem) - 8;
  int prod = val << block; /* scale = bias + block */
  if (prod > 127)
    prod = 127;
  if (prod < -128)
    prod = -128;
  return (int8_t)prod;
}

int main(void) {
  for (int b = 0; b < NUM_BLOCKS; b++) {
    for (int e = 0; e < FP4_PER_BLOCK; e += 2) {
      uint8_t lo = gen_fp4(b, e);
      uint8_t hi = gen_fp4(b, e + 1);
      input_mxfp4[b * BYTES_PER_BLOCK + e / 2] =
          (uint8_t)((hi << 4) | (lo & 0x0F));
    }
    scales_e8m0[b] = (uint8_t)(E8M0_BIAS + b);
  }

  const uint32_t in_bank = 1;
  const uint32_t scale_bank = 2;
  const uint32_t out_bank = 3;
  bb_mem_alloc(in_bank, 1, 1);
  bb_mem_alloc(scale_bank, 1, 1);
  bb_mem_alloc(out_bank, 1, 1);
  bb_mvin((uintptr_t)input_mxfp4, in_bank, NUM_BLOCKS, 1);
  bb_mvin((uintptr_t)scales_e8m0, scale_bank, 1, 1);
  bb_mxfp2int(in_bank, scale_bank, out_bank, NUM_BLOCKS);
  bb_mvout((uintptr_t)output_int8, out_bank, NUM_BLOCKS * 2, 1);

  int passed = 1;
  for (int b = 0; b < NUM_BLOCKS; b++) {
    for (int e = 0; e < FP4_PER_BLOCK; e++) {
      int8_t got = output_int8[b * FP4_PER_BLOCK + e];
      int8_t exp = golden(b, e);
      if (got != exp) {
        printf("MISMATCH b=%d e=%d got=%d exp=%d\n", b, e, (int)got, (int)exp);
        passed = 0;
      }
    }
  }

  bb_mem_release(in_bank);
  bb_mem_release(scale_bank);
  bb_mem_release(out_bank);
  printf("mxfp4 e8m0 two-block %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
