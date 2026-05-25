// MXFP4 to INT8 Dequantization Test with MMIO Per-Block Scale
//
// MXFP4 format (per block, 1 SRAM word = 16 bytes = 128 bits):
//   32 x 4-bit FP4 elements, packed LSB-first.
//   Byte i contains elements (2i) in bits[3:0] and (2i+1) in bits[7:4].
//
// Scale: per-block E8M0 (8-bit biased exponent, bias=127), loaded from MMIO.
//
// FP4 decoding (v1, simplified): signed_val = raw - 8 (range -8..+7).
//
// Output (per block, 2 SRAM words = 32 bytes):
//   32 x INT8 = saturate(signed_val << (scale - 127)).

#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <bbhw/mmio/mmio_allocator.c>
#include <bbhw/mmio/mmio_allocator.h>

#define NUM_BLOCKS 4
#define FP4_PER_BLOCK 32
#define BYTES_PER_BLOCK 16
#define INT8_PER_BLOCK 32
#define TOTAL_INPUT_BYTES (NUM_BLOCKS * BYTES_PER_BLOCK)
#define TOTAL_OUTPUT_BYTES (NUM_BLOCKS * INT8_PER_BLOCK)
#define TOTAL_SCALE_BYTES NUM_BLOCKS

#define E8M0_BIAS 127

static uint8_t input_mxfp4[TOTAL_INPUT_BYTES] __attribute__((aligned(64)));
static uint8_t scales_e8m0[16] __attribute__((aligned(16)));
static int8_t output_int8[TOTAL_OUTPUT_BYTES] __attribute__((aligned(64)));

static uint8_t gen_scale_e8m0(int block_idx) {
  return (uint8_t)(E8M0_BIAS + block_idx);
}

static int e8m0_to_shift(uint8_t raw) { return (int)raw - E8M0_BIAS; }

static uint8_t gen_fp4(int block_idx, int elem_idx) {
  return (uint8_t)((elem_idx + block_idx) & 0x0F);
}

static int fp4_to_int(uint8_t raw) { return (int)raw - 8; }

static int8_t golden_result(int block_idx, int elem_idx) {
  int val = fp4_to_int(gen_fp4(block_idx, elem_idx));
  int shift = e8m0_to_shift(gen_scale_e8m0(block_idx));
  int prod = val << shift;
  if (prod > 127)
    prod = 127;
  if (prod < -128)
    prod = -128;
  return (int8_t)prod;
}

static void generate_inputs(void) {
  for (int block = 0; block < NUM_BLOCKS; block++) {
    for (int e = 0; e < FP4_PER_BLOCK; e += 2) {
      uint8_t lo = gen_fp4(block, e);
      uint8_t hi = gen_fp4(block, e + 1);
      int byte_idx = block * BYTES_PER_BLOCK + (e / 2);
      input_mxfp4[byte_idx] = (uint8_t)((hi << 4) | (lo & 0x0F));
    }
  }
}

static void generate_scales(void) {
  for (int i = 0; i < 16; i++) {
    scales_e8m0[i] = 0;
  }
  for (int block = 0; block < NUM_BLOCKS; block++) {
    scales_e8m0[block] = gen_scale_e8m0(block);
  }
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  printf("=== MXFP4 to INT8 Test ===\n");

  generate_inputs();
  generate_scales();

  mmio_allocator_t mmio_alloc;
  mmio_allocator_init(&mmio_alloc);
  uint16_t mmio_addr = mmio_allocator_alloc(&mmio_alloc, 1);
  if (mmio_addr == (uint16_t)-1) {
    printf("FAIL: MMIO allocation failed\n");
    return 1;
  }

  uint32_t in_bank_id = 0;
  uint32_t out_bank_id = 1;
  bb_mem_alloc(in_bank_id, 1, 1);
  bb_mem_alloc(out_bank_id, 1, 1);

  bb_mmio_set(out_bank_id, mmio_addr, 1);
  bb_mvin_mmio((uintptr_t)scales_e8m0, mmio_addr, 1, TOTAL_SCALE_BYTES);
  bb_mvin((uintptr_t)input_mxfp4, in_bank_id, NUM_BLOCKS, 1);
  bb_mxfp2int(in_bank_id, out_bank_id, NUM_BLOCKS);
  bb_mvout((uintptr_t)output_int8, out_bank_id, NUM_BLOCKS * 2, 1);
  bb_fence();

  int passed = 1;
  int mismatch = 0;
  for (int block = 0; block < NUM_BLOCKS; block++) {
    for (int e = 0; e < FP4_PER_BLOCK; e++) {
      int out_idx = block * FP4_PER_BLOCK + e;
      int8_t expected = golden_result(block, e);
      int8_t actual = output_int8[out_idx];
      if (actual != expected) {
        if (mismatch < 8) {
          printf("MISMATCH block=%d elem=%d: got %d, expected %d\n", block, e,
                 (int)actual, (int)expected);
        }
        mismatch++;
        passed = 0;
      }
    }
  }

  bb_mmio_set(out_bank_id, 0, 0);
  bb_mem_release(in_bank_id);
  bb_mem_release(out_bank_id);
  mmio_allocator_free(&mmio_alloc, mmio_addr, 1);

  if (passed) {
    printf("MXFP2Int test PASSED\n");
  } else {
    printf("MXFP2Int test FAILED (%d mismatches)\n", mismatch);
  }

#ifdef MULTICORE
  exit(0);
#endif

  return !passed;
}
