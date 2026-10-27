#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/mxfp2int.h>
#include <stdint.h>
#include <stdio.h>

// 20 blocks, so the scales span two rows of the scale bank. The scales cover
// right shifts (rounding toward minus infinity), left shifts that saturate
// and shifts of more than 31 in both directions.
#define NUM_BLOCKS 20
#define FP4_PER_BLOCK 32
#define BYTES_PER_BLOCK 16

static uint8_t input_mxfp4[NUM_BLOCKS * BYTES_PER_BLOCK]
    __attribute__((aligned(64)));
static uint8_t scales_e8m0[32] __attribute__((aligned(64)));
static int8_t output_int8[NUM_BLOCKS * FP4_PER_BLOCK]
    __attribute__((aligned(64)));
static const uint8_t scale_of[NUM_BLOCKS] = {0,   1,   90,  120, 124, 125, 126,
                                             127, 128, 129, 130, 131, 132, 133,
                                             134, 150, 158, 159, 200, 255};

static int raw_fp4(int block, int elem) { return (elem * 7 + block * 3) & 0xF; }

static int8_t golden(int block, int elem) {
  int value = raw_fp4(block, elem) - 8;
  int shift = scale_of[block] - 127;
  long scaled;
  if (shift >= 0)
    scaled = (long)value << (shift > 8 ? 8 : shift);
  else
    scaled = value >> (-shift > 31 ? 31 : -shift);
  if (scaled > 127)
    scaled = 127;
  if (scaled < -128)
    scaled = -128;
  return (int8_t)scaled;
}

int main(void) {
  for (int b = 0; b < NUM_BLOCKS; b++) {
    for (int e = 0; e < FP4_PER_BLOCK; e += 2)
      input_mxfp4[b * BYTES_PER_BLOCK + e / 2] =
          (uint8_t)((raw_fp4(b, e + 1) << 4) | raw_fp4(b, e));
    scales_e8m0[b] = scale_of[b];
  }

  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)input_mxfp4, 1, NUM_BLOCKS, 1);
  bb_mvin((uintptr_t)scales_e8m0, 2, 2, 1);
  bb_mxfp2int(1, 2, 3, NUM_BLOCKS);
  bb_mvout((uintptr_t)output_int8, 3, NUM_BLOCKS * 2, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);

  for (int b = 0; b < NUM_BLOCKS; b++)
    for (int e = 0; e < FP4_PER_BLOCK; e++) {
      int8_t got = output_int8[b * FP4_PER_BLOCK + e];
      int8_t exp = golden(b, e);
      if (got != exp) {
        printf("mxfp4 scale range mismatch b=%d e=%d got=%d exp=%d\n", b, e,
               (int)got, (int)exp);
        return 1;
      }
    }
  printf("mxfp4 e8m0 scale range PASSED\n");
  return 0;
}
