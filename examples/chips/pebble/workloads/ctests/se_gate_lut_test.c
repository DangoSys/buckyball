#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int8mul.h>
#include <isa/lut.h>
#include <stdint.h>
#include <stdio.h>

enum { ROWS = 4, LANES = BANK_WIDTH / 8, VALUES = ROWS * LANES };
static int8_t gate[LANES] __attribute__((aligned(64)));
static int8_t input[VALUES] __attribute__((aligned(64)));
static int8_t table[256] __attribute__((aligned(64)));
static int8_t output[VALUES] __attribute__((aligned(64)));

int main(void) {
  for (int lane = 0; lane < LANES; ++lane)
    gate[lane] = 2;
  for (int i = 0; i < VALUES; ++i)
    input[i] = (int8_t)((i % LANES) * 2 - LANES);
  for (int i = 0; i < 256; ++i)
    table[i] = (int8_t)i;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mem_alloc(4, 1, 1);
  bb_mvin((uintptr_t)gate, 0, 1, 1);
  bb_mvin((uintptr_t)input, 1, ROWS, 1);
  bb_mvin((uintptr_t)table, 3, 16, 1);
  bb_int8mul(0, 1, 2, ROWS, .5f, 0);
  bb_lut(2, 3, 4, ROWS);
  bb_mvout((uintptr_t)output, 4, ROWS, 1);
  bb_fence();
  for (int i = 0; i < VALUES; ++i)
    if (output[i] != input[i]) {
      printf("se gate lut mismatch index=%d\n", i);
      return 1;
    }
  printf("se gate lut PASS\n");
  return 0;
}
