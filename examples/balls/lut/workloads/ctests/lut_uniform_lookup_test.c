#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/lut.h>
#include <stdint.h>
#include <stdio.h>

enum { LANES = BANK_WIDTH / 8, ROWS = 4, VALUES = ROWS * LANES };
static int8_t input[VALUES] __attribute__((aligned(64)));
static int8_t table[256] __attribute__((aligned(64)));
static int8_t output[VALUES] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 256; ++i)
    table[i] = (int8_t)(((i * 73 + 19) ^ 0xa5) & 0xff);
  for (int i = 0; i < VALUES; ++i)
    input[i] = (int8_t)(i * 29 - 128);
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, ROWS, 1);
  bb_mvin((uintptr_t)table, 1, 16, 1);
  bb_lut(0, 1, 2, ROWS);
  bb_mvout((uintptr_t)output, 2, ROWS, 1);
  bb_fence();
  for (int i = 0; i < VALUES; ++i)
    if (output[i] != table[(uint8_t)input[i]]) {
      printf("lut uniform mismatch index=%d\n", i);
      return 1;
    }
  printf("lut uniform PASS\n");
  return 0;
}
