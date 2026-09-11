#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/lut.h>
#include <stdint.h>
#include <stdio.h>

enum { LANES = BANK_WIDTH / 8, ROWS = 4, VALUES = ROWS * LANES };
static int8_t input[VALUES] __attribute__((aligned(64)));
static int8_t table[64][64] __attribute__((aligned(64)));
static int8_t output[VALUES] __attribute__((aligned(64)));

int main(void) {
  _Static_assert(LANES == 16, "LutBall has sixteen lane tables");
  for (int row = 0; row < 64; ++row)
    for (int col = 0; col < 64; ++col) {
      int channel = (col / 16) * 4 + row / 16;
      int index = (row % 16) * 16 + col % 16;
      table[row][col] = (int8_t)(index + channel * 17 - 128);
    }
  for (int i = 0; i < VALUES; ++i)
    input[i] = (int8_t)(i * 29 - 128);
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, ROWS, 1);
  bb_mvin((uintptr_t)table, 1, 64, 1);
  bb_lut(0, 1, 2, ROWS);
  bb_mvout((uintptr_t)output, 2, ROWS, 1);
  bb_fence();
  for (int i = 0; i < VALUES; ++i) {
    int lane = i % LANES;
    int index = (uint8_t)input[i];
    int8_t expected = (int8_t)(index + lane * 17 - 128);
    if (output[i] != expected) {
      printf("lut lane mismatch index=%d\n", i);
      return 1;
    }
  }
  printf("lut lane PASS\n");
  return 0;
}
