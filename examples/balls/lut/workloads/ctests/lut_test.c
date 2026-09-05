#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/lut.h>
#include <stdint.h>
#include <stdio.h>

static int8_t input[64] __attribute__((aligned(64)));
static int8_t table[256] __attribute__((aligned(64)));
static int8_t lane_table[16][256] __attribute__((aligned(64)));
static int8_t lane_table_packed[64][64] __attribute__((aligned(64)));
static int8_t output[64] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 256; ++i)
    table[i] = (int8_t)(((i * 73 + 19) ^ 0xa5) & 0xff);
  for (int i = 0; i < 64; ++i)
    input[i] = (int8_t)(i * 29 - 128);

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_mvin((uintptr_t)table, 1, 16, 1);
  bb_lut(0, 1, 2, 4);
  bb_mvout((uintptr_t)output, 2, 4, 1);
  bb_fence();

  for (int i = 0; i < 64; ++i) {
    int8_t expected = table[(uint8_t)input[i]];
    if (output[i] != expected) {
      printf("lut FAIL index=%d expected=%d actual=%d\n", i, expected,
             output[i]);
      return 1;
    }
  }

  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);

  for (int channel = 0; channel < 16; ++channel)
    for (int value = 0; value < 256; ++value)
      lane_table[channel][value] = (int8_t)(value + channel * 17 - 128);
  for (int row = 0; row < 64; ++row)
    for (int group = 0; group < 4; ++group)
      for (int byte = 0; byte < 16; ++byte)
        lane_table_packed[row][group * 16 + byte] =
            lane_table[group * 4 + row / 16][(row % 16) * 16 + byte];

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_mvin((uintptr_t)lane_table_packed, 1, 64, 1);
  bb_lut(0, 1, 2, 4);
  bb_mvout((uintptr_t)output, 2, 4, 1);
  bb_fence();

  for (int i = 0; i < 64; ++i) {
    int channel = i % 16;
    int8_t expected = lane_table[channel][(uint8_t)input[i]];
    if (output[i] != expected) {
      printf("lane lut FAIL index=%d channel=%d expected=%d actual=%d\n", i,
             channel, expected, output[i]);
      return 1;
    }
  }
  printf("lut PASS\n");
  return 0;
}
