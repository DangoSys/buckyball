#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int8mul.h>
#include <stdint.h>
#include <stdio.h>

enum { GATE_ROW = 1, ROWS = 7, LANES = 16, VALUES = ROWS * LANES };

static int8_t gate[2 * LANES] __attribute__((aligned(64)));
static int8_t input[VALUES] __attribute__((aligned(64)));
static int8_t output[VALUES] __attribute__((aligned(64)));

static int8_t expected(int8_t gate_value, int8_t input_value) {
  float value = (float)gate_value * (float)input_value * 0.25f;
  int integer = (int)value;
  float fraction = value - (float)integer;
  if (fraction > 0.5f || (fraction == 0.5f && (integer & 1)))
    ++integer;
  if (fraction < -0.5f || (fraction == -0.5f && (integer & 1)))
    --integer;
  if (integer < -128)
    integer = -128;
  if (integer > 127)
    integer = 127;
  return (int8_t)integer;
}

int main(void) {
  for (int lane = 0; lane < LANES; ++lane) {
    gate[lane] = 1;
    gate[GATE_ROW * LANES + lane] = (int8_t)(lane - 8);
  }
  for (int i = 0; i < VALUES; ++i) {
    input[i] = (int8_t)((i * 37 & 255) - 128);
    output[i] = 23;
  }

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)gate, 0, 2, 1);
  bb_mvin((uintptr_t)input, 1, ROWS, 1);
  bb_int8mul(0, 1, 2, ROWS, 0.25f, GATE_ROW);
  bb_mvout((uintptr_t)output, 2, ROWS, 1);
  bb_fence();

  for (int i = 0; i < VALUES; ++i) {
    int8_t want = expected(gate[GATE_ROW * LANES + i % LANES], input[i]);
    if (output[i] != want) {
      printf("int8mul mismatch index=%d got=%d expected=%d\n", i, output[i],
             want);
      return 1;
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  printf("int8mul PASS\n");
  return 0;
}
