#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int8add.h>
#include <stdint.h>
#include <stdio.h>

enum { ROWS = 7, LANES = BANK_WIDTH / 8, VALUES = ROWS * LANES };
static int8_t lhs[VALUES] __attribute__((aligned(64)));
static int8_t rhs[VALUES] __attribute__((aligned(64)));
static int8_t output[VALUES] __attribute__((aligned(64)));

static int round_even(float value) {
  int integer = (int)value;
  float fraction = value - integer;
  if (fraction > .5f || (fraction == .5f && (integer & 1)))
    ++integer;
  if (fraction < -.5f || (fraction == -.5f && (integer & 1)))
    --integer;
  return integer;
}

int main(void) {
  for (int i = 0; i < VALUES; ++i) {
    lhs[i] = (int8_t)((i * 37 & 255) - 128);
    rhs[i] = (int8_t)((i * 19 & 255) - 128);
  }
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)lhs, 0, ROWS, 1);
  bb_mvin((uintptr_t)rhs, 1, ROWS, 1);
  bb_int8add(0, 1, 2, ROWS, 0.5f, 0.25f);
  bb_mvout((uintptr_t)output, 2, ROWS, 1);
  bb_fence();
  for (int i = 0; i < VALUES; ++i) {
    int expected = round_even(lhs[i] * .5f + rhs[i] * .25f);
    if (output[i] != expected) {
      printf("int8add scaled mismatch index=%d\n", i);
      return 1;
    }
  }
  printf("int8add scaled PASS\n");
  return 0;
}
