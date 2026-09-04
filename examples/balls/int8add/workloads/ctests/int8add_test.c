#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int8add.h>
#include <stdint.h>
#include <stdio.h>

enum { ROWS = 7, VALUES = ROWS * 16 };

static int8_t lhs[VALUES] __attribute__((aligned(64)));
static int8_t rhs[VALUES] __attribute__((aligned(64)));
static int8_t output[VALUES] __attribute__((aligned(64)));

static int8_t expected(int8_t a, int8_t b, int relu) {
  float value = (float)a * 0.5f + (float)b * 0.25f;
  int integer = (int)value;
  float fraction = value - (float)integer;
  if (fraction > 0.5f || (fraction == 0.5f && (integer & 1)))
    ++integer;
  if (fraction < -0.5f || (fraction == -0.5f && (integer & 1)))
    --integer;
  if (relu && integer < 0)
    integer = 0;
  if (integer < -128)
    integer = -128;
  if (integer > 127)
    integer = 127;
  return (int8_t)integer;
}

static int run(int relu) {
  for (int i = 0; i < VALUES; ++i) {
    lhs[i] = (int8_t)((i * 37 & 255) - 128);
    rhs[i] = (int8_t)((i * 19 & 255) - 128);
    output[i] = 23;
  }
  bb_mvin((uintptr_t)lhs, 0, ROWS, 1);
  bb_mvin((uintptr_t)rhs, 1, ROWS, 1);
  if (relu)
    bb_int8add_relu(0, 1, 2, ROWS, 0.5f, 0.25f);
  else
    bb_int8add(0, 1, 2, ROWS, 0.5f, 0.25f);
  bb_mvout((uintptr_t)output, 2, ROWS, 1);
  bb_fence();
  for (int i = 0; i < VALUES; ++i) {
    int8_t want = expected(lhs[i], rhs[i], relu);
    if (output[i] != want) {
      printf("int8add mismatch relu=%d index=%d got=%d expected=%d\n", relu, i,
             output[i], want);
      return 1;
    }
  }
  return 0;
}

int main(void) {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  if (run(0) || run(1))
    return 1;
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  printf("int8add PASS\n");
  return 0;
}
