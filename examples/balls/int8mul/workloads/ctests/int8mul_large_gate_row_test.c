#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int8mul.h>
#include <stdint.h>
#include <stdio.h>

enum { GATE_ROW = 1024, ROWS = 4 };

static int8_t gate[(GATE_ROW + 1) * 16] __attribute__((aligned(64)));
static int8_t input[ROWS * 16] __attribute__((aligned(64)));
static int8_t output[ROWS * 16] __attribute__((aligned(64)));

int main(void) {
  if (BANK_LINES <= GATE_ROW) {
    printf("int8mul large gate row SKIP bank_lines=%d\n", BANK_LINES);
    return 0;
  }

  for (int lane = 0; lane < 16; ++lane)
    gate[GATE_ROW * 16 + lane] = lane - 8;
  for (int i = 0; i < ROWS * 16; ++i)
    input[i] = i % 7 - 3;

  for (int bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)gate, 0, GATE_ROW + 1, 1);
  bb_mvin((uintptr_t)input, 1, ROWS, 1);
  bb_int8mul(0, 1, 2, ROWS, 1.0f, GATE_ROW);
  bb_mvout((uintptr_t)output, 2, ROWS, 1);
  bb_fence();

  for (int row = 0; row < ROWS; ++row)
    for (int lane = 0; lane < 16; ++lane) {
      int expected = gate[GATE_ROW * 16 + lane] * input[row * 16 + lane];
      if (expected > 127)
        expected = 127;
      if (expected < -128)
        expected = -128;
      int index = row * 16 + lane;
      if (output[index] != expected) {
        printf(
            "int8mul large gate row FAIL row=%d lane=%d got=%d expected=%d\n",
            row, lane, output[index], expected);
        return 1;
      }
    }
  printf("int8mul large gate row PASS gate_row=%d\n", GATE_ROW);
  return 0;
}
