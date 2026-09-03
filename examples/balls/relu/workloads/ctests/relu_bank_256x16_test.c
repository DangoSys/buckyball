#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/relu.h>
#include <stdio.h>

#define I32_LANES (BANK_WIDTH / 32)
#define COLUMNS 16
#define ROWS (BANK_LINES * I32_LANES / COLUMNS)
#define ITER BANK_LINES
_Static_assert(BANK_WIDTH % 32 == 0, "relu bank pack needs i32 lanes");
_Static_assert(COLUMNS % I32_LANES == 0, "relu bank pack needs even columns");
_Static_assert(ROWS *(COLUMNS / I32_LANES) == BANK_LINES,
               "relu bank test must fill one bank");

static result_t input[ROWS * COLUMNS] __attribute__((aligned(64)));
static result_t packed_input[ROWS * COLUMNS] __attribute__((aligned(64)));
static result_t packed_output[ROWS * COLUMNS] __attribute__((aligned(64)));
static result_t output[ROWS * COLUMNS] __attribute__((aligned(64)));

static void pack(const result_t *source, result_t *destination) {
  for (int row = 0; row < ROWS; ++row)
    for (int column = 0; column < COLUMNS; ++column) {
      int line = row * (COLUMNS / I32_LANES) + column / I32_LANES;
      destination[line * I32_LANES + column % I32_LANES] =
          source[row * COLUMNS + column];
    }
}

static void unpack(const result_t *source, result_t *destination) {
  for (int row = 0; row < ROWS; ++row)
    for (int column = 0; column < COLUMNS; ++column) {
      int line = row * (COLUMNS / I32_LANES) + column / I32_LANES;
      destination[row * COLUMNS + column] =
          source[line * I32_LANES + column % I32_LANES];
    }
}

int main(void) {
  const uint32_t bank = 0;

  for (int i = 0; i < ROWS * COLUMNS; ++i)
    input[i] = (i * 97 % 401) - 200;

  pack(input, packed_input);
  bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)packed_input, bank, ITER, 1);
  bb_relu(bank, 0, ITER, ITER);
  bb_mvout((uintptr_t)packed_output, bank, ITER, 1);
  bb_fence();
  bb_mem_release(bank);
  unpack(packed_output, output);

  for (int i = 0; i < ROWS * COLUMNS; ++i) {
    result_t expected = input[i] < 0 ? 0 : input[i];
    if (output[i] != expected) {
      printf("relu_bank_256x16 mismatch at %d: got %d expected %d\n", i,
             output[i], expected);
      return 1;
    }
  }
  printf("relu_bank_256x16 PASSED\n");
  return 0;
}
