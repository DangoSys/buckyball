#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/relu.h>
#include <stdio.h>

#define ROWS 16
#define COLUMNS 16
#define ITER (ROWS * COLUMNS / 4)

static result_t input[ROWS * COLUMNS] __attribute__((aligned(64)));
static result_t packed_input[ROWS * COLUMNS] __attribute__((aligned(64)));
static result_t packed_output[ROWS * COLUMNS] __attribute__((aligned(64)));
static result_t output[ROWS * COLUMNS] __attribute__((aligned(64)));

static void pack(const result_t *source, result_t *destination) {
  for (int row = 0; row < ROWS; ++row) {
    for (int column = 0; column < COLUMNS; ++column) {
      int line = row * 4 + column / 4;
      destination[line * 4 + column % 4] = source[row * COLUMNS + column];
    }
  }
}

static void unpack(const result_t *source, result_t *destination) {
  for (int row = 0; row < ROWS; ++row) {
    for (int column = 0; column < COLUMNS; ++column) {
      int line = row * 4 + column / 4;
      destination[row * COLUMNS + column] = source[line * 4 + column % 4];
    }
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
      printf("relu mismatch at %d: got %d expected %d\n", i, output[i],
             expected);
      return 1;
    }
  }
  printf("relu_test PASSED\n");
  return 0;
}
