#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ROW_BYTES 16
#define ROWS 4
#define OFFSET 4
#define GUARD 0x5a

static elem_t input_matrix[ROWS * ROW_BYTES] __attribute__((aligned(128)));
static uint8_t output_bytes[OFFSET + ROWS * ROW_BYTES + ROW_BYTES]
    __attribute__((aligned(128)));

static void init_inputs(void) {
  for (int i = 0; i < ROWS * ROW_BYTES; i++) {
    input_matrix[i] = (elem_t)((i * 11 + 7) & 0x7f);
  }
  for (int i = 0; i < (int)sizeof(output_bytes); i++) {
    output_bytes[i] = GUARD;
  }
}

static int check_output(void) {
  for (int i = 0; i < OFFSET; i++) {
    if (output_bytes[i] != GUARD) {
      printf("FAILED: prefix guard byte %d expected 0x%02x, got 0x%02x\n", i,
             GUARD, output_bytes[i]);
      return 0;
    }
  }

  for (int i = 0; i < ROWS * ROW_BYTES; i++) {
    uint8_t expected = (uint8_t)input_matrix[i];
    uint8_t got = output_bytes[OFFSET + i];
    if (got != expected) {
      printf("FAILED: payload byte %d expected 0x%02x, got 0x%02x\n", i,
             expected, got);
      return 0;
    }
  }

  for (int i = OFFSET + ROWS * ROW_BYTES; i < (int)sizeof(output_bytes); i++) {
    if (output_bytes[i] != GUARD) {
      printf("FAILED: suffix guard byte %d expected 0x%02x, got 0x%02x\n", i,
             GUARD, output_bytes[i]);
      return 0;
    }
  }

  return 1;
}

int mvin_mvout_unaligned_test(void) {
  uint32_t bank_id = 0;

  init_inputs();
  bb_mem_alloc(bank_id, 1, 1);
  bb_mvin((uintptr_t)input_matrix, bank_id, ROWS, 1);
  bb_mvout((uintptr_t)(output_bytes + OFFSET), bank_id, ROWS, 1);
  bb_fence();

  if (!check_output()) {
    printf("mvin/mvout unaligned test FAILED\n");
    return 0;
  }

  printf("mvin/mvout unaligned test PASSED\n");
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = mvin_mvout_unaligned_test();
#ifdef MULTICORE
  exit(0);
#endif
  return passed ? 0 : 1;
}
