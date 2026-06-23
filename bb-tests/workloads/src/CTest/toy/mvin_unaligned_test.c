#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ROW_BYTES 16
#define ROWS 4
#define OFFSET 4
#define GUARD 0xa5

static uint8_t input_bytes[OFFSET + ROWS * ROW_BYTES + ROW_BYTES]
    __attribute__((aligned(128)));
static elem_t output_matrix[ROWS * ROW_BYTES] __attribute__((aligned(128)));

static void init_inputs(void) {
  for (int i = 0; i < (int)sizeof(input_bytes); i++) {
    input_bytes[i] = (uint8_t)((i * 13 + 5) & 0x7f);
  }
  for (int i = 0; i < ROWS * ROW_BYTES; i++) {
    output_matrix[i] = GUARD;
  }
}

static int check_output(void) {
  for (int i = 0; i < ROWS * ROW_BYTES; i++) {
    uint8_t expected = input_bytes[OFFSET + i];
    uint8_t got = (uint8_t)output_matrix[i];
    if (got != expected) {
      printf("FAILED: payload byte %d expected 0x%02x, got 0x%02x\n", i,
             expected, got);
      return 0;
    }
  }
  return 1;
}

int mvin_unaligned_test(void) {
  uint32_t bank_id = 0;

  init_inputs();
  bb_mem_alloc(bank_id, 1, 1);
  bb_mvin((uintptr_t)(input_bytes + OFFSET), bank_id, ROWS, 1);
  bb_mvout((uintptr_t)output_matrix, bank_id, ROWS, 1);
  bb_fence();

  if (!check_output()) {
    printf("mvin unaligned test FAILED\n");
    return 0;
  }

  printf("mvin unaligned test PASSED\n");
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = mvin_unaligned_test();
#ifdef MULTICORE
  exit(0);
#endif
  return passed ? 0 : 1;
}
