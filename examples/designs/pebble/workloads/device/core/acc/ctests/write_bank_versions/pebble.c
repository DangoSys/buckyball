#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/int8add.h>
#include <isa/transpose.h>
#include <multithread.h>
#include <stdint.h>
#include <stdio.h>

enum { ROWS = 16, NEW_ROWS = 8, BYTES = ROWS * BANK_WIDTH / 8 };
static int8_t input[BB_CORE_NUM][BYTES] __attribute__((aligned(64)));
static int8_t left[BB_CORE_NUM][BYTES] __attribute__((aligned(64)));
static int8_t right[BB_CORE_NUM][BYTES] __attribute__((aligned(64)));
static volatile int8_t warmup[BB_CORE_NUM][BYTES] __attribute__((aligned(64)));
static int8_t output[BB_CORE_NUM][BYTES] __attribute__((aligned(64)));

extern "C" int pebble_pipeline(uint32_t instance) {
  int c = instance;
  for (int i = 0; i < BYTES; ++i) {
    input[c][i] = 1;
    left[c][i] = 3;
    right[c][i] = 4;
  }
  for (int bank = 0; bank < 5; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)input[c], 0, ROWS, 1);
  bb_mvin((uintptr_t)left[c], 3, ROWS, 1);
  bb_mvin((uintptr_t)right[c], 4, ROWS, 1);
  bb_mvout((uintptr_t)warmup[c], 4, ROWS, 1);
  if (warmup[c][0] != 4)
    return 1;
  bb_transpose(0, 2, ROWS, 8);
  bb_int8add(3, 4, 2, NEW_ROWS, 1.0f, 1.0f);
  bb_mvout((uintptr_t)output[c], 2, NEW_ROWS, 1);
  for (int i = 0; i < NEW_ROWS * BANK_WIDTH / 8; ++i)
    if (output[c][i] != 7)
      return 1;
  for (int bank = 0; bank < 5; ++bank)
    bb_mem_release(bank);
  printf("write_bank_versions PASS\n");
  return 0;
}
