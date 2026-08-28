#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/matadd.h>
#include <stdint.h>
#include <stdio.h>

#define LINES BANK_LINES
#define GROUPS 2
#define VALUES (LINES * GROUPS * 4)

static int32_t a[VALUES] __attribute__((aligned(64)));
static int32_t b[VALUES] __attribute__((aligned(64)));
static int32_t c[VALUES] __attribute__((aligned(64)));

int main(void) {
  const uint32_t a_bank = 0;
  const uint32_t b_bank = 1;
  const uint32_t c_bank = 2;

  for (int i = 0; i < VALUES; ++i) {
    a[i] = (i * 73) - 300000;
    b[i] = 200000 - (i * 41);
  }

  bb_mem_alloc(a_bank, 1, GROUPS);
  bb_mem_alloc(b_bank, 1, GROUPS);
  bb_mem_alloc(c_bank, 1, GROUPS);
  bb_mvin((uintptr_t)a, a_bank, LINES, 1);
  bb_mvin((uintptr_t)b, b_bank, LINES, 1);
  bb_matadd(a_bank, b_bank, c_bank, LINES);
  bb_mvout((uintptr_t)c, c_bank, LINES, 1);
  bb_fence();
  bb_mem_release(a_bank);
  bb_mem_release(b_bank);
  bb_mem_release(c_bank);

  for (int i = 0; i < VALUES; ++i) {
    uint32_t expected = (uint32_t)a[i] + (uint32_t)b[i];
    if ((uint32_t)c[i] != expected) {
      printf("matadd bank mismatch at %d: got %d expected %d\n", i, c[i],
             (int32_t)expected);
      return 1;
    }
  }
  printf("matadd_bank_1024x8_i32 PASSED\n");
  return 0;
}
