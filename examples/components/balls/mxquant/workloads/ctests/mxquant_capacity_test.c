#include "buckyball.h"
#include <dma.h>
#include <isa/mxquant.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { BANK_BYTES = BANK_LINES * BANK_WIDTH / 8, MAX_COUNT = BANK_BYTES / 4 };
static uint32_t input[MAX_COUNT] __attribute__((aligned(64)));
static uint8_t actual[BANK_BYTES] __attribute__((aligned(64)));

int main(void) {
  const int counts[] = {480, 960, MAX_COUNT};
  for (int test = 0; test < 3; ++test) {
    int count = counts[test];
    for (int i = 0; i < MAX_COUNT; ++i)
      input[i] = (i & 1) ? 0xbf800000u : 0x3f800000u;
    memset(actual, 0x5a, sizeof(actual));
    bb_mem_alloc(3, 1, 1);
    bb_mem_alloc(4, 1, 1);
    bb_mvin((uintptr_t)input, 3, BANK_LINES, 1);
    bb_mvin((uintptr_t)actual, 4, BANK_LINES, 1);
    bb_mxquant(3, 4, count);
    bb_mvout((uintptr_t)actual, 4, BANK_LINES, 1);
    for (int i = 0; i < BANK_BYTES; ++i) {
      uint8_t expected = i < count                ? ((i & 1) ? 0xf8 : 0x78)
                         : i < count + count / 32 ? 119
                                                  : 0x5a;
      if (actual[i] != expected)
        return 1;
    }
    bb_mem_release(3);
    bb_mem_release(4);
  }
  puts("mxquant 480/960/bank-capacity/tail-guard PASS");
  return 0;
}
