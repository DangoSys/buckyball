#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/maxpool.h>
#include <stdint.h>
#include <stdio.h>

/* Input linear key i is in group (i / 4) % groups, row i / (4 * groups),
 * lane i % 4. Output row j contains [key(2j), index(2j), key(2j+1),
 * index(2j+1)]. Keys descend; equal keys retain ascending original indices.
 * Only N keys are valid. The unused pair in an odd K's last row is zero. */
static uint32_t input[1024] __attribute__((aligned(64)));
static uint32_t output[20] __attribute__((aligned(64)));

int main(void) {
  const unsigned cases[][4] = {{5, 2, 37, 9},     {2, 1, 7, 7}, {2, 3, 19, 1},
                               {128, 2, 1021, 9}, {2, 1, 7, 2}, {2, 2, 16, 8}};
  for (unsigned test = 0; test < sizeof(cases) / sizeof(cases[0]); ++test) {
    unsigned rows = cases[test][0], groups = cases[test][1];
    unsigned n = cases[test][2], k = cases[test][3];
    for (unsigned i = 0; i < rows * groups * 4; ++i) {
      if (i >= n)
        input[i] = UINT32_MAX;
      else if (test == 0)
        input[i] = i % 5 == 0   ? UINT32_MAX
                   : i % 5 == 1 ? 0x80000000u
                   : i % 5 == 3 ? 0
                                : 17;
      else
        input[i] = test == 1 ? 0 : 0x80000000u + i;
    }
    for (unsigned i = 0; i < 20; ++i)
      output[i] = 0xa5a5a5a5u;

    bb_mem_alloc(0, 1, groups);
    bb_mem_alloc(1, 1, 1);
    bb_mvin((uintptr_t)input, 0, rows, 1);
    bb_topk_u32(0, 1, rows, n, k);
    bb_mvout((uintptr_t)output, 1, (k + 1) / 2, 1);
    for (unsigned rank = 0; rank < k; ++rank) {
      unsigned index = test == 0   ? (rank < 8 ? rank * 5 : 1)
                       : test == 1 ? rank
                       : test == 2 ? 18
                                   : n - 1 - rank;
      if (output[rank * 2] != input[index] || output[rank * 2 + 1] != index) {
        printf("topk_u32 FAIL case=%u rank=%u key=%u index=%u expected=%u/%u\n",
               test, rank, output[rank * 2], output[rank * 2 + 1], input[index],
               index);
        return 1;
      }
    }
    if ((k & 1) && (output[k * 2] != 0 || output[k * 2 + 1] != 0)) {
      printf("topk_u32 FAIL odd output padding\n");
      return 1;
    }
    bb_mem_release(0);
    bb_mem_release(1);
  }
  printf("topk_u32 PASS unsigned keys, ties, multiple groups and tails\n");
  return 0;
}
