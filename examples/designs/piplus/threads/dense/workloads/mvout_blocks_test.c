#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* The two block move outs of a pi0 vision Linear on its thread: C of 43 rows
 * of 12 bank rows (after 12 bias rows) to 43 DDR rows 288 bank rows (4608
 * bytes) apart, the first 8 bank rows of each in one instruction and the
 * last 4 in another; every block in its own 4 KB page. */
#define TOKENS 43
#define PITCH 12
#define STRIDE 288
#define BANK 7
static uint8_t input[(TOKENS + 1) * PITCH * 16] __attribute__((aligned(64)));
/* Outside .bss and checked a word at a time: the CPU of a dense hart is slow.
 */
static uint8_t output[TOKENS * STRIDE * 16]
    __attribute__((section(".noinit"), aligned(4096)));

static void run(void *unused) {
  (void)unused;
  bb_mem_alloc(BANK, 1, 1);
  bb_mvin_group((uintptr_t)input, BANK, 0, (TOKENS + 1) * PITCH, 1);
  bb_mvout_blocks((uintptr_t)output, BANK, 0, PITCH, PITCH, TOKENS * 8, STRIDE,
                  3);
  bb_mvout_blocks((uintptr_t)output + 128, BANK, 0, PITCH + 8, PITCH,
                  TOKENS * 4, STRIDE, 2);
  bb_mem_release(BANK);
}

int main(void) {
  for (unsigned i = 0; i < sizeof(input); ++i)
    input[i] = (uint8_t)(i * 7 + i / 4096);
  memset(output, 0xa5, sizeof(output));
  bb_thread_run(run, 0);
  for (unsigned t = 0; t < TOKENS; ++t) {
    const uint8_t *row = &output[t * STRIDE * 16];
    if (memcmp(row, &input[(t + 1) * PITCH * 16], PITCH * 16) != 0) {
      printf("mvout blocks: row %u differs\n", t);
      return 1;
    }
    const uint64_t *gap = (const uint64_t *)(row + PITCH * 16);
    for (unsigned w = 0; w < (STRIDE - PITCH) * 2; ++w)
      if (gap[w] != UINT64_C(0xa5a5a5a5a5a5a5a5)) {
        printf("mvout blocks: row %u written past its blocks\n", t);
        return 1;
      }
  }
  printf("mvout blocks PASSED\n");
  return 0;
}
