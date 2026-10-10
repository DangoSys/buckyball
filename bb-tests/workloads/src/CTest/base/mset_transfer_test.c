#include <bbhw/isa/isa.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { ROW_BYTES = BANK_WIDTH / 8 };

static uint8_t first[2 * ROW_BYTES] __attribute__((aligned(16)));
static uint8_t second[4 * ROW_BYTES] __attribute__((aligned(16)));
static uint8_t replacement[2 * ROW_BYTES] __attribute__((aligned(16)));
static uint8_t result[6 * ROW_BYTES] __attribute__((aligned(16)));
static uint8_t zeros[BANK_NUM * ROW_BYTES] __attribute__((aligned(16)));

int main(int argc, char **argv) {
  for (unsigned i = 0; i < sizeof(first); ++i) {
    first[i] = i + 1;
    replacement[i] = 193;
  }
  for (unsigned i = 0; i < sizeof(second); ++i)
    second[i] = i + 65;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 2);
  bb_mvin((uintptr_t)first, 0, 2, 1);
  bb_mvin((uintptr_t)second, 1, 2, 1);
  if (argc > 1) {
    if (!strcmp(argv[1], "same-bank"))
      bb_mem_transfer(0, 0);
    else if (!strcmp(argv[1], "unallocated"))
      bb_mem_transfer(4, 3);
    else if (!strcmp(argv[1], "reserved"))
      BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(0) | BB_BANK2(3),
                                (UINT64_C(1) << 12) | (UINT64_C(1) << 10), 32);
    else
      return 2;
    puts("invalid transfer was accepted");
    return 1;
  }
  bb_mem_transfer(0, 3);
  bb_mem_transfer(1, 3);
#if defined(__linux__)
  if (dma_bank_cols(0) || dma_bank_cols(1) || dma_bank_cols(3) != 3)
    return 1;
#endif
  bb_mem_alloc(0, 1, 1);
  bb_mvin((uintptr_t)replacement, 0, 2, 1);
  bb_mvout((uintptr_t)result, 3, 2, 1);
  for (unsigned row = 0; row < 2; ++row) {
    if (memcmp(result + row * 3 * ROW_BYTES, first + row * ROW_BYTES,
               ROW_BYTES) ||
        memcmp(result + row * 3 * ROW_BYTES + ROW_BYTES,
               second + row * 2 * ROW_BYTES, 2 * ROW_BYTES))
      return 1;
  }
  bb_mem_release(0);
  bb_mem_release(3);
  bb_mset_clear(4, 1, BANK_NUM);
  bb_mvout((uintptr_t)zeros, 4, 1, 1);
  for (unsigned i = 0; i < sizeof(zeros); ++i)
    if (zeros[i])
      return 1;
  bb_mem_release(4);
  puts("MSET TRANSFER PASSED");
  return 0;
}
