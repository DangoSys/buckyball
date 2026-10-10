#include <bbhw/isa/isa.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
enum { ROW_BYTES = BANK_WIDTH / 8 };
static uint8_t initial[6 * ROW_BYTES] __attribute__((aligned(16)));
static uint8_t replacement[3 * ROW_BYTES] __attribute__((aligned(16)));
static uint8_t full[6 * ROW_BYTES] __attribute__((aligned(16)));
static uint8_t selected[3 * ROW_BYTES] __attribute__((aligned(16)));
int main(int argc, char **argv) {
  for (unsigned i = 0; i < sizeof(initial); ++i)
    initial[i] = i + 3;
  memset(replacement, 211, sizeof(replacement));
  memset(selected, 97, sizeof(selected));
  bb_mem_alloc(0, 1, 3);
  bb_mvin((uintptr_t)initial, 0, 2, 1);
  if (argc > 1) {
    if (!strcmp(argv[1], "range"))
      bb_mvin_group((uintptr_t)replacement, 0, 3, 2, 1);
    else if (!strcmp(argv[1], "reserved"))
      BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(0) | BB_ITER(2),
                                FIELD((uintptr_t)selected, 0, 38) |
                                    FIELD(1, 39, 57) | (UINT64_C(1) << 58),
                                16);
    else
      return 2;
    puts("invalid group DMA was accepted");
    return 1;
  }
  bb_mvin_group((uintptr_t)replacement, 0, 1, 2, 2);
  bb_mvout_group((uintptr_t)selected, 0, 1, 2, 2);
  bb_mvout((uintptr_t)full, 0, 2, 1);
  for (unsigned row = 0; row < 2; ++row)
    for (unsigned group = 0; group < 3; ++group)
      for (unsigned byte = 0; byte < ROW_BYTES; ++byte) {
        unsigned i = (row * 3 + group) * ROW_BYTES + byte;
        if (full[i] != (group == 1 ? 211 : initial[i]))
          return 1;
      }
  for (unsigned i = 0; i < sizeof(selected); ++i)
    if (selected[i] != (i < ROW_BYTES || i >= 2 * ROW_BYTES ? 211 : 97))
      return 1;
  bb_mem_release(0);
  puts("GROUP DMA PASSED");
  return 0;
}
