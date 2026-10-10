#include <bbhw/isa/isa.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>
#include <tlink.h>

static const uint32_t input[2][4] __attribute__((aligned(64))) = {
    {1, 2, 3, 4}, {0x12345678, 0xabcdef01, 7, 8}};
static volatile uint32_t output[8] __attribute__((aligned(64)));

int main(void) {
  unsigned bank = BB_SHARED_BANK_BASE;
  bb_mem_alloc(bank, 1, 2);
  for (unsigned i = 0; i < 8; ++i)
    output[i] = 0xaaaaaaaa;
  for (unsigned group = 0; group < 2; ++group)
    bb_mvin_group((uintptr_t)input[group], bank, group, 1, 1);
  uint64_t address = tlink_shared_export(0, bank, 1);
  if (tlink_shared_export(0, bank, 1) != address)
    __builtin_trap();
  tlink_shared_release(0, bank, 1);
  if (tlink_read64(address) != UINT64_C(0xabcdef0112345678))
    __builtin_trap();
  tlink_write64(address + 8, UINT64_C(0x76543210deadbeef));
  bb_mvout((uintptr_t)output, bank, 1, 1);
  for (unsigned i = 0; i < 4; ++i)
    if (output[i] != input[0][i])
      __builtin_trap();
  if (output[4] != input[1][0] || output[5] != input[1][1] ||
      output[6] != 0xdeadbeef || output[7] != 0x76543210)
    __builtin_trap();
  tlink_shared_release(0, bank, 1);
  bb_mem_release(bank);
  puts("Main TLink lease and NPU shared storage PASS");
  return 0;
}
