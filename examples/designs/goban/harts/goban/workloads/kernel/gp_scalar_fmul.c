#include <bbhw/isa/isa.h>
#include <stdint.h>

uint64_t entry(uint64_t *args) {
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_transfer(2, 4);
  bb_mem_transfer(3, 4);
  bb_mem_transfer(0, 5);
  bb_mem_transfer(1, 5);
  bb_mvin_group(args[3], 5, 1, 1, 1);
  bb_mvin_group(args[2], 5, 0, sizeof(struct operator_launch) / 16, 1);
  mvin_operator((const void *)args[0], args[1], HART_VIRTUAL_BANK_COUNT);
  run_operator(4, HART_VIRTUAL_BANK_COUNT, 5, 0);
  bb_mvout_group(args[3], 5, 1, 1, 1);
  release_operator(HART_VIRTUAL_BANK_COUNT);
  bb_mem_release(4);
  bb_mem_release(5);
  return 0;
}
