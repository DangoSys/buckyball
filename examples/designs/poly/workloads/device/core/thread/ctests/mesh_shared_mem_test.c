#include "ant_mesh.h"
#include <bbhw/isa/isa.h>
#include <params.h>

enum {
  BANKS = VIRTUAL_BANK_NUM - BB_SHARED_BANK_BASE,
  ROW_BYTES = BANK_WIDTH / 8
};
_Static_assert(
    BANK_WIDTH == 128 && BANKS > 0 && BANKS <= SHARED_BANK_NUM,
    "Shared bank test requires one physical bank per shared virtual bank");
uint64_t entry(const struct ant_mesh_args *args) {
  for (unsigned bank = 0; bank < BANKS; ++bank) {
    bb_mem_alloc(BB_SHARED_BANK_BASE + bank, 1, 1);
    bb_mvin(args->input + bank * ROW_BYTES, BB_SHARED_BANK_BASE + bank, 1, 1);
  }
  for (unsigned bank = 0; bank < BANKS; ++bank)
    bb_mvout(args->output + bank * ROW_BYTES, BB_SHARED_BANK_BASE + bank, 1, 1);
  for (unsigned bank = 0; bank < BANKS; ++bank)
    bb_mem_release(BB_SHARED_BANK_BASE + bank);
  return 0;
}
