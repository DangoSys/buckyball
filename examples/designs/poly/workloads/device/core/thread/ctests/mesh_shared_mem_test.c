#include "ant_mesh.h"
#include <bbhw/isa/isa.h>

enum {
  BANKS = CORE_VIRTUAL_BANK_COUNT - THREAD_FRONTEND_SHARED_VBANK_BASE,
  ROW_BYTES = THREAD_MEM_BANK_WIDTH / 8
};
_Static_assert(
    THREAD_MEM_BANK_WIDTH == 128 && BANKS > 0 && CORE_SHARED_MEM_ENABLE &&
        BANKS * CORE_SHARED_MEM_BANK_ENTRIES <= CORE_SHARED_MEM_ENTRIES,
    "Shared bank test requires one physical bank per shared virtual bank");
uint64_t entry(const struct ant_mesh_args *args) {
  for (unsigned bank = 0; bank < BANKS; ++bank) {
    bb_mem_alloc(THREAD_FRONTEND_SHARED_VBANK_BASE + bank, 1, 1);
    bb_mvin(args->input + bank * ROW_BYTES,
            THREAD_FRONTEND_SHARED_VBANK_BASE + bank, 1, 1);
  }
  for (unsigned bank = 0; bank < BANKS; ++bank)
    bb_mvout(args->output + bank * ROW_BYTES,
             THREAD_FRONTEND_SHARED_VBANK_BASE + bank, 1, 1);
  for (unsigned bank = 0; bank < BANKS; ++bank)
    bb_mem_release(THREAD_FRONTEND_SHARED_VBANK_BASE + bank);
  return 0;
}
