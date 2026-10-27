#include "ant_corelink_bank.h"
#include <bbhw/isa/isa.h>
uint64_t entry(const struct ant_corelink_bank_args *args) {
  if (args->phase == 0) {
    bb_mset(THREAD_FRONTEND_SHARED_VBANK_BASE, 1, 1, 1);
    bb_mset(THREAD_FRONTEND_SHARED_VBANK_BASE + 1, 1, 1, 1);
    bb_mvin(args->input, THREAD_FRONTEND_SHARED_VBANK_BASE, CORELINK_CROSS_ROWS,
            1);
  } else if (args->phase == 1) {
    bb_mvout(args->output, THREAD_FRONTEND_SHARED_VBANK_BASE,
             CORELINK_CROSS_ROWS, 1);
  } else if (args->phase == 2) {
    bb_mset(THREAD_FRONTEND_SHARED_VBANK_BASE, 0, 0, 0);
    bb_mset(THREAD_FRONTEND_SHARED_VBANK_BASE + 1, 0, 0, 0);
  } else
    __builtin_trap();
  return 0;
}
