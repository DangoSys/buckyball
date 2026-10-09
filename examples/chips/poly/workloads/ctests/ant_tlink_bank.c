#include "ant_tlink_bank.h"
#include <bbhw/isa/isa.h>
uint64_t ant_main(const struct ant_tlink_bank_args *args) {
  if (args->phase == 0) {
    bb_mset(BB_SHARED_BANK_BASE, 1, 1, 1);
    bb_mset(BB_SHARED_BANK_BASE + 1, 1, 1, 1);
    bb_mvin(args->input, BB_SHARED_BANK_BASE, TLINK_CROSS_ROWS, 1);
  } else if (args->phase == 1) {
    bb_mvout(args->output, BB_SHARED_BANK_BASE, TLINK_CROSS_ROWS, 1);
  } else if (args->phase == 2) {
    bb_mset(BB_SHARED_BANK_BASE, 0, 0, 0);
    bb_mset(BB_SHARED_BANK_BASE + 1, 0, 0, 0);
  } else
    __builtin_trap();
  return 0;
}
