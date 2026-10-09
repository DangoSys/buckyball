#include "ant_bank.h"
#include <bbhw/isa/isa.h>

uint64_t ant_main(const struct ant_bank_args *args) {
  bb_mset(3, 1, 1, 1);
  bb_mvin(args->input, 3, BANK_LINES, 1);
  bb_mvout(args->private_out, 3, BANK_LINES, 1);
  bb_fence();
  bb_mset(3, 0, 0, 0);
  bb_mset(BB_SHARED_BANK_BASE, 1, 1, 1);
  bb_mvin(args->input, BB_SHARED_BANK_BASE, args->shared_rows, 1);
  bb_mvout(args->shared_out, BB_SHARED_BANK_BASE, args->shared_rows, 1);
  bb_fence();
  bb_mset(BB_SHARED_BANK_BASE, 0, 0, 0);
  bb_mset(3, 1, 1, 1);
  bb_mvin_2d(args->input, 3, 2, 16, 2, 0, 2, 8);
  bb_mvout(args->tail_out, 3, 4, 1);
  bb_fence();
  bb_mset(3, 0, 0, 0);
  return 0;
}
