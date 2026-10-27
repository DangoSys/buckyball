#include "ant_bank.h"
#include <bbhw/isa/isa.h>

uint64_t entry(const struct ant_bank_args *args) {
  bb_mset(3, 1, 1, 1);
  bb_mvin(args->input, 3, THREAD_MEM_BANK_ENTRIES, 1);
  bb_mvout(args->private_out, 3, THREAD_MEM_BANK_ENTRIES, 1);
  bb_mset(3, 0, 0, 0);
  bb_mset(THREAD_FRONTEND_SHARED_VBANK_BASE, 1, 1, 1);
  bb_mvin(args->input, THREAD_FRONTEND_SHARED_VBANK_BASE, args->shared_rows, 1);
  bb_mvout(args->shared_out, THREAD_FRONTEND_SHARED_VBANK_BASE,
           args->shared_rows, 1);
  bb_mset(THREAD_FRONTEND_SHARED_VBANK_BASE, 0, 0, 0);
  bb_mset(3, 1, 1, 1);
  bb_mvin_2d(args->input, 3, 2, 16, 2, 0, 2, 8);
  bb_mvout(args->tail_out, 3, 4, 1);
  bb_mset(3, 0, 0, 0);
  return 0;
}
