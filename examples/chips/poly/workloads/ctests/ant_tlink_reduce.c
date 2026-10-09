#include "ant_tlink_reduce.h"
#include <bbhw/isa/isa.h>
#include <isa/mxmm.h>
uint64_t ant_main(const struct ant_tlink_reduce_args *args) {
  if (args->phase == 0) {
    bb_mset(BB_SHARED_BANK_BASE, 1, 1, 1);
    bb_mset(0, 1, 1, 1);
    bb_mset(1, 1, 1, 1);
    bb_mvin(args->input, 0, 4, 1);
    bb_mvin(args->input + 16 * sizeof(float), 1, 64, 1);
    bb_mxmm_fma32(0, 1, BB_SHARED_BANK_BASE, 1, 16, 16, 1, 1, 0);
    bb_mset(0, 0, 0, 0);
    bb_mset(1, 0, 0, 0);
  } else if (args->phase == 1) {
    bb_mvout(args->output, BB_SHARED_BANK_BASE, 2, 1);
  } else if (args->phase == 2) {
    bb_mset(BB_SHARED_BANK_BASE, 0, 0, 0);
  } else
    __builtin_trap();
  return 0;
}
