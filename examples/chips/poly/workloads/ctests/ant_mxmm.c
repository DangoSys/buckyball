#include "ant_mxmm.h"
#include <bbhw/isa/isa.h>
#include <isa/mxmm.h>

uint64_t ant_main(const struct ant_mxmm_args *args) {
  for (unsigned bank = 3; bank <= 5; ++bank)
    bb_mset(bank, 1, 1, 1);
  bb_mvin(args->a, 3, ANT_M * ANT_K * 4 / (BANK_WIDTH / 8), 1);
  bb_mvin(args->b, 4, ANT_N * ANT_K * 4 / (BANK_WIDTH / 8), 1);
  bb_mxmm_f32(3, 4, 5, ANT_M, ANT_N, ANT_K, 1, 1, 0);
  bb_mvout(args->output, 5, ANT_M * ANT_N * 4 / (BANK_WIDTH / 8), 1);
  for (unsigned bank = 3; bank <= 5; ++bank)
    bb_mset(bank, 0, 0, 0);
  return 0;
}
