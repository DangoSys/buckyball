#include "matmul.h"
#include <bbhw/isa/isa.h>
#include <isa/smatmul.h>

uint64_t entry(const struct matmul_task *args) {
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mset(bank, 1, 1, 1);
  bb_mvin(args->bias, 0, 4, 1);
  bb_mvin(args->a, 1, 16, 1);
  bb_mvin(args->b, 2, 16, 1);
  bb_smatmul_bias(0, 0);
  bb_smatmul_os(1, 2, 3, 16, 16, 16, 1, 1, 0);
  bb_mvout(args->output, 3, 64, 1);
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mset(bank, 0, 0, 0);
  return 0;
}
