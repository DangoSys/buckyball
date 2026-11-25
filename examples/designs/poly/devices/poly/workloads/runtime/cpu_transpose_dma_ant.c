#include "cpu_transpose_dma.h"
#include <bbhw/isa/dma_strided.h>
#include <bbhw/isa/isa.h>
uint64_t entry(const struct transpose_dma_args *args) {
  bb_mset(1, 1, 1, 1);
  if (args->phase == 0) {
    bb_mvin_strided(args->input, 1, 0, 0, 1, 32, 4, 4, 0);
    bb_mvout_strided(args->output, 1, 0, 0, 1, 32, 512, 4, 0);
  } else {
    bb_mvin(args->input, 1, 8, 1);
    bb_mvout(args->output, 1, 8, 1);
  }
  bb_mset(1, 0, 0, 0);
  return 0;
}
