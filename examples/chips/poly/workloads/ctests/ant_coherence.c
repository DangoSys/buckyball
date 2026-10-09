#include "ant_coherence.h"
#include <bbhw/isa/isa.h>

uint64_t ant_main(const struct ant_coherence_args *args) {
  if (args->phase == 0)
    bb_mset(3, 1, 1, 1);
  else if (args->phase == 1) {
    bb_mvin(args->input, 3, 1, 1);
    bb_mvout(args->output, 3, 1, 1);
  } else if (args->phase == 2)
    bb_mset(3, 0, 0, 0);
  else
    __builtin_trap();
  return 0;
}
