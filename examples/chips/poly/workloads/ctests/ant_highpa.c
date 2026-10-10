#include "ant_highpa.h"
#include <bbhw/isa/isa.h>
uint64_t ant_main(const struct ant_highpa_args *args) {
  bb_mset(3, 1, 1, 1);
  bb_mvin(args->input, 3, 2, 1);
  bb_mvout(args->output, 3, 2, 1);
  bb_mset(3, 0, 0, 0);
  return 0;
}
