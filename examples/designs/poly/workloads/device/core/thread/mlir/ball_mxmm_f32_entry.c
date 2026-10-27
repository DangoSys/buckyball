#include "mlir.h"
#include <bbhw/isa/isa.h>

void _mlir_ciface_single(void);

// A in bank 1, B in bank 2, C in bank 3.
uint64_t entry(const struct mlir_mxmm_args *args) {
  bb_mset(1, 1, 1, 1);
  bb_mset(2, 1, 1, 1);
  bb_mset(3, 1, 1, 1);
  bb_mvin(args->a, 1, 16, 1);
  bb_mvin(args->b, 2, 16, 1);
  bb_mvin(args->c, 3, 65, 1);
  _mlir_ciface_single();
  bb_mvout(args->c, 3, 65, 1);
  bb_mset(1, 0, 0, 0);
  bb_mset(2, 0, 0, 0);
  bb_mset(3, 0, 0, 0);
  return 0;
}
