#include "mlir.h"
#include <bbhw/isa/isa.h>

void _mlir_ciface_first(void);
void _mlir_ciface_last(void);

// A in bank 1, B in bank 2, C in bank 3; C after the first step goes to
// `between`, after the last to `c`.
uint64_t entry(const struct mlir_mxmm_chain_args *args) {
  bb_mset(1, 1, 1, 1);
  bb_mset(2, 1, 1, 1);
  bb_mset(3, 1, 1, 1);
  bb_mvin(args->a, 1, 33, 1);
  bb_mvin(args->b, 2, 33, 1);
  bb_mvin(args->c, 3, 65, 1);
  _mlir_ciface_first();
  bb_mvout(args->between, 3, 65, 1);
  _mlir_ciface_last();
  bb_mvout(args->c, 3, 65, 1);
  bb_mset(1, 0, 0, 0);
  bb_mset(2, 0, 0, 0);
  bb_mset(3, 0, 0, 0);
  return 0;
}
