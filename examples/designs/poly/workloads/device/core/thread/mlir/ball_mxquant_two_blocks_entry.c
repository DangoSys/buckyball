#include "mlir.h"
#include <bbhw/isa/isa.h>

void _mlir_ciface_quant(void);

// The input in bank 1, the output in bank 2.
uint64_t entry(const struct mlir_mxquant_args *args) {
  bb_mset(1, 1, 1, 1);
  bb_mset(2, 1, 1, 1);
  bb_mvin(args->input, 1, 16, 1);
  bb_mvin(args->output, 2, 5, 1);
  _mlir_ciface_quant();
  bb_mvout(args->output, 2, 5, 1);
  bb_mset(1, 0, 0, 0);
  bb_mset(2, 0, 0, 0);
  return 0;
}
