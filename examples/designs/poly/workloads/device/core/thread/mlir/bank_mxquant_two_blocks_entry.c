#include "mlir.h"

void _mlir_ciface_quant(struct mlir_memref2 *, struct mlir_memref2 *);

static void describe(struct mlir_memref2 *memref, uint64_t address,
                     int64_t rows, int64_t columns) {
  memref->allocated = address;
  memref->aligned = address;
  memref->offset = 0;
  memref->sizes[0] = rows;
  memref->sizes[1] = columns;
  memref->strides[0] = columns;
  memref->strides[1] = 1;
}

// The MLIR function moves the buffers in and out of its banks itself.
uint64_t entry(const struct mlir_mxquant_args *args) {
  struct mlir_memref2 input, output;
  describe(&input, args->input, 16, 4);
  describe(&output, args->output, 5, 16);
  _mlir_ciface_quant(&input, &output);
  return 0;
}
