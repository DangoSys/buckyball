#include "mlir.h"

void _mlir_ciface_window(struct mlir_memref2 *, struct mlir_memref2 *,
                         struct mlir_memref2 *);

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
uint64_t entry(const struct mlir_mxmm_args *args) {
  struct mlir_memref2 a, b, c;
  describe(&a, args->a, 5, 16);
  describe(&b, args->b, 33, 16);
  describe(&c, args->c, 5, 16);
  _mlir_ciface_window(&a, &b, &c);
  return 0;
}
