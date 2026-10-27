#pragma once
#include <stdint.h>

// Arguments of the thread programs that run the MxmmBall and MxquantBall MLIR
// tests: the DRAM addresses of their buffers.
struct mlir_mxmm_args {
  uint64_t a, b, c;
};
struct mlir_mxmm_chain_args {
  uint64_t a, b, c, between;
};
struct mlir_mxquant_args {
  uint64_t input, output;
};

// A two-dimensional memref descriptor as _mlir_ciface functions take it.
struct mlir_memref2 {
  uint64_t allocated, aligned;
  int64_t offset, sizes[2], strides[2];
};
