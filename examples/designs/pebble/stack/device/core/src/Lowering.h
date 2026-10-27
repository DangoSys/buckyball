//===- Lowering.h - Custom computations in the Pebble midend -------------===//
#ifndef PEBBLE_LOWERING_H
#define PEBBLE_LOWERING_H

#include "Custom/CustomOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::linalg {
class GenericOp;
}
namespace pebble {
bool elementwise(mlir::linalg::GenericOp op, llvm::StringRef name);
mlir::Value tensorBuffer(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value tensor);

// Emits `body` for index = begin, begin + step, ... < end and threads the
// carried banks through it. The iterations share one scf.for unless `unroll`
// is set: then each iteration gets a constant index, for bodies whose index
// selects a bank.
llvm::SmallVector<mlir::Value>
repeat(mlir::OpBuilder &builder, mlir::Location loc, int64_t begin, int64_t end,
       int64_t step, bool unroll, mlir::ValueRange carried,
       llvm::function_ref<llvm::SmallVector<mlir::Value>(
           mlir::OpBuilder &, mlir::Value, mlir::ValueRange)>
           body);

// The bank capabilities the core tiles for: a matmul_i8 takes at most
// matmulRows() rows, im2col_i8 output feeds it in M tiles of tileRows().
int64_t matmulRows();
int64_t tileRows();

// Moves the cycle markers of `op`, whose result stays in `bank`, to the bank.
void observeResident(mlir::Operation *op, mlir::Value bank);

// The groups of the im2col_i8 output (and of the matching weights) for
// `windows` windows of a `kernel` x `kernel` convolution over `blocks`
// channel blocks: the fewest that divide the K tiles and fit a group each.
// Zero when they exceed the banks a convolution may use.
int64_t im2colGroups(int64_t kernel, int64_t blocks, int64_t windows);

// A convolution computes `side` x `side` output pixels (a multiple of the
// pool window) per matmul from an input tile of `inputSide` pixels per side,
// and reads the input channels in chunks of `chunk` channel blocks. An FP32
// input tile takes four lines per pixel. `side` is zero when no tile fits.
struct ConvTiling {
  int64_t side, inputSide, chunk;
};
ConvTiling convTiling(int64_t height, int64_t width, int64_t pool,
                      int64_t stride, int64_t kernel, int64_t blocks,
                      bool depthwise, bool floating);

// A Linear reduces K in `calls` matmul calls of `groups` groups of A and B,
// `blockK` columns each, over `rows` rows per tile.
struct LinearSplit {
  int64_t calls, groups, blockK;
};
LinearSplit linearSplit(int64_t k, int64_t rows, bool floating);

// The result of a Linear with custom.resident stays in `resident`, one
// group per sixteen columns; the others give `result`.
mlir::LogicalResult lowerLinear(::buddy::custom::CustomOp op,
                                mlir::Value residentInput, mlir::Value &result,
                                mlir::Value &resident);
mlir::LogicalResult lowerGlobalAverage(::buddy::custom::CustomOp op,
                                       mlir::Value &result);
mlir::LogicalResult lowerChannelOp(::buddy::custom::CustomOp op,
                                   mlir::Value &result);
mlir::LogicalResult lowerTranspose(::buddy::custom::CustomOp op,
                                   mlir::Value &result);
void formSpatialOps(mlir::func::FuncOp function);
// Lowers the convolutions; the ones whose results stay in banks remain in
// `consumed` until their consumers are lowered.
mlir::LogicalResult
lowerSpatialOps(mlir::func::FuncOp function,
                llvm::DenseMap<mlir::Value, mlir::Value> &residentVectors,
                llvm::SmallVector<mlir::Operation *> &consumed);
} // namespace pebble
#endif
