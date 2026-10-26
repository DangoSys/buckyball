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
// selects a bank or sets a Ball attribute.
llvm::SmallVector<mlir::Value>
repeat(mlir::OpBuilder &builder, mlir::Location loc, int64_t begin, int64_t end,
       int64_t step, bool unroll, mlir::ValueRange carried,
       llvm::function_ref<llvm::SmallVector<mlir::Value>(
           mlir::OpBuilder &, mlir::Value, mlir::ValueRange)>
           body);

// Emits `body` over [0, total) in pieces of `size` that accumulate into one
// bank. The first piece starts the accumulation and the last completes it;
// the pieces between go through `repeat`.
mlir::Value
accumulate(mlir::OpBuilder &builder, mlir::Location loc, int64_t total,
           int64_t size, bool unroll, mlir::Value accumulator,
           llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Value,
                                          int64_t, bool, bool, mlir::Value)>
               body);

// A convolution computes tile x tile pooled output pixels per matmul; a
// resident image keeps side x side pixels of sixteen channels per bank. The
// tile divides the output and its input window fits the eight-pixel side that
// MAXPOOL addresses. The tile is zero when none fits.
struct SpatialTile {
  int64_t tile, side;
};
SpatialTile spatialTile(int64_t height, int64_t width, int64_t pool,
                        int64_t stride, int64_t kernel);

mlir::LogicalResult lowerLinearTiles(::buddy::custom::CustomOp op,
                                     mlir::Value &result);
mlir::LogicalResult lowerGlobalAverage(::buddy::custom::CustomOp op,
                                       mlir::Value &result);
mlir::LogicalResult lowerChannelOp(::buddy::custom::CustomOp op,
                                   mlir::Value &result);
mlir::LogicalResult lowerTranspose(::buddy::custom::CustomOp op,
                                   mlir::Value &result);
void formSpatialOps(mlir::func::FuncOp function);
mlir::LogicalResult
lowerSpatialOps(mlir::func::FuncOp function,
                llvm::DenseMap<mlir::Value, mlir::Value> &residentVectors,
                llvm::SmallVector<mlir::Operation *> &consumed);
} // namespace pebble
#endif
