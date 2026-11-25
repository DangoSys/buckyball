//===- Lowering.h - Custom computations in the Pebble midend -------------===//
#ifndef PEBBLE_LOWERING_H
#define PEBBLE_LOWERING_H

#include "Acc/AccOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::linalg {
class GenericOp;
}
namespace pebble {
mlir::LogicalResult lowerFixedOps(mlir::func::FuncOp function);
mlir::LogicalResult lowerNonlinearOps(mlir::func::FuncOp function);
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

// The bank capabilities the hart tiles for: a matmul_i8 takes at most
// matmulRows() rows, im2col_i8 output feeds it in M tiles of tileRows().
int64_t matmulRows();
int64_t tileRows();

// Encoded byte stride and row count for one bounded strided DMA packet.
std::pair<mlir::Value, mlir::Value>
dmaRows(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value byteStride,
        int64_t blockBytes, int64_t maximumRows);

// Moves the cycle markers of `op`, whose result stays in `bank`, to the bank.
void observeResident(mlir::Operation *op, mlir::Value bank);

// The location of a fused computation: `result`, the location of the linalg
// operation whose value its result equals, and `quantization`, the location
// of the input quantization it absorbed; quantizationLoc reads the latter
// back, for a producer that takes over that quantization (the location of a
// custom operation given as input, which records none).
mlir::Location fusedLoc(mlir::Location result, mlir::Location quantization);
mlir::Location quantizationLoc(mlir::Operation *op);

// Fused logical regions share the producer's execution interval.
void moveTraceStarts(mlir::Operation *producer,
                     llvm::ArrayRef<mlir::Operation *> ends);

// The groups of the im2col_i8 output (and of the matching weights) for
// `windows` windows of a `kernel` x `kernel` convolution over `blocks`
// channel blocks: the fewest that divide the K tiles and fit a group each.
// Zero when they exceed the banks a convolution may use.
int64_t im2colGroups(int64_t kernel, int64_t blocks, int64_t windows);

// The channel blocks of a chunk of a convolution over `blocks` input channel
// blocks: the most, at most eight, that divide them; one for depthwise. The
// recipe frontend stores the weights of every other convolution in chunks of
// as many blocks (LowerSpatialOps).
int64_t convChunk(int64_t blocks, bool depthwise);

// Shared matrix schedule, also used by a single-pixel 1x1 convolution.
mlir::Value linearTiles(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value input, mlir::Value resident,
                        int64_t residentBlock, mlir::Value multiplier,
                        mlir::Value weight, mlir::Value bias,
                        mlir::Value scales, bool relu, mlir::Value output);

// A convolution computes `side` x `side` output pixels (a multiple of the
// pool window) per matmul from an input tile of `inputSide` pixels per side,
// and reads the input channels in chunks of `chunk` (convChunk) channel
// blocks. An FP32 input tile takes four lines per pixel. `side` is zero when
// no tile fits.
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

// The result of a Linear with bbacc.resident stays in `resident`, one
// group per sixteen columns; the others give `result`.
mlir::LogicalResult lowerLinear(::buddy::acc::CustomOp op,
                                mlir::Value residentInput, mlir::Value &result,
                                mlir::Value &resident);
mlir::LogicalResult lowerGlobalAverage(::buddy::acc::CustomOp op,
                                       mlir::Value &result);
mlir::LogicalResult lowerBatchMatmul(::buddy::acc::CustomOp op,
                                     mlir::Value &result);
mlir::LogicalResult lowerChannelOp(::buddy::acc::CustomOp op,
                                   mlir::Value &result);
mlir::LogicalResult lowerTranspose(::buddy::acc::CustomOp op,
                                   mlir::Value &result);
mlir::LogicalResult lowerMaxPool(::buddy::acc::CustomOp op,
                                 mlir::Value &result);
mlir::LogicalResult lowerPool(::buddy::acc::CustomOp op, mlir::Value &result);
mlir::LogicalResult lowerQuantize(mlir::func::FuncOp function);
mlir::LogicalResult lowerTopK(mlir::func::FuncOp function);
void formCustomOps(mlir::func::FuncOp function);
void formSpatialOps(mlir::func::FuncOp function);
// Lowers the convolutions; the ones whose results stay in banks remain in
// `consumed` until their consumers are lowered.
mlir::LogicalResult
lowerSpatialOps(mlir::func::FuncOp function,
                llvm::DenseMap<mlir::Value, mlir::Value> &residentVectors,
                llvm::SmallVector<mlir::Operation *> &consumed);
} // namespace pebble
#endif
