//===- Lowering.h - Custom computations in the Pebble midend -------------===//
#ifndef PEBBLE_LOWERING_H
#define PEBBLE_LOWERING_H

#include "Custom/CustomOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::linalg {
class GenericOp;
}
namespace pebble {
bool elementwise(mlir::linalg::GenericOp op, llvm::StringRef name);
mlir::Value tensorBuffer(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value tensor);
mlir::LogicalResult lowerLinearTiles(::buddy::custom::CustomOp op,
                                     mlir::Value &result);
mlir::LogicalResult lowerGlobalAverage(::buddy::custom::CustomOp op,
                                       mlir::Value &result, bool resident);
mlir::LogicalResult lowerChannelOp(::buddy::custom::CustomOp op,
                                   mlir::Value &result);
mlir::LogicalResult lowerTranspose(::buddy::custom::CustomOp op,
                                   mlir::Value &result);
void formQuantizedChains(mlir::func::FuncOp function);
mlir::LogicalResult lowerQuantizedChains(mlir::func::FuncOp function);
void formSpatialOps(mlir::func::FuncOp function);
mlir::LogicalResult
lowerSpatialOps(mlir::func::FuncOp function,
                llvm::DenseMap<mlir::Value, mlir::Value> &residentVectors,
                llvm::SmallVector<mlir::Operation *> &consumed);
} // namespace pebble
#endif
