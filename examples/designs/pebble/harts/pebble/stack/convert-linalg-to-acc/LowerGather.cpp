#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace ::buddy::frontend;
using namespace ::buddy::acc;

namespace pebble {
LogicalResult lowerGatherRows(GatherRowsOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  auto indicesType = cast<RankedTensorType>(op.getIndices().getType());
  if (!resultType.hasStaticShape() || resultType.getRank() != 2 ||
      indicesType.getShape() != ArrayRef<int64_t>{resultType.getDimSize(0)} ||
      op.getInputs().empty())
    return op.emitOpError(
        "requires row indices, matrices and a matching result matrix");
  auto index = [&](int64_t x) -> Value {
    return arith::ConstantIndexOp::create(b, loc, x);
  };
  Value zero = index(0), one = index(1);
  Type element = resultType.getElementType();
  int64_t columns = resultType.getDimSize(1), end = 0;
  int64_t rowBytes = target::getTarget().bankWidthBits / 8;
  int64_t bytes = columns * element.getIntOrFloatBitWidth() / 8;
  if (bytes <= 0 || bytes % rowBytes ||
      bytes > target::getTarget().bankDepth * rowBytes)
    return op.emitOpError(
        "requires rows occupying whole bank rows and fitting one bank");
  auto layout =
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                             {ShapedType::kDynamic, ShapedType::kDynamic});
  auto matrixType =
      MemRefType::get({ShapedType::kDynamic, columns}, element, layout);
  SmallVector<Value> matrices;
  SmallVector<int64_t> starts;
  for (Value tensor : op.getInputs()) {
    auto type = cast<RankedTensorType>(tensor.getType());
    if (!type.hasStaticShape() || type.getRank() != 2 ||
        type.getDimSize(1) != columns || type.getElementType() != element)
      return op.emitOpError(
          "requires matrices with the same columns and element type");
    Value memory = bufferization::ToBufferOp::create(
        b, loc, MemRefType::get(type.getShape(), element, layout), tensor,
        true);
    matrices.push_back(memref::CastOp::create(b, loc, matrixType, memory));
    starts.push_back(end);
    end += type.getDimSize(0);
  }
  Value indices = bufferization::ToBufferOp::create(
      b, loc, MemRefType::get(indicesType.getShape(), b.getI64Type()),
      op.getIndices(), true);
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(resultType.getShape(), element));
  Value temporary = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
  auto rows = scf::ForOp::create(b, loc, zero, index(resultType.getDimSize(0)),
                                 one, ValueRange{temporary});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(rows.getBody());
    Value row = rows.getInductionVar();
    Value at = memref::LoadOp::create(b, loc, indices, row);
    at = arith::IndexCastOp::create(b, loc, b.getIndexType(), at);
    Value source = matrices.back();
    Value local = arith::SubIOp::create(b, loc, at, index(starts.back()));
    for (int64_t i = matrices.size() - 2; i >= 0; --i) {
      Value before = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                           at, index(starts[i + 1]));
      auto selected = scf::IfOp::create(
          b, loc, TypeRange{matrixType, b.getIndexType()}, before, true);
      {
        OpBuilder::InsertionGuard branch(b);
        b.setInsertionPointToStart(&selected.getThenRegion().front());
        scf::YieldOp::create(
            b, loc,
            ValueRange{matrices[i],
                       arith::SubIOp::create(b, loc, at, index(starts[i]))});
        b.setInsertionPointToStart(&selected.getElseRegion().front());
        scf::YieldOp::create(b, loc, ValueRange{source, local});
      }
      source = selected.getResult(0);
      local = selected.getResult(1);
    }
    Value memory = memref::SubViewOp::create(
        b, loc, source, ArrayRef<OpFoldResult>{local, b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(columns)},
        SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
    Value bank = rows.getRegionIterArgs()[0];
    bank = BankMvinOp::create(b, loc, b.getI64Type(), memory, bank,
                              bytes / rowBytes, 1);
    memory = memref::SubViewOp::create(
        b, loc, output, ArrayRef<OpFoldResult>{row, b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(columns)},
        SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
    bank = BankMvoutOp::create(b, loc, b.getI64Type(), memory, bank,
                               bytes / rowBytes, 1);
    scf::YieldOp::create(b, loc, bank);
  }
  BankReleaseOp::create(b, loc, rows.getResult(0));
  Value result =
      bufferization::ToTensorOp::create(b, loc, resultType, output, true, true);
  op.getResult().replaceAllUsesWith(result);
  op.erase();
  return success();
}
} // namespace pebble
