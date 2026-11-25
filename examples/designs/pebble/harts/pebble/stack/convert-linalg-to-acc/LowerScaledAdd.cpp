#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "int8addOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace ::buddy::acc;

namespace pebble {
LogicalResult lowerScaledAdd(::buddy::frontend::ScaledAddI8Op op) {
  auto type = cast<RankedTensorType>(op.getResult().getType());
  if (!type.hasStaticShape() || type.getRank() != 2 ||
      op.getLhs().getType() != type || op.getRhs().getType() != type ||
      type.getNumElements() % 16 || type.getNumElements() == 0)
    return op.emitOpError(
        "requires matching static INT8 matrices with whole bank rows");
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto index = [&](int64_t x) -> Value {
    return arith::ConstantIndexOp::create(b, loc, x);
  };
  int64_t elements = type.getNumElements();
  int64_t rows =
      std::min<int64_t>(target::getTarget().bankDepth,
                        (target::getTarget().dmaMaxRanges - 1) * 4096 / 16);
  while ((elements / 16) % rows)
    --rows;
  SmallVector<Value> inputs;
  for (Value value : {op.getLhs(), op.getRhs()}) {
    Value memory = tensorBuffer(b, loc, value);
    inputs.push_back(memref::CollapseShapeOp::create(
        b, loc, memory, ArrayRef<ReassociationIndices>{{0, 1}}));
  }
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(type.getShape(), b.getI8Type()));
  Value flatOutput = memref::CollapseShapeOp::create(
      b, loc, output, ArrayRef<ReassociationIndices>{{0, 1}});
  Value lhsRatio = arith::ConstantOp::create(b, loc, op.getLhsRatioAttr());
  Value rhsRatio = arith::ConstantOp::create(b, loc, op.getRhsRatioAttr());
  SmallVector<Value> banks;
  for (unsigned i = 0; i < 3; ++i)
    banks.push_back(BankAllocOp::create(b, loc, b.getI64Type(), 1, 1));
  auto loop = scf::ForOp::create(b, loc, index(0), index(elements),
                                 index(rows * 16), banks);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    SmallVector<Value> loaded;
    for (unsigned i = 0; i < 2; ++i) {
      Value memory = memref::SubViewOp::create(
          b, loc, inputs[i], ArrayRef<OpFoldResult>{loop.getInductionVar()},
          ArrayRef<OpFoldResult>{b.getIndexAttr(rows * 16)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      loaded.push_back(BankMvinOp::create(b, loc, b.getI64Type(), memory,
                                          loop.getRegionIterArgs()[i], rows,
                                          1));
    }
    Value result = ::buddy::int8add_ball::BankAddI8Op::create(
        b, loc, b.getI64Type(), loaded[0], loaded[1],
        loop.getRegionIterArgs()[2], lhsRatio, rhsRatio, rows, false);
    Value memory = memref::SubViewOp::create(
        b, loc, flatOutput, ArrayRef<OpFoldResult>{loop.getInductionVar()},
        ArrayRef<OpFoldResult>{b.getIndexAttr(rows * 16)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
    result =
        BankMvoutOp::create(b, loc, b.getI64Type(), memory, result, rows, 1);
    scf::YieldOp::create(b, loc, ValueRange{loaded[0], loaded[1], result});
  }
  for (Value bank : loop.getResults())
    BankReleaseOp::create(b, loc, bank);
  Value result =
      bufferization::ToTensorOp::create(b, loc, type, output, true, true);
  op.getResult().replaceAllUsesWith(result);
  op.erase();
  return success();
}
} // namespace pebble
