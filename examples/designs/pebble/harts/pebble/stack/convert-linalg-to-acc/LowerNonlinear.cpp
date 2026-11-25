#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "lutOps.h"
#include "maxpoolOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace ::buddy::acc;

namespace pebble {
LogicalResult lowerScaledAdd(::buddy::frontend::ScaledAddI8Op op);
LogicalResult lowerGatherRows(::buddy::frontend::GatherRowsOp op);
LogicalResult lowerConcatColumns(::buddy::frontend::ConcatColumnsOp op);
} // namespace pebble

LogicalResult pebble::lowerNonlinearOps(func::FuncOp function) {
  SmallVector<Operation *> operations;
  function.walk([&](Operation *op) {
    if (isa<::buddy::frontend::LookupI32Op, ::buddy::frontend::SoftmaxI8Op,
            ::buddy::frontend::RowMaxU32Op, ::buddy::frontend::GatherRowsOp,
            ::buddy::frontend::ConcatColumnsOp,
            ::buddy::frontend::ScaledAddI8Op>(op))
      operations.push_back(op);
  });
  for (Operation *op : operations) {
    if (auto add = dyn_cast<::buddy::frontend::ScaledAddI8Op>(op)) {
      if (failed(pebble::lowerScaledAdd(add)))
        return failure();
      continue;
    }
    if (auto concat = dyn_cast<::buddy::frontend::ConcatColumnsOp>(op)) {
      if (failed(pebble::lowerConcatColumns(concat)))
        return failure();
      continue;
    }
    if (auto gather = dyn_cast<::buddy::frontend::GatherRowsOp>(op)) {
      if (failed(pebble::lowerGatherRows(gather)))
        return failure();
      continue;
    }
    auto softmax = dyn_cast<::buddy::frontend::SoftmaxI8Op>(op);
    auto type = cast<RankedTensorType>(op->getOperand(0).getType());
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    int64_t capacity = target::getTarget().bankDepth * 4;
    if (isa<::buddy::frontend::RowMaxU32Op>(op)) {
      if (!type.hasStaticShape() || type.getRank() != 2 ||
          type.getDimSize(1) <= 0 || type.getDimSize(1) % 4 ||
          type.getDimSize(1) > capacity ||
          resultType.getShape() != ArrayRef<int64_t>{type.getDimSize(0)})
        return op->emitOpError("requires matrix rows divisible by four fitting "
                               "one bank and one result per row");
      OpBuilder b(op);
      Location loc = op->getLoc();
      auto index = [&](int64_t x) -> Value {
        return arith::ConstantIndexOp::create(b, loc, x);
      };
      Value zero = index(0), one = index(1);
      Value input = pebble::tensorBuffer(b, loc, op->getOperand(0));
      Value output = memref::AllocOp::create(
          b, loc, MemRefType::get(resultType.getShape(), b.getI32Type()));
      Value sourceBank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value resultBank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      auto loop = scf::ForOp::create(b, loc, zero, index(type.getDimSize(0)),
                                     one, ValueRange{sourceBank, resultBank});
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        Value row = loop.getInductionVar();
        Value memory = memref::SubViewOp::create(
            b, loc, input, ArrayRef<OpFoldResult>{row, b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                   b.getIndexAttr(type.getDimSize(1))},
            SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
        Value source = loop.getRegionIterArgs()[0];
        source = BankMvinOp::create(b, loc, b.getI64Type(), memory, source,
                                    type.getDimSize(1) / 4, 1);
        Value result = loop.getRegionIterArgs()[1];
        result = ::buddy::maxpool_ball::BankTopKU32Op::create(
            b, loc, b.getI64Type(), source, result, type.getDimSize(1) / 4,
            type.getDimSize(1), 1);
        memory = memref::SubViewOp::create(
            b, loc, output, ArrayRef<OpFoldResult>{row},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        result = BankMvoutStridedOp::create(b, loc, b.getI64Type(), memory,
                                            result, zero, one, index(4), 0, 1,
                                            index(4), zero);
        scf::YieldOp::create(b, loc, ValueRange{source, result});
      }
      BankReleaseOp::create(b, loc, loop.getResult(0));
      BankReleaseOp::create(b, loc, loop.getResult(1));
      Value result = bufferization::ToTensorOp::create(b, loc, resultType,
                                                       output, true, true);
      op->getResult(0).replaceAllUsesWith(result);
      op->erase();
      continue;
    }
    if (!type.hasStaticShape() || type.getRank() != 2 ||
        type.getShape() != resultType.getShape() || type.getDimSize(1) <= 0 ||
        type.getDimSize(1) % 4 || (softmax && type.getDimSize(1) > capacity) ||
        cast<RankedTensorType>(op->getOperand(1).getType()).getShape() !=
            ArrayRef<int64_t>{1024})
      return op->emitOpError(
          "requires matrices with columns divisible by four, a 1024-entry "
          "table, and softmax rows fitting one bank");
    OpBuilder b(op);
    Location loc = op->getLoc();
    auto index = [&](int64_t x) -> Value {
      return arith::ConstantIndexOp::create(b, loc, x);
    };
    Value zero = index(0), one = index(1);
    Value input = pebble::tensorBuffer(b, loc, op->getOperand(0));
    Value table = pebble::tensorBuffer(b, loc, op->getOperand(1));
    Value output = memref::AllocOp::create(
        b, loc, MemRefType::get(type.getShape(), resultType.getElementType()));
    Value tableBank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    tableBank =
        BankMvinOp::create(b, loc, b.getI64Type(), table, tableBank, 256, 1);
    auto loop =
        scf::ForOp::create(b, loc, zero, index(type.getDimSize(0)), one);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value row = loop.getInductionVar();
      for (int64_t first = 0; first < type.getDimSize(1); first += capacity) {
        int64_t count = std::min(capacity, type.getDimSize(1) - first);
        Value memory = memref::SubViewOp::create(
            b, loc, input, ArrayRef<OpFoldResult>{row, b.getIndexAttr(first)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(count)},
            SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
        Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        a = BankMvinOp::create(b, loc, b.getI64Type(), memory, a, count / 4, 1);
        Value result = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        if (softmax)
          result = ::buddy::lut_ball::BankSoftmaxI8Op::create(
              b, loc, b.getI64Type(), a, tableBank, result, count / 4,
              softmax.getMultiplier());
        else
          result = ::buddy::lut_ball::BankLookupI32Op::create(
              b, loc, b.getI64Type(), a, tableBank, result, count / 4);
        BankReleaseOp::create(b, loc, a);
        int64_t rows = softmax ? count / 16 : count / 4;
        if (rows) {
          memory = memref::SubViewOp::create(
              b, loc, output,
              ArrayRef<OpFoldResult>{row, b.getIndexAttr(first)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(count)},
              SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
          result = BankMvoutOp::create(b, loc, b.getI64Type(), memory, result,
                                       rows, 1);
        }
        if (softmax && count % 16) {
          memory = memref::SubViewOp::create(
              b, loc, output,
              ArrayRef<OpFoldResult>{row, b.getIndexAttr(first + rows * 16)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                     b.getIndexAttr(count % 16)},
              SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
          result = BankMvoutStridedOp::create(
              b, loc, b.getI64Type(), memory, result, index(rows), one,
              index(16), 0, 1, index(count % 16), zero);
        }
        BankReleaseOp::create(b, loc, result);
      }
    }
    BankReleaseOp::create(b, loc, tableBank);
    Value result = bufferization::ToTensorOp::create(b, loc, resultType, output,
                                                     true, true);
    op->getResult(0).replaceAllUsesWith(result);
    op->erase();
  }
  return success();
}
