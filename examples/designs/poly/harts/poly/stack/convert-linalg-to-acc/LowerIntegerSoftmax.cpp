#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <numeric>

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

void preparePolyIntegerSoftmax(ModuleOp module) {
  SmallVector<IntegerSoftmaxOp> operations;
  module.walk([&](IntegerSoftmaxOp op) {
    auto profile =
        op->getParentOfType<func::FuncOp>()->getAttrOfType<StringAttr>(
            "bbacc.target");
    if (profile && profile.getValue() == target::getTarget().hart)
      operations.push_back(op);
  });
  for (auto op : operations) {
    if (op.getResult().hasOneUse() &&
        isa<bufferization::MaterializeInDestinationOp>(
            *op.getResult().getUsers().begin()))
      continue;
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto type = op.getResult().getType();
    SmallVector<Value> dimensions;
    for (int64_t i = 0; i < type.getRank(); ++i)
      if (type.isDynamicDim(i))
        dimensions.push_back(tensor::DimOp::create(b, loc, op.getInput(), i));
    Value empty = tensor::EmptyOp::create(b, loc, type.getShape(),
                                          type.getElementType(), dimensions);
    b.setInsertionPointAfter(op);
    auto destination = bufferization::MaterializeInDestinationOp::create(
        b, loc, op.getResult(), empty);
    op.getResult().replaceAllUsesExcept(destination.getResult(), destination);
  }
}

LogicalResult lowerPolyIntegerSoftmax(ModuleOp module) {
  SmallVector<IntegerSoftmaxOp> operations;
  module.walk([&](IntegerSoftmaxOp op) {
    auto profile =
        op->getParentOfType<func::FuncOp>()->getAttrOfType<StringAttr>(
            "bbacc.target");
    if (profile && profile.getValue() == target::getTarget().hart)
      operations.push_back(op);
  });
  if (!operations.empty() &&
      !module.getContext()->getLoadedDialect("intsoftmax_ball"))
    return module.emitError(
        "integer softmax requires the registered intsoftmax Ball dialect");
  for (auto op : operations) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto c = [&](int64_t n) -> Value {
      return arith::ConstantIndexOp::create(b, loc, n);
    };
    auto written = dyn_cast<bufferization::MaterializeInDestinationOp>(
        *op.getResult().getUsers().begin());
    if (!written || !op.getResult().hasOneUse())
      return op.emitError("integer softmax requires its hart destination");
    Value outputMemory;
    auto flatten = [&](Value tensor, bool readOnly) -> Value {
      auto type = cast<RankedTensorType>(tensor.getType());
      SmallVector<int64_t> strides(type.getRank(), ShapedType::kDynamic);
      strides.back() = 1;
      auto layout =
          StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, strides);
      Value memory = bufferization::ToBufferOp::create(
          b, loc,
          MemRefType::get(type.getShape(), type.getElementType(), layout),
          tensor, readOnly);
      if (!readOnly)
        outputMemory = memory;
      if (type.getRank() == 1)
        return memory;
      ReassociationIndices axes(type.getRank());
      std::iota(axes.begin(), axes.end(), 0);
      return memref::CollapseShapeOp::create(
          b, loc, memory, ArrayRef<ReassociationIndices>{axes});
    };
    Value input = flatten(op.getInput(), true);
    Value tableMemory = flatten(op.getTable(), true);
    Value output = flatten(written.getDest(), false);
    auto type = op.getInput().getType();
    Value width =
        tensor::DimOp::create(b, loc, op.getInput(), type.getRank() - 1);
    if (type.isDynamicDim(type.getRank() - 1))
      cf::AssertOp::create(
          b, loc,
          arith::AndIOp::create(
              b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, width,
                                    c(0)),
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, width,
                                    c(128))),
          "integer softmax width must be between one and 128");
    Value rows = c(1);
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis)
      rows = arith::MulIOp::create(
          b, loc, rows, tensor::DimOp::create(b, loc, op.getInput(), axis));
    Value bankRows = arith::CeilDivUIOp::create(b, loc, width, c(4));
    Value chunk = arith::DivUIOp::create(
        b, loc, c(target::getTarget().bankDepth), bankRows);
    Value table = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    table =
        BankMvinOp::create(b, loc, b.getI64Type(), tableMemory, table, 1024, 1);
    Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    Value out = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    auto emitChunk = [&](Value first, Value count) {
      auto window = [&](Value memory, Value offset, Value elements) -> Value {
        return memref::SubViewOp::create(
            b, loc, memory, ArrayRef<OpFoldResult>{offset},
            ArrayRef<OpFoldResult>{elements},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      };
      auto transfer = [&](Value memory, Value bank, bool load) {
        Value tail = arith::RemUIOp::create(b, loc, width, c(4));
        auto aligned = scf::IfOp::create(
            b, loc,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, tail, c(0)),
            true);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&aligned.getThenRegion().front());
          Value start = arith::MulIOp::create(b, loc, first, width);
          Value size = arith::MulIOp::create(b, loc, count, width);
          Value tile = window(memory, start, size);
          Value depth = arith::MulIOp::create(b, loc, count, bankRows);
          if (load)
            BankMvinStridedOp::create(b, loc, b.getI64Type(), tile, bank, c(0),
                                      depth, c(16), 0, 1, c(16), c(0));
          else
            BankMvoutStridedOp::create(b, loc, b.getI64Type(), tile, bank, c(0),
                                       depth, c(16), 0, 1, c(16), c(0));
        }
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&aligned.getElseRegion().front());
          auto row = scf::ForOp::create(b, loc, c(0), count, c(1));
          row->setAttr("hart.stream", b.getUnitAttr());
          b.setInsertionPointToStart(row.getBody());
          Value offset = arith::MulIOp::create(
              b, loc,
              arith::AddIOp::create(b, loc, first, row.getInductionVar()),
              width);
          Value bankOffset =
              arith::MulIOp::create(b, loc, row.getInductionVar(), bankRows);
          Value full = arith::DivUIOp::create(b, loc, width, c(4));
          auto bulk = scf::IfOp::create(
              b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, full,
                                    c(0)),
              false);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(&bulk.getThenRegion().front());
            Value tile = window(memory, offset,
                                arith::MulIOp::create(b, loc, full, c(4)));
            if (load)
              BankMvinStridedOp::create(b, loc, b.getI64Type(), tile, bank,
                                        bankOffset, full, c(16), 0, 1, c(16),
                                        c(0));
            else
              BankMvoutStridedOp::create(b, loc, b.getI64Type(), tile, bank,
                                         bankOffset, full, c(16), 0, 1, c(16),
                                         c(0));
          }
          Value tailOffset = arith::AddIOp::create(
              b, loc, offset, arith::MulIOp::create(b, loc, full, c(4)));
          Value tile = window(memory, tailOffset, tail);
          Value last = arith::AddIOp::create(b, loc, bankOffset, full);
          Value bytes = arith::MulIOp::create(b, loc, tail, c(4));
          if (load)
            BankMvinStridedOp::create(b, loc, b.getI64Type(), tile, bank, last,
                                      c(1), c(16), 0, 1, bytes, c(0));
          else
            BankMvoutStridedOp::create(b, loc, b.getI64Type(), tile, bank, last,
                                       c(1), c(16), 0, 1, bytes, c(0));
        }
      };
      transfer(input, a, true);
      Value m = arith::IndexCastOp::create(b, loc, b.getI64Type(), count);
      Value k = arith::IndexCastOp::create(b, loc, b.getI64Type(), width);
      OperationState operation(loc, "intsoftmax_ball.bank_softmax");
      operation.addOperands({a, table, out, m, k});
      operation.addTypes(b.getI64Type());
      Value result = b.create(operation)->getResult(0);
      transfer(output, result, false);
    };
    Value full = arith::DivUIOp::create(b, loc, rows, chunk);
    auto loop = scf::ForOp::create(b, loc, c(0), full, c(1));
    loop->setAttr("hart.stream", b.getUnitAttr());
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      emitChunk(arith::MulIOp::create(b, loc, loop.getInductionVar(), chunk),
                chunk);
    }
    Value remaining = arith::RemUIOp::create(b, loc, rows, chunk);
    auto tail = scf::IfOp::create(
        b, loc,
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, remaining,
                              c(0)),
        false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&tail.getThenRegion().front());
      emitChunk(arith::MulIOp::create(b, loc, full, chunk), remaining);
    }
    BankReleaseOp::create(b, loc, a);
    BankReleaseOp::create(b, loc, out);
    BankReleaseOp::create(b, loc, table);
    Value result = bufferization::ToTensorOp::create(
        b, loc, written.getDest().getType(), outputMemory, true, true);
    written.getResult().replaceAllUsesWith(result);
    written.erase();
    op.erase();
  }
  return success();
}
