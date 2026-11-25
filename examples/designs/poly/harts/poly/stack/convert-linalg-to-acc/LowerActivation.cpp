#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mxint8actOps.h"
#include <numeric>

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

void preparePolyActivations(ModuleOp module) {
  SmallVector<Operation *> values;
  module.walk([&](MXINT8ActivationOp op) {
    auto function = op->getParentOfType<func::FuncOp>();
    auto profile = function->getAttrOfType<StringAttr>("bbacc.target");
    if (target::getTarget().hart != "reduce" &&
        (!profile || profile.getValue() != target::getTarget().hart))
      return;
    if (target::getTarget().hart != "reduce")
      for (Value value : {op.getInput(), op.getGate()})
        if (auto quant = value.getDefiningOp<FP32ToMXINT8Op>();
            quant && !llvm::is_contained(values, quant.getOperation()))
          values.push_back(quant);
    values.push_back(op);
  });
  for (Operation *op : values) {
    Value value = op->getResult(0);
    if (value.hasOneUse() && isa<bufferization::MaterializeInDestinationOp>(
                                 *value.getUsers().begin()))
      continue;
    OpBuilder b(op);
    auto type = cast<RankedTensorType>(value.getType());
    SmallVector<Value> dims;
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      if (type.isDynamicDim(axis))
        dims.push_back(
            tensor::DimOp::create(b, op->getLoc(), op->getOperand(0), axis));
    Value storage = tensor::EmptyOp::create(b, op->getLoc(), type.getShape(),
                                            type.getElementType(), dims);
    b.setInsertionPointAfter(op);
    auto written = bufferization::MaterializeInDestinationOp::create(
        b, op->getLoc(), value, storage);
    value.replaceAllUsesExcept(written.getResult(), written);
  }
}

LogicalResult lowerPolyActivations(ModuleOp module) {
  SmallVector<MXINT8ActivationOp> operations;
  module.walk([&](MXINT8ActivationOp op) {
    auto profile =
        op->getParentOfType<func::FuncOp>()->getAttrOfType<StringAttr>(
            "bbacc.target");
    if (target::getTarget().hart == "reduce" ||
        (profile && profile.getValue() == target::getTarget().hart))
      operations.push_back(op);
  });
  for (auto op : operations) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto type = op.getInput().getType();
    int64_t rank = type.getRank(), packed = type.getShape().back(),
            k = packed * 32 / 33;
    if (op.getAxis() != rank - 1 || k <= 0 || k % 32 || packed != k + k / 32 ||
        op.getGate().getType() != type || op.getResult().getType() != type ||
        op.getTable().getType().getShape() != ArrayRef<int64_t>{4096})
      return op.emitError("poly integer activation requires complete last-axis "
                          "MXINT8 blocks and a 16 KiB table");
    int64_t chunk = k <= 512 ? k : std::gcd(k, int64_t(512));
    int64_t bankRows = (chunk + chunk / 32 + 15) / 16;
    auto written = cast<bufferization::MaterializeInDestinationOp>(
        *op.getResult().getUsers().begin());
    auto c = [&](int64_t n) -> Value {
      return arith::ConstantIndexOp::create(b, loc, n);
    };
    Value outputMemory;
    auto flatten = [&](Value value, bool readOnly) -> Value {
      auto t = cast<RankedTensorType>(value.getType());
      Value memory = bufferization::ToBufferOp::create(
          b, loc, MemRefType::get(t.getShape(), t.getElementType()), value,
          readOnly);
      if (!readOnly)
        outputMemory = memory;
      if (t.getRank() == 1)
        return memory;
      ReassociationIndices axes(t.getRank());
      std::iota(axes.begin(), axes.end(), 0);
      return memref::CollapseShapeOp::create(
          b, loc, memory, ArrayRef<ReassociationIndices>{axes});
    };
    Value input = flatten(op.getInput(), true);
    Value gate = op.getGated() ? flatten(op.getGate(), true) : input;
    Value tableMemory = flatten(op.getTable(), true);
    Value output = flatten(written.getDest(), false);
    Value rows = c(1);
    for (int64_t axis = 0; axis < rank - 1; ++axis)
      rows = arith::MulIOp::create(
          b, loc, rows, tensor::DimOp::create(b, loc, op.getInput(), axis));
    Value table = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    table =
        BankMvinOp::create(b, loc, b.getI64Type(), tableMemory, table, 1024, 1);
    auto loop = scf::ForOp::create(b, loc, c(0), rows, c(1));
    loop->setAttr("hart.stream", b.getUnitAttr());
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      auto blocks = scf::ForOp::create(b, loc, c(0), c(k / chunk), c(1));
      blocks->setAttr("hart.stream", b.getUnitAttr());
      b.setInsertionPointToStart(blocks.getBody());
      Value row =
          arith::MulIOp::create(b, loc, loop.getInductionVar(), c(packed));
      Value block = blocks.getInductionVar();
      Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      a = BankClearOp::create(b, loc, b.getI64Type(), a, 1, 1);
      Value out = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      auto window = [&](Value memory, Value offset, int64_t bytes) -> Value {
        return memref::SubViewOp::create(
            b, loc, memory, ArrayRef<OpFoldResult>{offset},
            ArrayRef<OpFoldResult>{b.getIndexAttr(bytes)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      };
      for (int operand = 0; operand < (op.getGated() ? 2 : 1); ++operand)
        for (int part = 0; part < 2; ++part) {
          int64_t bytes = part ? chunk / 32 : chunk;
          Value offset = arith::AddIOp::create(
              b, loc, row,
              arith::AddIOp::create(
                  b, loc, c(part ? k : 0),
                  arith::MulIOp::create(b, loc, block, c(bytes))));
          a = BankMvinStridedOp::create(
              b, loc, b.getI64Type(),
              window(operand ? gate : input, offset, bytes), a,
              c(operand * bankRows + (part ? chunk / 16 : 0)),
              c(part ? 1 : chunk / 16), c(16), 0, 1, c(part ? bytes : 16),
              c(0));
        }
      out = ::buddy::mxint8act_ball::BankActMxint8Op::create(
          b, loc, b.getI64Type(), a, table, out, 1, chunk, 0, 0, 0,
          op.getGated());
      for (int part = 0; part < 2; ++part) {
        int64_t bytes = part ? chunk / 32 : chunk;
        Value offset = arith::AddIOp::create(
            b, loc, row,
            arith::AddIOp::create(
                b, loc, c(part ? k : 0),
                arith::MulIOp::create(b, loc, block, c(bytes))));
        out = BankMvoutStridedOp::create(
            b, loc, b.getI64Type(), window(output, offset, bytes), out,
            c(part ? chunk / 16 : 0), c(part ? 1 : chunk / 16), c(16), 0, 1,
            c(part ? bytes : 16), c(0));
      }
      BankReleaseOp::create(b, loc, a);
      BankReleaseOp::create(b, loc, out);
    }
    BankReleaseOp::create(b, loc, table);
    auto outputType = cast<RankedTensorType>(written.getDest().getType());
    Value result = bufferization::ToTensorOp::create(b, loc, outputType,
                                                     outputMemory, true, true);
    written.getResult().replaceAllUsesWith(result);
    written.erase();
    op.erase();
  }
  return success();
}
