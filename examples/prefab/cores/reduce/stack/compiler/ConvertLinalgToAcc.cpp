#include "Acc/AccOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "f32addOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace ::buddy::acc;
namespace {
linalg::GenericOp addition(Value value) {
  auto op = value.getDefiningOp<linalg::GenericOp>();
  if (!op || op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op.getNumResults() != 1 || op.getNumParallelLoops() != op.getNumLoops())
    return {};
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.getElementType().isF32() || !type.getRank() ||
      type.isDynamicDim(type.getRank() - 1) || type.getShape().back() <= 0 ||
      type.getShape().back() % 4)
    return {};
  for (AffineMap map : op.getIndexingMapsArray())
    if (!map.isIdentity())
      return {};
  Block &body = op.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return {};
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!add || !yield || add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1) ||
      yield.getValues()[0] != add.getResult())
    return {};
  return op;
}

class ConvertLinalgToAccPass
    : public PassWrapper<ConvertLinalgToAccPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToAccPass)
  StringRef getArgument() const final { return "convert-linalg-to-acc"; }
  StringRef getDescription() const final {
    return "Run ordered FP32 sum chains on the reduce core accelerator";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::acc::AccDialect,
                    ::buddy::f32add_ball::F32AddBallDialect,
                    arith::ArithDialect, bufferization::BufferizationDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankNum < 3) {
      getOperation().emitError("reduce needs 128-bit banks and three banks");
      return signalPassFailure();
    }
    SmallVector<linalg::GenericOp> roots;
    getOperation().walk([&](linalg::GenericOp op) {
      if (op.getNumResults() != 1 || !addition(op.getResult(0)))
        return;
      if (op.getResult(0).hasOneUse()) {
        auto user =
            dyn_cast<linalg::GenericOp>(*op.getResult(0).getUsers().begin());
        if (user && user.getNumResults() == 1 && addition(user.getResult(0)) &&
            user.getInputs()[0] == op.getResult(0))
          return;
      }
      roots.push_back(op);
    });
    for (auto root : roots) {
      SmallVector<linalg::GenericOp> chain;
      SmallVector<Value> terms;
      auto current = root;
      for (;;) {
        chain.push_back(current);
        terms.push_back(current.getInputs()[1]);
        Value left = current.getInputs()[0];
        auto prior = addition(left);
        if (!prior || !left.hasOneUse()) {
          terms.push_back(left);
          break;
        }
        current = prior;
      }
      std::reverse(terms.begin(), terms.end());
      OpBuilder b(root);
      Location loc = root.getLoc();
      auto type = cast<RankedTensorType>(root.getResult(0).getType());
      auto index = [&](int64_t value) {
        return arith::ConstantIndexOp::create(b, loc, value).getResult();
      };
      SmallVector<Value> dynamic;
      Value elements = index(1);
      for (int64_t axis = 0; axis < type.getRank(); ++axis) {
        Value size =
            type.isDynamicDim(axis)
                ? tensor::DimOp::create(b, loc, terms.front(), axis).getResult()
                : index(type.getDimSize(axis));
        if (type.isDynamicDim(axis))
          dynamic.push_back(size);
        elements = b.createOrFold<arith::MulIOp>(loc, elements, size);
      }
      auto memoryType = MemRefType::get(type.getShape(), b.getF32Type());
      Value output = memref::AllocOp::create(b, loc, memoryType, dynamic);
      output.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(64));
      ReassociationIndices all;
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        all.push_back(axis);
      auto flatten = [&](Value memory) -> Value {
        if (type.getRank() == 1)
          return memory;
        auto flat =
            MemRefType::get({type.hasStaticShape() ? type.getNumElements()
                                                   : ShapedType::kDynamic},
                            b.getF32Type());
        return memref::CollapseShapeOp::create(
            b, loc, flat, memory, ArrayRef<ReassociationIndices>{all});
      };
      SmallVector<Value> inputs;
      for (Value term : terms)
        inputs.push_back(flatten(
            bufferization::ToBufferOp::create(b, loc, memoryType, term, true)));
      Value flatOutput = flatten(output);
      Value remaining = b.createOrFold<arith::DivUIOp>(loc, elements, index(4));
      Value offset = index(0);
      for (int64_t rows = target.bankDepth; rows; rows /= 2) {
        Value trips =
            b.createOrFold<arith::DivUIOp>(loc, remaining, index(rows));
        auto loop = scf::ForOp::create(b, loc, index(0), trips, index(1));
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(loop.getBody());
          Value first = b.createOrFold<arith::AddIOp>(
              loc, offset,
              b.createOrFold<arith::MulIOp>(loc, loop.getInductionVar(),
                                            index(rows)));
          first = b.createOrFold<arith::MulIOp>(loc, first, index(4));
          auto window = [&](Value memory) -> Value {
            return memref::SubViewOp::create(
                b, loc, memory, ArrayRef<OpFoldResult>{first},
                ArrayRef<OpFoldResult>{b.getIndexAttr(rows * 4)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
          };
          Value sum = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
          sum = BankMvinOp::create(b, loc, b.getI64Type(), window(inputs[0]),
                                   sum, rows, 1);
          for (Value input : ArrayRef<Value>(inputs).drop_front()) {
            Value addend = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            addend = BankMvinOp::create(b, loc, b.getI64Type(), window(input),
                                        addend, rows, 1);
            Value destination =
                BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            Value next = ::buddy::f32add_ball::BankAddF32Op::create(
                b, loc, b.getI64Type(), sum, addend, destination, rows);
            BankReleaseOp::create(b, loc, sum);
            BankReleaseOp::create(b, loc, addend);
            sum = next;
          }
          Value stored = BankMvoutOp::create(b, loc, b.getI64Type(),
                                             window(flatOutput), sum, rows, 1);
          BankReleaseOp::create(b, loc, stored);
        }
        offset = b.createOrFold<arith::AddIOp>(
            loc, offset,
            b.createOrFold<arith::MulIOp>(loc, trips, index(rows)));
        remaining = b.createOrFold<arith::RemUIOp>(loc, remaining, index(rows));
      }
      Value result =
          bufferization::ToTensorOp::create(b, loc, type, output, true, true);
      root.getResult(0).replaceAllUsesWith(result);
      for (auto op : chain)
        op.erase();
    }
  }
};
PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
