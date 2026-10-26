//===- ConvertLinalgToBuckyball.cpp - Pebble core lowering ----------------===//

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Custom/CustomOps.h"
#include "Lowering.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace ::buddy::buckyball;

// The recipe supplies contiguous tensors; offsets may point into packed
// weights.
Value pebble::tensorBuffer(OpBuilder &builder, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> strides(type.getRank());
  int64_t stride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    strides[dim] = stride;
    stride *= type.getDimSize(dim);
  }
  auto layout = StridedLayoutAttr::get(builder.getContext(),
                                       ShapedType::kDynamic, strides);
  auto memory = MemRefType::get(type.getShape(), type.getElementType(), layout);
  return bufferization::ToBufferOp::create(builder, loc, memory, tensor, true);
}

namespace {
LogicalResult lowerMatmul(::buddy::custom::CustomOp op, Value residentInput,
                          Value nextMultiplier, bool relu, Value &result) {
  bool linear = op.getKind() == "linear_i8";
  if (op.getInputs().size() != (linear ? 6 : 3) || op.getNumResults() != 1)
    return op.emitOpError("has the wrong number of inputs or results for ")
           << op.getKind();
  ValueRange inputs = op.getInputs();
  auto a = cast<RankedTensorType>(inputs[0].getType());
  auto b = cast<RankedTensorType>(inputs[1].getType());
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  if (a.hasStaticShape() && b.hasStaticShape() && a.getRank() == 2 &&
      b.getRank() == 2 &&
      (a.getDimSize(0) != 1 || a.getDimSize(1) % 16 || b.getDimSize(1) % 16)) {
    if (residentInput || nextMultiplier)
      return op.emitOpError("resident Linear requires one row and aligned K/N");
    return pebble::lowerLinearTiles(op, result);
  }
  const auto &target = buckyball_target::getBuckyballTarget();
  if (!a.hasStaticShape() || !b.hasStaticShape() || a.getRank() != 2 ||
      b.getRank() != 2 || a.getDimSize(0) != 1 ||
      a.getDimSize(1) != b.getDimSize(0) || a.getDimSize(1) <= 0 ||
      b.getDimSize(1) <= 0 || a.getDimSize(1) % 16 || b.getDimSize(1) % 16)
    return op.emitOpError("requires 1xK and KxN tensors with positive K and N "
                          "multiples of sixteen");
  int64_t k = a.getDimSize(1), n = b.getDimSize(1);
  if (target.bankWidthBits != 128 || k > target.bankDepth)
    return op.emitOpError("operands do not fit the core bank geometry");
  if (!b.getElementType().isInteger(8) ||
      !(linear ? a.getElementType().isF32()
               : a.getElementType().isInteger(8)) ||
      !(linear ? resultType.getElementType().isF32()
               : resultType.getElementType().isInteger(32)) ||
      resultType.getShape() != ArrayRef<int64_t>({1, n}))
    return op.emitOpError("has invalid operand or result types for ")
           << op.getKind();
  const int64_t scalarDimensions[] = {1, 1};
  ArrayRef<int64_t> scalarShape(scalarDimensions);
  if (linear &&
      (cast<RankedTensorType>(inputs[3].getType()).getShape() != scalarShape ||
       cast<RankedTensorType>(inputs[4].getType()).getShape() != scalarShape ||
       cast<RankedTensorType>(inputs[5].getType()).getShape() !=
           ArrayRef<int64_t>({1, n})))
    return op.emitOpError("requires one activation scale and one weight scale "
                          "per output channel");
  auto initialType = cast<RankedTensorType>(inputs[2].getType());
  if (!initialType.getElementType().isInteger(32) ||
      initialType.getShape() != resultType.getShape())
    return op.emitOpError(
        "requires an INT32 initial accumulator matching the result");

  if (nextMultiplier && (n / 16 > 128 || n / 16 > target.bankDepth))
    return op.emitOpError(
        "resident output exceeds the quantizer row address range");

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Type i64 = builder.getI64Type();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value weight = pebble::tensorBuffer(builder, loc, inputs[1]);
  Value output;
  if (!nextMultiplier)
    output = memref::AllocOp::create(
        builder, loc, MemRefType::get({1, n}, resultType.getElementType()));
  Value residentOutput;
  if (nextMultiplier)
    residentOutput = BankAllocOp::create(builder, loc, i64, 1, 1);
  Value initial = pebble::tensorBuffer(builder, loc, inputs[2]);
  Value scales, inputStep, outputMultiplier;
  Value codes = residentInput;
  if (linear) {
    inputStep = tensor::ExtractOp::create(builder, loc, inputs[4],
                                          ValueRange{zero, zero});
    if (nextMultiplier)
      outputMultiplier = tensor::ExtractOp::create(builder, loc, nextMultiplier,
                                                   ValueRange{zero, zero});
    scales = memref::AllocOp::create(
        builder, loc, MemRefType::get({16}, builder.getF32Type()));
  }
  if (!codes) {
    Value input = pebble::tensorBuffer(builder, loc, inputs[0]);
    codes = BankAllocOp::create(builder, loc, i64, 1, 1);
    if (linear) {
      Value multiplier = tensor::ExtractOp::create(builder, loc, inputs[3],
                                                   ValueRange{zero, zero});
      Value inputBank = BankAllocOp::create(builder, loc, i64, 1, 1);
      inputBank =
          BankMvinOp::create(builder, loc, i64, input, inputBank, k / 4, 1);
      auto quantized =
          BankCustomOp::create(builder, loc, i64, ValueRange{inputBank}, codes,
                               ValueRange{multiplier}, "fp32_to_int8");
      quantized->setAttr("rows", builder.getI64IntegerAttr(k / 4));
      codes = quantized.getDestinationOut();
      BankReleaseOp::create(builder, loc, inputBank);
    } else {
      codes = BankMvinOp::create(builder, loc, i64, input, codes, k / 16, 1);
    }
  }

  // Keep the activation in its bank while processing each sixteen-column slice.
  for (int64_t column = 0; column < n; column += 16) {
    for (int64_t lane = 0; linear && lane < 16; ++lane) {
      Value index = arith::ConstantIndexOp::create(builder, loc, lane);
      Value channel =
          arith::ConstantIndexOp::create(builder, loc, column + lane);
      Value step = tensor::ExtractOp::create(builder, loc, inputs[5],
                                             ValueRange{zero, channel});
      Value product = arith::MulFOp::create(builder, loc, inputStep, step);
      if (nextMultiplier)
        product =
            arith::MulFOp::create(builder, loc, product, outputMultiplier);
      memref::StoreOp::create(builder, loc, product, scales, index);
    }
    Value weightSlice = memref::SubViewOp::create(
        builder, loc, weight, ArrayRef<int64_t>{0, column},
        ArrayRef<int64_t>{k, 16}, ArrayRef<int64_t>{1, 1});
    Value weightBank = BankAllocOp::create(builder, loc, i64, 1, 1);
    weightBank = BankMvinOp::create(builder, loc, i64, weightSlice, weightBank,
                                    k, n / 16);
    Value accumulator = BankAllocOp::create(builder, loc, i64, 1, 1);
    Value initialSlice = memref::SubViewOp::create(
        builder, loc, initial, ArrayRef<int64_t>{0, column},
        ArrayRef<int64_t>{1, 16}, ArrayRef<int64_t>{1, 1});
    accumulator =
        BankMvinOp::create(builder, loc, i64, initialSlice, accumulator, 4, 1);
    auto matmul =
        BankCustomOp::create(builder, loc, i64, ValueRange{codes, weightBank},
                             accumulator, ValueRange{}, "matmul_i8", true);
    matmul->setAttr("rows", builder.getI64IntegerAttr(1));
    matmul->setAttr("k", builder.getI64IntegerAttr(k));
    accumulator = matmul.getDestinationOut();
    BankReleaseOp::create(builder, loc, weightBank);

    Value resultBank = accumulator;
    if (linear) {
      Value scaleBank = BankAllocOp::create(builder, loc, i64, 1, 1);
      scaleBank =
          BankMvinOp::create(builder, loc, i64, scales, scaleBank, 4, 1);
      if (nextMultiplier) {
        auto quantized = BankCustomOp::create(
            builder, loc, i64, ValueRange{accumulator, scaleBank},
            residentOutput, ValueRange{}, "int32_to_int8", true);
        quantized->setAttr("rows", builder.getI64IntegerAttr(1));
        quantized->setAttr("wr_base", builder.getI64IntegerAttr(column / 16));
        quantized->setAttr("relu", builder.getBoolAttr(relu));
        residentOutput = quantized.getDestinationOut();
      } else {
        resultBank = BankAllocOp::create(builder, loc, i64, 1, 1);
        auto restored = BankCustomOp::create(
            builder, loc, i64, ValueRange{accumulator, scaleBank}, resultBank,
            ValueRange{}, "int32_to_fp32");
        restored->setAttr("rows", builder.getI64IntegerAttr(1));
        resultBank = restored.getDestinationOut();
      }
      BankReleaseOp::create(builder, loc, scaleBank);
    }
    if (!nextMultiplier) {
      Value outputSlice = memref::SubViewOp::create(
          builder, loc, output, ArrayRef<int64_t>{0, column},
          ArrayRef<int64_t>{1, 16}, ArrayRef<int64_t>{1, 1});
      resultBank =
          BankMvoutOp::create(builder, loc, i64, outputSlice, resultBank, 4, 1);
      if (linear)
        BankReleaseOp::create(builder, loc, resultBank);
    }
    BankReleaseOp::create(builder, loc, accumulator);
  }
  BankReleaseOp::create(builder, loc, codes);
  result = residentOutput;
  if (!nextMultiplier)
    result = bufferization::ToTensorOp::create(
        builder, loc, op.getResult(0).getType(), output, true, true);
  return success();
}

class ConvertLinalgToBuckyballPass
    : public PassWrapper<ConvertLinalgToBuckyballPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToBuckyballPass)
  StringRef getArgument() const final { return "convert-linalg-to-buckyball"; }
  StringRef getDescription() const final {
    return "Lower custom tensor computations to the core bank dataflow";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BuckyballDialect, arith::ArithDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    if (failed(pebble::lowerQuantizedChains(getOperation())))
      return signalPassFailure();
    DenseMap<Value, Value> residentVectors;
    SmallVector<Operation *> spatial;
    if (failed(
            pebble::lowerSpatialOps(getOperation(), residentVectors, spatial)))
      return signalPassFailure();
    llvm::MapVector<int64_t, SmallVector<Operation *>> chains;
    getOperation().walk([&](Operation *op) {
      if (auto group = op->getAttrOfType<IntegerAttr>("custom.chain"))
        chains[group.getInt()].push_back(op);
    });
    for (auto &[id, links] : chains) {
      SmallVector<::buddy::custom::CustomOp> stages;
      for (Operation *link : links)
        if (auto op = dyn_cast<::buddy::custom::CustomOp>(link))
          if (op.getKind() == "linear_i8")
            stages.push_back(op);
      DenseMap<Value, Value> resident;
      Value input = residentVectors.lookup(stages.front().getInputs()[0]);
      for (auto [index, stage] : llvm::enumerate(stages)) {
        bool last = index + 1 == stages.size();
        Value multiplier = last ? Value() : stages[index + 1].getInputs()[3];
        bool relu =
            !last && stage->getAttrOfType<BoolAttr>("custom.relu").getValue();
        Value lowered;
        if (failed(lowerMatmul(stage, input, multiplier, relu, lowered)))
          return signalPassFailure();
        if (last)
          stage.getResult(0).replaceAllUsesWith(lowered);
        else
          resident[stage.getResult(0)] = input = lowered;
      }
      for (Operation *link : links) {
        if (auto op = dyn_cast<::buddy::custom::CustomOp>(link)) {
          if (op.getKind() == "relu")
            resident[op.getResult(0)] = resident.at(op.getInputs()[0]);
        } else {
          // Cycle-only markers observe the resident bank without spilling it.
          Value bank = resident.at(link->getOperand(0));
          resident[link->getResult(0)] = bank;
          OpBuilder builder(link);
          Operation *trace = builder.clone(*link);
          trace->setOperand(0, bank);
          trace->getResult(0).setType(bank.getType());
          trace->removeAttr("custom.chain");
        }
      }
      for (Operation *link : llvm::reverse(links))
        link->erase();
    }
    SmallVector<::buddy::custom::CustomOp> computations;
    getOperation().walk([&](::buddy::custom::CustomOp op) {
      if (!llvm::is_contained(spatial, op.getOperation()))
        computations.push_back(op);
    });
    for (auto op : computations) {
      if (op.getKind() == "global_avg_i8" || op.getKind() == "lookup_i8" ||
          op.getKind() == "gate_mul_i8") {
        Value result;
        LogicalResult status =
            op.getKind() == "global_avg_i8"
                ? pebble::lowerGlobalAverage(op, result, false)
                : pebble::lowerChannelOp(op, result);
        if (failed(status))
          return signalPassFailure();
        op.getResult(0).replaceAllUsesWith(result);
        op.erase();
        continue;
      }
      if (op.getKind() == "transpose_i8" ||
          op.getKind() == "linear_i8_requant") {
        Value result;
        if (failed(op.getKind() == "transpose_i8"
                       ? pebble::lowerTranspose(op, result)
                       : pebble::lowerLinearTiles(op, result)))
          return signalPassFailure();
        op.getResult(0).replaceAllUsesWith(result);
        op.erase();
        continue;
      }
      if (op.getKind() != "linear_i8" && op.getKind() != "matmul_i8") {
        op.emitOpError("has no bank lowering on this core for ")
            << op.getKind();
        return signalPassFailure();
      }
      Value result;
      if (failed(lowerMatmul(op, residentVectors.lookup(op.getInputs()[0]),
                             Value(), false, result)))
        return signalPassFailure();
      op.getResult(0).replaceAllUsesWith(result);
      op.erase();
    }
    for (Operation *op : llvm::reverse(spatial))
      op->erase();
  }
};

PassRegistration<ConvertLinalgToBuckyballPass> registration;
} // namespace
