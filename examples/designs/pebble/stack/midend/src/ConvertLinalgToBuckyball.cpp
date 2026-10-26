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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"

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

SmallVector<Value> pebble::repeat(
    OpBuilder &builder, Location loc, int64_t begin, int64_t end, int64_t step,
    bool unroll, ValueRange carried,
    function_ref<SmallVector<Value>(OpBuilder &, Value, ValueRange)> body) {
  SmallVector<Value> values(carried);
  if (unroll || end - begin <= step) {
    for (int64_t index = begin; index < end; index += step)
      values = body(
          builder, arith::ConstantIndexOp::create(builder, loc, index), values);
    return values;
  }
  auto loop = scf::ForOp::create(
      builder, loc, arith::ConstantIndexOp::create(builder, loc, begin),
      arith::ConstantIndexOp::create(builder, loc, end),
      arith::ConstantIndexOp::create(builder, loc, step), carried,
      [&](OpBuilder &nested, Location loc, Value index, ValueRange args) {
        scf::YieldOp::create(nested, loc, body(nested, index, args));
      });
  return loop.getResults();
}

Value pebble::accumulate(
    OpBuilder &builder, Location loc, int64_t total, int64_t size, bool unroll,
    Value accumulator,
    function_ref<Value(OpBuilder &, Value, int64_t, bool, bool, Value)> body) {
  int64_t pieces = (total + size - 1) / size;
  int64_t last = (pieces - 1) * size;
  accumulator = body(builder, arith::ConstantIndexOp::create(builder, loc, 0),
                     std::min(size, total), true, pieces == 1, accumulator);
  accumulator = pebble::repeat(
      builder, loc, size, last, size, unroll, accumulator,
      [&](OpBuilder &b, Value start, ValueRange args) -> SmallVector<Value> {
        return {body(b, start, size, false, false, args[0])};
      })[0];
  if (pieces > 1)
    accumulator =
        body(builder, arith::ConstantIndexOp::create(builder, loc, last),
             total - last, false, true, accumulator);
  return accumulator;
}

pebble::SpatialTile pebble::spatialTile(int64_t height, int64_t width,
                                        int64_t pool, int64_t stride,
                                        int64_t kernel) {
  int64_t tile = 4 / pool;
  while (tile > 0 && (height % tile || width % tile ||
                      (tile * pool - 1) * stride + kernel > 8))
    --tile;
  if (!tile)
    return {0, 0};
  int64_t side = std::min<int64_t>(8, std::min(height, width));
  while (side % tile)
    --side;
  return {tile, side};
}

namespace {
LogicalResult lowerMatmul(::buddy::custom::CustomOp op, Value residentInput,
                          Value &result) {
  bool linear = op.getKind() == "linear_i8";
  if (op.getInputs().size() != (linear ? 6 : 3) || op.getNumResults() != 1)
    return op.emitOpError("has the wrong number of inputs or results for ")
           << op.getKind();
  ValueRange inputs = op.getInputs();
  auto a = cast<RankedTensorType>(inputs[0].getType());
  auto b = cast<RankedTensorType>(inputs[1].getType());
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  if (!a.hasStaticShape() || !b.hasStaticShape() || a.getRank() != 2 ||
      b.getRank() != 2 || a.getDimSize(1) != b.getDimSize(0))
    return op.emitOpError("requires static MxK and KxN tensors");
  const auto &target = buckyball_target::getBuckyballTarget();
  int64_t k = a.getDimSize(1), n = b.getDimSize(1);
  // One activation row with aligned K and N stays in one bank.
  if (a.getDimSize(0) != 1 || k % 16 || n % 16 || k > target.bankDepth)
    return pebble::lowerLinearTiles(op, result);
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

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Type i64 = builder.getI64Type();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value weight = pebble::tensorBuffer(builder, loc, inputs[1]);
  Value output = memref::AllocOp::create(
      builder, loc, MemRefType::get({1, n}, resultType.getElementType()));
  Value initial = pebble::tensorBuffer(builder, loc, inputs[2]);
  Value scales, inputStep;
  Value codes = residentInput;
  if (linear) {
    inputStep = tensor::ExtractOp::create(builder, loc, inputs[4],
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

  // Keep the activation in its bank while processing each sixteen-column
  // slice.
  pebble::repeat(
      builder, loc, 0, n, 16, false, {},
      [&](OpBuilder &b, Value column, ValueRange) -> SmallVector<Value> {
        if (linear) {
          auto lanes = scf::ForOp::create(
              b, loc, zero, arith::ConstantIndexOp::create(b, loc, 16),
              arith::ConstantIndexOp::create(b, loc, 1));
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(lanes.getBody());
          Value lane = lanes.getInductionVar();
          Value channel = arith::AddIOp::create(b, loc, column, lane);
          Value step = tensor::ExtractOp::create(b, loc, inputs[5],
                                                 ValueRange{zero, channel});
          Value product = arith::MulFOp::create(b, loc, inputStep, step);
          memref::StoreOp::create(b, loc, product, scales, lane);
        }
        SmallVector<OpFoldResult> offsets{b.getIndexAttr(0), column};
        SmallVector<OpFoldResult> unit{b.getIndexAttr(1), b.getIndexAttr(1)};
        Value weightSlice = memref::SubViewOp::create(
            b, loc, weight, offsets,
            ArrayRef<OpFoldResult>{b.getIndexAttr(k), b.getIndexAttr(16)},
            unit);
        Value weightBank = BankAllocOp::create(b, loc, i64, 1, 1);
        weightBank =
            BankMvinOp::create(b, loc, i64, weightSlice, weightBank, k, n / 16);
        Value accumulator = BankAllocOp::create(b, loc, i64, 1, 1);
        Value initialSlice = memref::SubViewOp::create(
            b, loc, initial, offsets,
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(16)},
            unit);
        accumulator =
            BankMvinOp::create(b, loc, i64, initialSlice, accumulator, 4, 1);
        auto matmul =
            BankCustomOp::create(b, loc, i64, ValueRange{codes, weightBank},
                                 accumulator, ValueRange{}, "matmul_i8", true);
        matmul->setAttr("rows", b.getI64IntegerAttr(1));
        matmul->setAttr("k", b.getI64IntegerAttr(k));
        accumulator = matmul.getDestinationOut();
        BankReleaseOp::create(b, loc, weightBank);

        Value resultBank = accumulator;
        if (linear) {
          Value scaleBank = BankAllocOp::create(b, loc, i64, 1, 1);
          scaleBank = BankMvinOp::create(b, loc, i64, scales, scaleBank, 4, 1);
          resultBank = BankAllocOp::create(b, loc, i64, 1, 1);
          auto restored = BankCustomOp::create(
              b, loc, i64, ValueRange{accumulator, scaleBank}, resultBank,
              ValueRange{}, "int32_to_fp32");
          restored->setAttr("rows", b.getI64IntegerAttr(1));
          restored->setAttr("relu", b.getBoolAttr(false));
          resultBank = restored.getDestinationOut();
          BankReleaseOp::create(b, loc, scaleBank);
        }
        Value outputSlice = memref::SubViewOp::create(
            b, loc, output, offsets,
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(16)},
            unit);
        resultBank =
            BankMvoutOp::create(b, loc, i64, outputSlice, resultBank, 4, 1);
        if (linear)
          BankReleaseOp::create(b, loc, resultBank);
        BankReleaseOp::create(b, loc, accumulator);
        return SmallVector<Value>{};
      });
  BankReleaseOp::create(builder, loc, codes);
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
    DenseMap<Value, Value> residentVectors;
    SmallVector<Operation *> spatial;
    if (failed(
            pebble::lowerSpatialOps(getOperation(), residentVectors, spatial)))
      return signalPassFailure();
    SmallVector<::buddy::custom::CustomOp> computations;
    getOperation().walk([&](::buddy::custom::CustomOp op) {
      if (!llvm::is_contained(spatial, op.getOperation()))
        computations.push_back(op);
    });
    for (auto op : computations) {
      if (op.getKind() == "global_avg_i8" || op.getKind() == "lookup_i8" ||
          op.getKind() == "gate_mul_i8") {
        Value result;
        LogicalResult status = op.getKind() == "global_avg_i8"
                                   ? pebble::lowerGlobalAverage(op, result)
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
                             result)))
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
