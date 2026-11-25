//===- FormCustomOps.cpp - Recognize and fuse hart computations -----------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using ::buddy::acc::CustomOp;

namespace {
Operation *frontend(Value value, StringRef name) {
  Operation *op = value.getDefiningOp();
  return op && op->getName().getStringRef() == name ? op : nullptr;
}

} // namespace

bool pebble::elementwise(linalg::GenericOp op, StringRef name) {
  if (!op || !op.hasPureTensorSemantics() ||
      !linalg::isaElemwiseSingleBinaryOpInterface(op) ||
      op.getBody()->front().getName().getStringRef() != name)
    return false;
  auto result = cast<RankedTensorType>(op.getResult(0).getType());
  auto maps = op.getIndexingMapsArray();
  if (!result.hasStaticShape() || !maps.back().isIdentity())
    return false;
  for (auto [index, input] : llvm::enumerate(op.getInputs())) {
    auto type = cast<RankedTensorType>(input.getType());
    if (!type.hasStaticShape() || type.getRank() != result.getRank())
      return false;
    SmallVector<AffineExpr> dimensions;
    for (int64_t dim = 0; dim < type.getRank(); ++dim)
      dimensions.push_back(type.getDimSize(dim) == 1 &&
                                   result.getDimSize(dim) != 1
                               ? getAffineConstantExpr(0, op.getContext())
                               : getAffineDimExpr(dim, op.getContext()));
    if (maps[index] !=
        AffineMap::get(type.getRank(), 0, dimensions, op.getContext()))
      return false;
  }
  return true;
}

// Recognizes the computations of the hart and fuses them into custom
// operations, which convert-linalg-to-acc then lowers.
void pebble::formCustomOps(func::FuncOp function) {
  // Static recipe coefficients must dominate the computations that fuse them.
  for (auto constant :
       llvm::make_early_inc_range(function.getOps<arith::ConstantOp>()))
    constant->moveBefore(&function.getBody().front(),
                         function.getBody().front().begin());
  // MLIR checks both indexing maps and scalar computations, including casts.
  SmallVector<linalg::GenericOp> generics;
  function.walk([&](linalg::GenericOp op) { generics.push_back(op); });
  IRRewriter rewriter(function.getContext());
  for (auto op : generics) {
    if (op.hasPureTensorSemantics() && (linalg::isaContractionOpInterface(op) ||
                                        linalg::isaTransposeOpInterface(op))) {
      rewriter.setInsertionPoint(op);
      (void)linalg::specializeGenericOp(rewriter, op);
    }
  }

  SmallVector<::buddy::frontend::LookupI8Op> lookups;
  function.walk(
      [&](::buddy::frontend::LookupI8Op op) { lookups.push_back(op); });
  for (auto op : lookups) {
    rewriter.setInsertionPoint(op);
    auto lookup = CustomOp::create(rewriter, op.getLoc(), op->getResultTypes(),
                                   "lookup_i8", op->getOperands());
    rewriter.replaceOp(op, lookup.getOutputs());
  }

  SmallVector<linalg::BatchMatmulOp> batches;
  function.walk([&](linalg::BatchMatmulOp op) { batches.push_back(op); });
  for (auto op : batches) {
    if (!op.hasPureTensorSemantics() || op.hasUserDefinedMaps() ||
        op.getCast() != linalg::TypeFn::cast_signed)
      continue;
    auto a = cast<RankedTensorType>(op.getInputs()[0].getType());
    auto b = cast<RankedTensorType>(op.getInputs()[1].getType());
    auto result = cast<RankedTensorType>(op.getResult(0).getType());
    Value initial = op.getOutputs()[0];
    if (auto fill = initial.getDefiningOp<linalg::FillOp>())
      initial = fill.getInputs()[0];
    if (!a.hasStaticShape() || !b.hasStaticShape() ||
        !a.getElementType().isInteger(8) || !b.getElementType().isInteger(8) ||
        !result.getElementType().isInteger(32) || b.getDimSize(2) % 16 ||
        !matchPattern(initial, m_Zero()))
      continue;
    rewriter.setInsertionPoint(op);
    auto custom = CustomOp::create(rewriter, op.getLoc(), TypeRange{result},
                                   "batch_matmul_i8", op.getInputs());
    rewriter.replaceOp(op, custom.getOutputs());
  }

  SmallVector<linalg::MatmulOp> matmuls;
  function.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  DominanceInfo dominance(function);
  for (auto op : matmuls) {
    if (!op.hasPureTensorSemantics() || op.hasUserDefinedMaps() ||
        op.getCast() != linalg::TypeFn::cast_signed)
      continue;
    rewriter.setInsertionPoint(op);
    auto a = cast<RankedTensorType>(op.getInputs()[0].getType());
    auto b = cast<RankedTensorType>(op.getInputs()[1].getType());
    auto result = cast<RankedTensorType>(op.getResult(0).getType());
    if (a.getElementType().isInteger(8) && b.getElementType().isInteger(8) &&
        result.getElementType().isInteger(32)) {
      // matmul_i8(A, B, C) = C + A @ B, with signed INT8 operands.
      Value initial = op.getOutputs()[0], value = op.getResult(0);
      Operation *trace = nullptr;
      linalg::GenericOp biasAdd;
      if (matchPattern(initial, m_Zero())) {
        if (value.hasOneUse()) {
          auto end = dyn_cast<::buddy::frontend::TraceEndOp>(
              *value.getUsers().begin());
          if (end && !end.getTensor()) {
            trace = end;
            value = end.getResult();
          }
        }
        if (value.hasOneUse()) {
          auto add = dyn_cast<linalg::GenericOp>(*value.getUsers().begin());
          if (pebble::elementwise(add, "arith.addi") &&
              llvm::is_contained(add.getInputs(), value)) {
            Value other = add.getInputs()[add.getInputs()[0] == value ? 1 : 0];
            if (other.getType() == result && dominance.dominates(other, op)) {
              initial = other;
              biasAdd = add;
            }
          }
        }
      }
      auto custom = CustomOp::create(
          rewriter, biasAdd ? biasAdd.getLoc() : op.getLoc(), TypeRange{result},
          "matmul_i8",
          ValueRange{op.getInputs()[0], op.getInputs()[1], initial});
      rewriter.replaceOp(op, custom.getOutputs());
      if (biasAdd)
        rewriter.replaceOp(biasAdd,
                           trace ? trace->getResult(0) : custom.getResult(0));
      continue;
    }

    Operation *lhs = frontend(op.getInputs()[0], "frontend.int8_to_fp32");
    Operation *rhs = frontend(op.getInputs()[1], "frontend.int8_to_fp32");
    if (!lhs || !rhs || !matchPattern(op.getOutputs()[0], m_AnyZeroFloat()))
      continue;
    Operation *quantize = frontend(lhs->getOperand(0), "frontend.fp32_to_int8");
    if (!quantize)
      continue;
    // Match bias dequantized with exactly input_step * weight_step.
    Value biasedResult = op.getResult(0);
    Operation *trace = nullptr;
    if (biasedResult.hasOneUse()) {
      Operation *user = *biasedResult.getUsers().begin();
      auto end = dyn_cast<::buddy::frontend::TraceEndOp>(user);
      // Cycle markers follow the logical Linear. Tensor traces keep their
      // observed boundary and therefore prevent folding bias across it.
      if (end && !end.getTensor()) {
        trace = user;
        biasedResult = user->getResult(0);
      }
    }
    Value bias;
    linalg::GenericOp biasAdd, biasScale;
    Operation *biasDequant = nullptr;
    if (biasedResult.hasOneUse()) {
      auto add = dyn_cast<linalg::GenericOp>(*biasedResult.getUsers().begin());
      if (pebble::elementwise(add, "arith.addf") &&
          llvm::is_contained(add.getInputs(), biasedResult)) {
        Value other =
            add.getInputs()[add.getInputs()[0] == biasedResult ? 1 : 0];
        auto dequant = frontend(other, "frontend.int32_to_fp32");
        if (dequant) {
          auto scale =
              dequant->getOperand(1).getDefiningOp<linalg::GenericOp>();
          auto biasType =
              cast<RankedTensorType>(dequant->getOperand(0).getType());
          if (pebble::elementwise(scale, "arith.mulf") &&
              biasType.getElementType().isInteger(32) &&
              biasType.getShape() == result.getShape() &&
              ((scale.getInputs()[0] == lhs->getOperand(1) &&
                scale.getInputs()[1] == rhs->getOperand(1)) ||
               (scale.getInputs()[1] == lhs->getOperand(1) &&
                scale.getInputs()[0] == rhs->getOperand(1))) &&
              dominance.properlyDominates(dequant->getOperand(0), op)) {
            bias = dequant->getOperand(0);
            biasAdd = add;
            biasScale = scale;
            biasDequant = dequant;
          }
        }
      }
    }
    if (!bias) {
      auto biasType =
          RankedTensorType::get(result.getShape(), rewriter.getI32Type());
      bias = arith::ConstantOp::create(rewriter, op.getLoc(), biasType,
                                       rewriter.getZeroAttr(biasType));
    }
    // linear_i8(x, W, bias, multiplier, input_step, weight_step) accumulates
    // signed INT8 products on INT32 bias, then restores FP32 values.
    auto custom = CustomOp::create(
        rewriter,
        pebble::fusedLoc(biasAdd ? biasAdd.getLoc() : op.getLoc(),
                         quantize->getLoc()),
        TypeRange{result}, "linear_i8",
        ValueRange{quantize->getOperand(0), rhs->getOperand(0), bias,
                   quantize->getOperand(1), lhs->getOperand(1),
                   rhs->getOperand(1)});
    rewriter.replaceOp(op, custom.getOutputs());
    if (biasAdd) {
      rewriter.replaceOp(biasAdd,
                         trace ? trace->getResult(0) : custom.getResult(0));
      if (biasDequant->use_empty())
        rewriter.eraseOp(biasDequant);
      if (biasScale->use_empty())
        rewriter.eraseOp(biasScale);
    }
    if (rhs != lhs && rhs->use_empty())
      rewriter.eraseOp(rhs);
    if (lhs->use_empty())
      rewriter.eraseOp(lhs);
    if (quantize->use_empty())
      rewriter.eraseOp(quantize);
  }

  SmallVector<linalg::TransposeOp> transposes;
  function.walk([&](linalg::TransposeOp op) { transposes.push_back(op); });
  const auto &target = target::getTarget();
  for (auto op : transposes) {
    auto type = dyn_cast<RankedTensorType>(op.getInput().getType());
    if (!type || type.getRank() < 2 || !type.hasStaticShape() ||
        !type.getElementType().isInteger(8))
      continue;
    int64_t rank = type.getRank();
    SmallVector<int64_t> permutation;
    for (int64_t axis = 0; axis < rank; ++axis)
      permutation.push_back(axis);
    std::swap(permutation[rank - 2], permutation[rank - 1]);
    int64_t rows = type.getDimSize(rank - 2),
            columns = type.getDimSize(rank - 1);
    if (op.getPermutation() != ArrayRef<int64_t>(permutation) ||
        rows > target.bankDepth || columns % 16 ||
        rows * columns > (target.dmaMaxRanges - 1) * 4096)
      continue;
    rewriter.setInsertionPoint(op);
    auto custom = CustomOp::create(rewriter, op.getLoc(), op->getResultTypes(),
                                   "transpose_i8", ValueRange{op.getInput()});
    rewriter.replaceOp(op, custom.getOutputs());
  }
  pebble::formSpatialOps(function);

  // A one-row Linear whose single consumer chain [relu] ends in another
  // one-row Linear requantizes for it. Its INT8 row stays in banks, one
  // group per sixteen columns, when both Linears fit the banks with it.
  SmallVector<CustomOp> linears;
  function.walk([&](CustomOp op) {
    if (op.getKind() == "linear_i8")
      linears.push_back(op);
  });
  auto oneRow = [](CustomOp op) {
    return cast<RankedTensorType>(op.getResult(0).getType()).getDimSize(0) == 1;
  };
  for (CustomOp op : linears) {
    if (!oneRow(op))
      continue;
    Value value = op.getResult(0);
    SmallVector<Operation *> links, traces;
    bool relu = false;
    while (value.hasOneUse()) {
      Operation *user = *value.getUsers().begin();
      if (user->getBlock() != op->getBlock())
        break;
      if (auto end = dyn_cast<::buddy::frontend::TraceEndOp>(user)) {
        if (end.getTensor())
          break;
        traces.push_back(user);
      } else {
        auto maximum = dyn_cast<linalg::GenericOp>(user);
        if (relu || !pebble::elementwise(maximum, "arith.maximumf") ||
            !llvm::is_contained(maximum.getInputs(), value) ||
            !matchPattern(
                maximum.getInputs()[maximum.getInputs()[0] == value ? 1 : 0],
                m_AnyZeroFloat()))
          break;
        relu = true;
      }
      links.push_back(user);
      value = user->getResult(0);
    }
    if (!value.hasOneUse())
      continue;
    auto next = dyn_cast<CustomOp>(*value.getUsers().begin());
    if (!next || next.getKind() != "linear_i8" || !oneRow(next) ||
        next.getInputs()[0] != value ||
        !dominance.properlyDominates(next.getInputs()[3], op))
      continue;
    bool barrier = false;
    for (Operation *between = op->getNextNode(); between != next;
         between = between->getNextNode())
      if (!isMemoryEffectFree(between) &&
          !isa<::buddy::frontend::TraceStartOp, ::buddy::frontend::TraceEndOp>(
              between)) {
        barrier = true;
        break;
      }
    if (barrier)
      continue;
    // Banks of the producer: its A (and FP32 source) and B, or the resident
    // row and B, the row it builds and four more; the consumer reads the
    // row with as many groups of B.
    auto input = op.getInputs()[0];
    auto producer = input.getDefiningOp<CustomOp>();
    bool residentInput = producer && (producer->hasAttr("bbacc.resident") ||
                                      producer.getKind() == "flatten_nchw");
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t k = inputType.getDimSize(1);
    int64_t n = cast<RankedTensorType>(value.getType()).getDimSize(1);
    int64_t rowGroups = (n + 15) / 16;
    int64_t inputBanks;
    if (residentInput) {
      inputBanks = producer.getKind() == "flatten_nchw" ? 2 : 2 * (k / 16);
    } else {
      bool floating = inputType.getElementType().isF32();
      inputBanks =
          (floating ? 3 : 2) * pebble::linearSplit(k, 1, floating).groups;
    }
    // A resident Linear consumes one reduction call. At peak it holds A/B,
    // the preceding output groups, one accumulator and one scale bank;
    // B is released before allocating the converted output group.
    int64_t temporaries =
        residentInput && producer.getKind() == "linear_i8" ? 1 : 4;
    bool resident =
        n % 16 == 0 &&
        inputBanks + rowGroups + temporaries <= target.bankNum - 1 &&
        2 * rowGroups + 4 <= target.bankNum - 1;
    pebble::moveTraceStarts(op, traces);
    rewriter.setInsertionPoint(op);
    SmallVector<Value> inputs(op.getInputs());
    inputs.push_back(next.getInputs()[3]);
    // The INT8 row is the input quantization of `next`.
    auto fused = CustomOp::create(
        rewriter,
        pebble::fusedLoc(pebble::quantizationLoc(next),
                         pebble::quantizationLoc(op)),
        TypeRange{RankedTensorType::get({1, n}, rewriter.getI8Type())},
        "linear_i8", inputs);
    fused->setAttr("relu", rewriter.getBoolAttr(relu));
    if (resident)
      fused->setAttr("bbacc.resident", rewriter.getUnitAttr());
    rewriter.setInsertionPointAfter(fused);
    for (Operation *trace : traces) {
      Operation *moved = rewriter.clone(*trace);
      moved->setOperand(0, fused.getResult(0));
      moved->getResult(0).setType(fused.getResult(0).getType());
    }
    next->setOperand(0, fused.getResult(0));
    for (Operation *link : llvm::reverse(links))
      rewriter.eraseOp(link);
    rewriter.eraseOp(op);
  }
  SmallVector<linalg::PoolingNhwcMaxOp> integerPools;
  function.walk([&](linalg::PoolingNhwcMaxOp op) {
    auto element =
        cast<RankedTensorType>(op.getResult(0).getType()).getElementType();
    if (element.isInteger(32) || element.isInteger(8))
      integerPools.push_back(op);
  });
  for (auto pool : integerPools) {
    Value input = pool.getInputs()[0];
    auto pad = input.getDefiningOp<tensor::PadOp>();
    Value source = pad ? pad.getSource() : input;
    bool bytes =
        cast<RankedTensorType>(source.getType()).getElementType().isInteger(8);
    Value producer = source;
    while (true) {
      if (auto slice = producer.getDefiningOp<tensor::ExtractSliceOp>())
        producer = slice.getSource();
      else if (auto trace =
                   producer.getDefiningOp<::buddy::frontend::TraceEndOp>())
        producer = trace.getInput();
      else
        break;
    }
    auto relu = producer.getDefiningOp<::buddy::frontend::FixedUnaryOp>();
    auto convolution = producer.getDefiningOp<CustomOp>();
    if (!bytes && !(relu && relu.getKind() == "relu") &&
        !(convolution && convolution->hasAttr("fixed_relu")))
      continue;
    auto type = cast<RankedTensorType>(pool.getResult(0).getType());
    auto filter = cast<RankedTensorType>(pool.getInputs()[1].getType());
    auto strides = llvm::to_vector(pool.getStrides().getValues<int64_t>());
    if (!type.hasStaticShape() || type.getDimSize(0) != 1 ||
        type.getDimSize(3) % 16 ||
        filter.getDimSize(0) != filter.getDimSize(1) ||
        strides[0] != strides[1] ||
        !llvm::all_of(pool.getDilations().getValues<int64_t>(),
                      [](int64_t n) { return n == 1; }))
      continue;
    SmallVector<int64_t> padding{0, 0};
    if (pad) {
      IntegerAttr fill;
      auto low = pad.getStaticLow(), high = pad.getStaticHigh();
      if (llvm::is_contained(low, ShapedType::kDynamic) ||
          llvm::is_contained(high, ShapedType::kDynamic) || low[0] || low[3] ||
          high[0] || high[3] ||
          !matchPattern(pad.getConstantPaddingValue(), m_Constant(&fill)) ||
          (bytes ? fill.getInt() != -128 : fill.getInt() > 0))
        continue;
      padding = {low[1], low[2]};
    }
    rewriter.setInsertionPoint(pool);
    auto result = CustomOp::create(rewriter, pool.getLoc(), TypeRange{type},
                                   bytes ? "maxpool_i8" : "maxpool_i32",
                                   ValueRange{source});
    result->setAttr("kernel", rewriter.getI64IntegerAttr(filter.getDimSize(0)));
    result->setAttr("stride", rewriter.getI64IntegerAttr(strides[0]));
    result->setAttr("padding", rewriter.getDenseI64ArrayAttr(padding));
    rewriter.replaceOp(pool, result.getOutputs());
    if (pad && pad->use_empty())
      rewriter.eraseOp(pad);
  }

  // With positive, finite per-channel scales, INT8 quantization commutes with
  // max pooling and ReLU. Keep the overlapping pooling windows on the Balls.
  SmallVector<CustomOp> consumers;
  function.walk([&](CustomOp op) {
    if (op.getKind() == "conv2d_i8" || op.getKind() == "linear_i8")
      consumers.push_back(op);
  });
  for (CustomOp consumer : consumers) {
    Value pooled = consumer.getInputs()[0];
    tensor::CollapseShapeOp flatten;
    linalg::TransposeOp transpose;
    if (consumer.getKind() == "linear_i8") {
      flatten = pooled.getDefiningOp<tensor::CollapseShapeOp>();
      transpose = flatten
                      ? flatten.getSrc().getDefiningOp<linalg::TransposeOp>()
                      : linalg::TransposeOp();
      if (!transpose || !flatten->hasOneUse() || !transpose->hasOneUse() ||
          transpose.getPermutation() != ArrayRef<int64_t>{0, 3, 1, 2} ||
          flatten.getReassociationIndices() !=
              SmallVector<ReassociationIndices>{{0}, {1, 2, 3}})
        continue;
      pooled = transpose.getInput();
    }
    auto pool = pooled.getDefiningOp<linalg::PoolingNhwcMaxOp>();
    DenseFPElementsAttr scales;
    if (!pool || !pool.getResult(0).hasOneUse() ||
        !matchPattern(consumer.getInputs()[3], m_Constant(&scales)) ||
        !llvm::all_of(scales.getValues<APFloat>(), [](const APFloat &v) {
          return v.isFinite() && !v.isNegative() && !v.isZero();
        }))
      continue;
    auto type = cast<RankedTensorType>(pool.getResult(0).getType());
    auto filter = cast<RankedTensorType>(pool.getInputs()[1].getType());
    auto strides = llvm::to_vector(pool.getStrides().getValues<int64_t>());
    auto inputType = cast<RankedTensorType>(pool.getInputs()[0].getType());
    if (!type.hasStaticShape() || !inputType.hasStaticShape() ||
        (flatten &&
         type.getDimSize(1) * type.getDimSize(2) > target.bankDepth) ||
        type.getDimSize(0) != 1 || type.getDimSize(3) % 16 ||
        filter.getDimSize(0) != filter.getDimSize(1) ||
        strides[0] != strides[1] ||
        !llvm::all_of(pool.getDilations().getValues<int64_t>(),
                      [](int64_t n) { return n == 1; }) ||
        (scales.getNumElements() != type.getDimSize(3) &&
         scales.getNumElements() != 1))
      continue;
    Value input = pool.getInputs()[0];
    auto pad = input.getDefiningOp<tensor::PadOp>();
    auto relu =
        (pad ? pad.getSource() : input).getDefiningOp<linalg::GenericOp>();
    bool clamp = pebble::elementwise(relu, "arith.maximumf") &&
                 matchPattern(relu.getInputs()[1], m_AnyZeroFloat());
    SmallVector<int64_t> padding{0, 0};
    if (pad && clamp) {
      FloatAttr fill;
      auto low = pad.getStaticLow(), high = pad.getStaticHigh();
      clamp = !llvm::is_contained(low, ShapedType::kDynamic) &&
              !llvm::is_contained(high, ShapedType::kDynamic) && !low[0] &&
              !low[3] && !high[0] && !high[3] &&
              matchPattern(pad.getConstantPaddingValue(), m_Constant(&fill)) &&
              !fill.getValue().isNaN() &&
              (fill.getValue().isNegative() || fill.getValue().isZero());
      if (clamp)
        padding = {low[1], low[2]};
    }
    if (clamp)
      input = relu.getInputs()[0];
    auto fixedRelu =
        pad ? pad.getSource().getDefiningOp<::buddy::frontend::FixedUnaryOp>()
            : ::buddy::frontend::FixedUnaryOp();
    if (fixedRelu && fixedRelu.getKind() == "relu") {
      IntegerAttr fill;
      auto low = pad.getStaticLow(), high = pad.getStaticHigh();
      if (!llvm::is_contained(low, ShapedType::kDynamic) &&
          !llvm::is_contained(high, ShapedType::kDynamic) && !low[0] &&
          !low[3] && !high[0] && !high[3] &&
          matchPattern(pad.getConstantPaddingValue(), m_Constant(&fill)) &&
          fill.getInt() <= 0) {
        input = pad.getSource();
        padding = {low[1], low[2]};
      }
    }
    rewriter.setInsertionPoint(pool);
    Value factors = consumer.getInputs()[3];
    if (scales.getNumElements() == 1)
      factors = arith::ConstantOp::create(
          rewriter, pool.getLoc(),
          DenseFPElementsAttr::get(
              RankedTensorType::get({1, 1, 1, type.getDimSize(3)},
                                    rewriter.getF32Type()),
              *scales.getValues<APFloat>().begin()));
    auto shape =
        cast<RankedTensorType>(consumer.getInputs()[0].getType()).getShape();
    auto codes = CustomOp::create(
        rewriter, pebble::quantizationLoc(consumer),
        TypeRange{RankedTensorType::get(shape, rewriter.getI8Type())},
        "pool_quant_i8", ValueRange{input, factors});
    codes->setAttr("pool_shape",
                   rewriter.getDenseI64ArrayAttr(type.getShape()));
    codes->setAttr("padding", rewriter.getDenseI64ArrayAttr(padding));
    codes->setAttr("kernel", rewriter.getI64IntegerAttr(filter.getDimSize(0)));
    codes->setAttr("stride", rewriter.getI64IntegerAttr(strides[0]));
    codes->setAttr("relu", rewriter.getBoolAttr(clamp));
    consumer->setOperand(0, codes.getResult(0));
    if (flatten) {
      rewriter.eraseOp(flatten);
      rewriter.eraseOp(transpose);
    }
    rewriter.eraseOp(pool);
    if (pad && pad->use_empty())
      rewriter.eraseOp(pad);
    if (clamp && relu->use_empty())
      rewriter.eraseOp(relu);
  }
}
