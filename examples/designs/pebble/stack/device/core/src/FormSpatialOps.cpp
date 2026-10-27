//===- FormSpatialOps.cpp - Recognize quantized convolution chains --------===//
#include "Custom/CustomOps.h"
#include "Lowering.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <limits>

using namespace mlir;
using ::buddy::custom::CustomOp;

namespace {
Operation *frontend(Value value, StringRef name) {
  Operation *op = value.getDefiningOp();
  return op && op->getName().getStringRef() == name ? op : nullptr;
}
} // namespace

void pebble::formSpatialOps(func::FuncOp function) {
  IRRewriter rewriter(function.getContext());
  SmallVector<Operation *> convolutions;
  function.walk([&](Operation *op) {
    if (isa<linalg::Conv2DNhwcFhwcOp, linalg::DepthwiseConv2DNhwcHwcmOp>(op))
      convolutions.push_back(op);
  });
  for (Operation *op : convolutions) {
    if (!cast<linalg::LinalgOp>(op).hasPureTensorSemantics())
      continue;
    bool depthwise = isa<linalg::DepthwiseConv2DNhwcHwcmOp>(op);
    auto transpose = op->getOperand(1).getDefiningOp<linalg::TransposeOp>();
    if (!transpose || transpose.getPermutation() !=
                          (depthwise ? ArrayRef<int64_t>({2, 3, 0, 1})
                                     : ArrayRef<int64_t>({0, 2, 3, 1})))
      continue;
    Value input = op->getOperand(0);
    auto pad = input.getDefiningOp<tensor::PadOp>();
    SmallVector<int64_t> padding{0, 0, 0, 0};
    if (pad) {
      auto low = pad.getStaticLow(), high = pad.getStaticHigh();
      Value padValue = pad.getConstantPaddingValue();
      if (llvm::is_contained(low, ShapedType::kDynamic) ||
          llvm::is_contained(high, ShapedType::kDynamic) || low[0] || low[3] ||
          high[0] || high[3] ||
          (!padValue || !matchPattern(padValue, m_AnyZeroFloat())))
        continue;
      padding = {low[1], high[1], low[2], high[2]};
      input = pad.getSource();
    }
    auto lhs = frontend(input, "frontend.int8_to_fp32");
    auto rhs = frontend(transpose.getInput(), "frontend.int8_to_fp32");
    if (!lhs || !rhs)
      continue;
    auto quant = frontend(lhs->getOperand(0), "frontend.fp32_to_int8");
    // IM2COL takes square kernels up to its size with unit dilation, and the
    // array tiles a single image with one stride.
    auto weightType = cast<RankedTensorType>(rhs->getOperand(0).getType());
    auto inputType = cast<RankedTensorType>(input.getType());
    auto outputType = cast<RankedTensorType>(op->getResult(0).getType());
    auto strides =
        llvm::to_vector(cast<DenseIntElementsAttr>(op->getAttr("strides"))
                            .getValues<int64_t>());
    auto dilations =
        llvm::to_vector(cast<DenseIntElementsAttr>(op->getAttr("dilations"))
                            .getValues<int64_t>());
    if (!weightType.hasStaticShape() || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() || inputType.getDimSize(0) != 1 ||
        weightType.getDimSize(2) != weightType.getDimSize(3) ||
        weightType.getDimSize(2) >
            buckyball_target::getBuckyballBallParam("IM2COL", "maxKSize") ||
        strides[0] != strides[1] || dilations != SmallVector<int64_t>{1, 1} ||
        !pebble::convTiling(outputType.getDimSize(1), outputType.getDimSize(2),
                            1, strides[0], weightType.getDimSize(2),
                            (inputType.getDimSize(3) + 15) / 16, depthwise,
                            true)
             .side)
      continue;
    Operation *end = op;
    Value biasValue;
    Operation *initial = op->getOperand(2).getDefiningOp();
    tensor::CollapseShapeOp collapsed;
    if (depthwise) {
      auto type = cast<RankedTensorType>(op->getResult(0).getType());
      Value zero = op->getOperand(2);
      if (auto fill = zero.getDefiningOp<linalg::FillOp>())
        zero = fill.getInputs()[0];
      if (type.getDimSize(4) != 1 || !matchPattern(zero, m_AnyZeroFloat()) ||
          !op->getResult(0).hasOneUse())
        continue;
      collapsed = dyn_cast<tensor::CollapseShapeOp>(
          *op->getResult(0).getUsers().begin());
      if (!collapsed || !collapsed.getResult().hasOneUse() ||
          collapsed.getReassociationIndices() !=
              SmallVector<ReassociationIndices>{{0}, {1}, {2}, {3, 4}})
        continue;
      auto add = dyn_cast<linalg::GenericOp>(
          *collapsed.getResult().getUsers().begin());
      if (!add || !linalg::isaElemwiseSingleBinaryOpInterface(add) ||
          !isa<arith::AddFOp>(add.getBody()->front()))
        continue;
      int biasIndex = add.getInputs()[0] == collapsed.getResult() ? 1 : 0;
      auto maps = add.getIndexingMapsArray();
      if (maps[biasIndex] !=
              AffineMap::get(4, 0, rewriter.getAffineDimExpr(3)) ||
          !maps[1 - biasIndex].isIdentity() || !maps[2].isIdentity())
        continue;
      biasValue = add.getInputs()[biasIndex];
      end = add;
    } else {
      auto broadcast = dyn_cast_or_null<linalg::GenericOp>(initial);
      if (!broadcast || broadcast.getInputs().size() != 1 ||
          broadcast.getBody()->getOperations().size() != 1)
        continue;
      auto maps = broadcast.getIndexingMapsArray();
      if (maps[0] != AffineMap::get(4, 0, rewriter.getAffineDimExpr(3)) ||
          !maps[1].isIdentity() ||
          broadcast.getBody()->getTerminator()->getOperand(0) !=
              broadcast.getBody()->getArgument(0))
        continue;
      biasValue = broadcast.getInputs()[0];
    }
    auto bias = frontend(biasValue, "frontend.int32_to_fp32");
    if (!quant || !bias)
      continue;
    auto reshape = bias->getOperand(1).getDefiningOp<tensor::CollapseShapeOp>();
    auto scale = reshape ? reshape.getSrc().getDefiningOp<linalg::GenericOp>()
                         : linalg::GenericOp();
    if (!scale || !pebble::elementwise(scale, "arith.mulf") ||
        !llvm::is_contained(scale.getInputs(), lhs->getOperand(1)) ||
        !llvm::is_contained(scale.getInputs(), rhs->getOperand(1)))
      continue;
    rewriter.setInsertionPoint(end);
    auto custom =
        CustomOp::create(rewriter, end->getLoc(), end->getResultTypes(),
                         depthwise ? "depthwise_conv2d_i8" : "conv2d_i8",
                         ValueRange{quant->getOperand(0), rhs->getOperand(0),
                                    bias->getOperand(0), quant->getOperand(1),
                                    lhs->getOperand(1), rhs->getOperand(1)});
    custom->setAttr("strides", rewriter.getDenseI64ArrayAttr(strides));
    custom->setAttr("dilations", rewriter.getDenseI64ArrayAttr(dilations));
    custom->setAttr("padding", rewriter.getDenseI64ArrayAttr(padding));
    rewriter.replaceOp(end, custom.getOutputs());
    if (depthwise) {
      rewriter.eraseOp(collapsed);
      rewriter.eraseOp(op);
    }
    if (pad && pad->use_empty())
      rewriter.eraseOp(pad);
    for (Operation *dead :
         {initial, bias, reshape.getOperation(), scale.getOperation(),
          transpose.getOperation(), rhs, lhs, quant})
      if (dead->use_empty())
        rewriter.eraseOp(dead);
  }

  // A convolution fuses its single-consumer chain [relu] [maxpool] [flatten]
  // ending in a quantization: the convolution requantizes for the consumer
  // and produces the INT8 tensor the consumer reads.
  SmallVector<CustomOp> roots;
  function.walk([&](CustomOp op) {
    if (op.getKind() == "conv2d_i8" || op.getKind() == "depthwise_conv2d_i8")
      roots.push_back(op);
  });
  DominanceInfo dominance(function);
  const auto &target = buckyball_target::getBuckyballTarget();
  for (auto root : roots) {
    Value value = root.getResult(0);
    auto convType = cast<RankedTensorType>(value.getType());
    // The chain in program order; the cycle markers among it.
    SmallVector<Operation *> links, traces;
    bool relu = false, flatten = false;
    int64_t pool = 1;
    while (value.hasOneUse()) {
      Operation *user = *value.getUsers().begin();
      if (user->getBlock() != root->getBlock())
        break;
      if (user->getName().getStringRef() == "buddy_trace.end") {
        if (cast<BoolAttr>(user->getAttr("tensor")).getValue())
          break;
        traces.push_back(user);
      } else if (auto maximum = dyn_cast<linalg::GenericOp>(user)) {
        if (relu || pool > 1 || flatten ||
            !linalg::isaElemwiseSingleBinaryOpInterface(maximum) ||
            !isa<arith::MaximumFOp>(maximum.getBody()->front()) ||
            !llvm::all_of(maximum.getIndexingMapsArray(),
                          [](AffineMap map) { return map.isIdentity(); }) ||
            !matchPattern(
                maximum.getInputs()[maximum.getInputs()[0] == value ? 1 : 0],
                m_AnyZeroFloat()))
          break;
        relu = true;
      } else if (auto pooling = dyn_cast<linalg::PoolingNhwcMaxOp>(user)) {
        // MAXPOOL pools square windows that tile the convolution output.
        auto filter =
            cast<RankedTensorType>(pooling.getInputs()[1].getType()).getShape();
        if (pool > 1 || flatten ||
            !llvm::all_of(pooling.getDilations().getValues<int64_t>(),
                          [](int64_t v) { return v == 1; }) ||
            filter[0] != filter[1] ||
            !llvm::all_of(pooling.getStrides().getValues<int64_t>(),
                          [&](int64_t v) { return v == filter[0]; }) ||
            convType.getDimSize(1) % filter[0] ||
            convType.getDimSize(2) % filter[0])
          break;
        Value initial = pooling.getOutputs()[0];
        if (auto fill = initial.getDefiningOp<linalg::FillOp>())
          initial = fill.getInputs()[0];
        APFloat minimum(0.0f);
        bool negativeLimit =
            matchPattern(initial, m_ConstantFloat(&minimum)) &&
            minimum.convertToFloat() <= -std::numeric_limits<float>::max();
        if (!negativeLimit &&
            !(relu && matchPattern(initial, m_AnyZeroFloat())))
          break;
        pool = filter[0];
      } else if (auto transpose = dyn_cast<linalg::TransposeOp>(user)) {
        if (flatten ||
            transpose.getPermutation() != ArrayRef<int64_t>({0, 3, 1, 2}) ||
            !transpose->getResult(0).hasOneUse())
          break;
        auto reshape = dyn_cast<tensor::CollapseShapeOp>(
            *transpose->getResult(0).getUsers().begin());
        if (!reshape || reshape.getReassociationIndices() !=
                            SmallVector<ReassociationIndices>{{0}, {1, 2, 3}})
          break;
        flatten = true;
        links.push_back(transpose);
        user = reshape;
      } else {
        break;
      }
      links.push_back(user);
      value = user->getResult(0);
    }
    if (!value.hasOneUse())
      continue;
    Operation *consumer = *value.getUsers().begin();
    auto next = dyn_cast<CustomOp>(consumer);
    bool terminal =
        consumer->getName().getStringRef() == "frontend.fp32_to_int8";
    if (terminal ? flatten
                 : (!next || next.getInputs()[0] != value ||
                    (flatten ? next.getKind() != "linear_i8"
                             : next.getKind() != "conv2d_i8" &&
                                   next.getKind() != "depthwise_conv2d_i8")))
      continue;
    Value multiplier = terminal ? consumer->getOperand(1) : next.getInputs()[3];
    if (!dominance.properlyDominates(multiplier, root))
      continue;
    bool barrier = false;
    for (Operation *between = root->getNextNode(); between != consumer;
         between = between->getNextNode())
      if (!isMemoryEffectFree(between) &&
          between->getName().getDialectNamespace() != "buddy_trace") {
        barrier = true;
        break;
      }
    auto strides = cast<DenseI64ArrayAttr>(root->getAttr("strides"));
    auto weightType = cast<RankedTensorType>(root.getInputs()[1].getType());
    bool depthwise = root.getKind() == "depthwise_conv2d_i8";
    int64_t kernel = weightType.getDimSize(2);
    int64_t blocksIn =
        (cast<RankedTensorType>(root.getInputs()[0].getType()).getDimSize(3) +
         15) /
        16;
    int64_t height = convType.getDimSize(1), width = convType.getDimSize(2),
            channels = convType.getDimSize(3);
    pebble::ConvTiling tiling =
        pebble::convTiling(height / pool, width / pool, pool, strides[0],
                           kernel, blocksIn, depthwise,
                           cast<RankedTensorType>(root.getInputs()[0].getType())
                               .getElementType()
                               .isF32());
    auto producer = root.getInputs()[0].getDefiningOp<CustomOp>();
    bool residentInput = producer && producer->hasAttr("custom.resident");
    if (barrier || (!residentInput && !tiling.side))
      continue;

    // The INT8 result stays in banks for a consumer that reads it whole: one
    // tile covers the convolution, and both stages fit the banks with one
    // group per channel block.
    int64_t blocksOut = (channels + 15) / 16;
    int64_t side = height / pool;
    bool resident = false;
    if (!depthwise && height == width &&
        (residentInput || tiling.side == height)) {
      int64_t groups = pebble::im2colGroups(
          kernel, residentInput ? blocksIn : tiling.chunk, height * width);
      bool fits = (residentInput ? blocksIn : 0) + blocksOut + 2 * groups + 4 <=
                  target.bankNum - 1;
      if (flatten) {
        auto linearWeight =
            cast<RankedTensorType>(next.getInputs()[1].getType());
        resident = fits && side * side == 16 && channels == 16 &&
                   linearWeight.getDimSize(0) <= target.bankDepth;
      } else if (!terminal && next.getKind() == "conv2d_i8") {
        auto nextWeight = cast<RankedTensorType>(next.getInputs()[1].getType());
        auto nextPadding = cast<DenseI64ArrayAttr>(next->getAttr("padding"));
        int64_t nextStride =
            cast<DenseI64ArrayAttr>(next->getAttr("strides"))[0];
        int64_t nextKernel = nextWeight.getDimSize(2);
        int64_t span = side + 2 * nextPadding[0] - nextKernel;
        int64_t outputs = span / nextStride + 1;
        int64_t nextGroups =
            pebble::im2colGroups(nextKernel, blocksOut, outputs * outputs);
        int64_t nextBlocks = (nextWeight.getDimSize(0) + 15) / 16;
        resident =
            fits && llvm::all_equal(nextPadding.asArrayRef()) &&
            nextPadding[0] <= buckyball_target::getBuckyballBallParam(
                                  "IM2COL", "maxPadding") &&
            side <=
                buckyball_target::getBuckyballBallParam("IM2COL", "maxIter") &&
            span >= 0 && span % nextStride == 0 &&
            (outputs * outputs + pebble::tileRows() - 1) / pebble::tileRows() *
                    pebble::tileRows() <=
                pebble::matmulRows() &&
            nextGroups &&
            blocksOut + nextBlocks + 2 * nextGroups + 4 <= target.bankNum - 1;
      }
    }

    rewriter.setInsertionPoint(root);
    SmallVector<Value> inputs(root.getInputs());
    inputs.push_back(multiplier);
    auto fused = CustomOp::create(
        rewriter, root.getLoc(),
        TypeRange{RankedTensorType::get({1, side, width / pool, channels},
                                        rewriter.getI8Type())},
        root.getKind(), inputs);
    fused->setAttrs(root->getAttrs());
    fused->setAttr("relu", rewriter.getBoolAttr(relu));
    fused->setAttr("pool", rewriter.getI64IntegerAttr(pool));
    if (resident)
      fused->setAttr("custom.resident", rewriter.getUnitAttr());
    Value codes = fused.getResult(0);
    // Cycle markers now observe the fused result.
    rewriter.setInsertionPointAfter(fused);
    for (Operation *trace : traces) {
      Operation *moved = rewriter.clone(*trace);
      moved->setOperand(0, codes);
      moved->getResult(0).setType(codes.getType());
    }
    rewriter.setInsertionPoint(consumer);
    if (terminal) {
      rewriter.replaceOp(consumer, codes);
    } else if (flatten && resident) {
      next->setOperand(
          0, CustomOp::create(rewriter, root.getLoc(),
                              TypeRange{cast<RankedTensorType>(value.getType())
                                            .clone(rewriter.getI8Type())},
                              "flatten_nchw", ValueRange{codes})
                 .getResult(0));
    } else if (flatten) {
      auto nchw = RankedTensorType::get({1, channels, side, width / pool},
                                        rewriter.getI8Type());
      Value empty = tensor::EmptyOp::create(
          rewriter, root.getLoc(), nchw.getShape(), nchw.getElementType());
      Value transposed =
          linalg::TransposeOp::create(rewriter, root.getLoc(), codes, empty,
                                      ArrayRef<int64_t>{0, 3, 1, 2})
              .getResult()[0];
      next->setOperand(0, tensor::CollapseShapeOp::create(
                              rewriter, root.getLoc(), transposed,
                              ArrayRef<ReassociationIndices>{{0}, {1, 2, 3}}));
    } else {
      next->setOperand(0, codes);
    }
    for (Operation *link : llvm::reverse(links))
      rewriter.eraseOp(link);
    rewriter.eraseOp(root);
  }
}
