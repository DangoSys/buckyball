//===- FormSpatialOps.cpp - Recognize quantized convolution chains --------===//
#include "Custom/CustomOps.h"
#include "Lowering.h"
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
    custom->setAttr("strides",
                    rewriter.getDenseI64ArrayAttr(llvm::to_vector(
                        cast<DenseIntElementsAttr>(op->getAttr("strides"))
                            .getValues<int64_t>())));
    custom->setAttr("dilations",
                    rewriter.getDenseI64ArrayAttr(llvm::to_vector(
                        cast<DenseIntElementsAttr>(op->getAttr("dilations"))
                            .getValues<int64_t>())));
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

  // A spatial stage retains its ordinary tensor semantics until bank lowering.
  // Only single-consumer links ending in another quantized computation fuse.
  SmallVector<CustomOp> roots;
  function.walk([&](CustomOp op) {
    if (op.getKind() == "conv2d_i8" || op.getKind() == "depthwise_conv2d_i8")
      roots.push_back(op);
  });
  DominanceInfo dominance(function);
  int64_t group = 0;
  for (auto root : roots) {
    Value value = root.getResult(0);
    SmallVector<Operation *> links;
    bool relu = false, pool = false, flatten = false;
    while (value.hasOneUse()) {
      Operation *user = *value.getUsers().begin();
      if (user->getBlock() != root->getBlock())
        break;
      if (user->getName().getStringRef() == "buddy_trace.end") {
        if (cast<BoolAttr>(user->getAttr("tensor")).getValue())
          break;
      } else if (auto maximum = dyn_cast<linalg::GenericOp>(user)) {
        if (relu || pool ||
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
        if (pool || flatten ||
            !llvm::all_of(pooling.getDilations().getValues<int64_t>(),
                          [](int64_t v) { return v == 1; }))
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
        pool = true;
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
    bool terminalQuant =
        consumer->getName().getStringRef() == "frontend.fp32_to_int8";
    if (!terminalQuant && (!next ||
                           (next.getKind() != "conv2d_i8" &&
                            next.getKind() != "depthwise_conv2d_i8" &&
                            next.getKind() != "linear_i8") ||
                           next.getInputs()[0] != value))
      continue;
    Value multiplier =
        terminalQuant ? consumer->getOperand(1) : next.getInputs()[3];
    if ((terminalQuant && flatten) ||
        !dominance.properlyDominates(multiplier, root))
      continue;
    bool barrier = false;
    for (Operation *between = root->getNextNode(); between != consumer;
         between = between->getNextNode())
      if (!isMemoryEffectFree(between) &&
          between->getName().getDialectNamespace() != "buddy_trace") {
        barrier = true;
        break;
      }
    if (barrier)
      continue;
    if (terminalQuant)
      links.push_back(consumer);
    auto id = rewriter.getI64IntegerAttr(group++);
    root->setAttr("custom.spatial_chain", id);
    for (Operation *link : links) {
      rewriter.setInsertionPoint(link);
      if (auto maximum = dyn_cast<linalg::GenericOp>(link)) {
        Value input = matchPattern(maximum.getInputs()[0], m_AnyZeroFloat())
                          ? maximum.getInputs()[1]
                          : maximum.getInputs()[0];
        auto custom = CustomOp::create(rewriter, maximum.getLoc(),
                                       maximum->getResultTypes(), "relu",
                                       ValueRange{input});
        custom->setAttr("custom.spatial_chain", id);
        rewriter.replaceOp(maximum, custom.getOutputs());
      } else if (auto pooling = dyn_cast<linalg::PoolingNhwcMaxOp>(link)) {
        auto custom = CustomOp::create(rewriter, pooling.getLoc(),
                                       pooling->getResultTypes(), "maxpool2d",
                                       ValueRange{pooling.getInputs()[0]});
        custom->setAttr("kernel", rewriter.getDenseI64ArrayAttr(
                                      cast<RankedTensorType>(
                                          pooling.getInputs()[1].getType())
                                          .getShape()));
        custom->setAttr("strides",
                        rewriter.getDenseI64ArrayAttr(llvm::to_vector(
                            pooling.getStrides().getValues<int64_t>())));
        custom->setAttr("custom.spatial_chain", id);
        rewriter.replaceOp(pooling, custom.getOutputs());
      } else if (auto reshape = dyn_cast<tensor::CollapseShapeOp>(link)) {
        auto transpose = reshape.getSrc().getDefiningOp<linalg::TransposeOp>();
        auto custom = CustomOp::create(
            rewriter, reshape.getLoc(), reshape->getResultTypes(),
            "flatten_nchw", ValueRange{transpose.getInput()});
        custom->setAttr("custom.spatial_chain", id);
        rewriter.replaceOp(reshape, custom.getOutputs());
        rewriter.eraseOp(transpose);
      } else if (!isa<linalg::TransposeOp>(link)) {
        link->setAttr("custom.spatial_chain", id);
      }
    }
  }
}
