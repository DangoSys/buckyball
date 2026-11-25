//===- OutlineLayerNorm.cpp - Layer normalization as LayerNormBall work ---===//
//
// Part of convert-linalg-to-hart: finds the layer normalization of FP32
// tensors over their last dimension (w) in the form of the frontend's
// decomposition,
//
//   s    = reduce sum (x)               (init 0)
//   mean = c * s                        (c the constant 1/w)
//   d    = x - mean                     (mean broadcast along the row)
//   q    = reduce sum (d * d)           (init 0)
//   r    = rsqrt (c * q + eps)          (eps a constant)
//   y    = ((d * r) * gamma) + beta     (gamma, beta [w] broadcast)
//
// and replaces y by a bbacc.custom "layernorm" on x collapsed to [rows, w]
// (16 <= w <= 2048), gamma and beta, with eps, expanded back to the shape of
// y. RMSNorm,
//
//   q    = reduce sum (x ^ 2)           (fpowi 2 or x * x, init 0)
//   r    = rsqrt (c * q + eps)
//   y    = gamma * (x * r)            (gamma [w] broadcast or [1, ..., 1, w])
//
// becomes the same with `rms` and no beta. convert-linalg-to-hart then runs it
// on the thread of its hart, whose convert-linalg-to-acc lowers it to
// LayerNormBall.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

using namespace mlir;

namespace {
// The only user of `value`, if it has exactly one.
Operation *soleUser(Value value) {
  return value.hasOneUse() ? *value.getUsers().begin() : nullptr;
}

// A pointwise generic of one T over its `inputs` inputs, in any order.
template <typename T>
linalg::GenericOp pointwise(Operation *op, unsigned inputs) {
  auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic || generic.getNumDpsInputs() != inputs ||
      generic.getNumResults() != 1 || generic.getNumReductionLoops())
    return {};
  Block &body = generic.getRegion().front();
  if (body.getOperations().size() != 2 || !isa<T>(body.front()))
    return {};
  for (Value operand : body.front().getOperands())
    if (!llvm::is_contained(body.getArguments(), operand))
      return {};
  return generic;
}

// The FP32 value of a splat constant tensor.
std::optional<float> splat(Value value) {
  DenseElementsAttr constant;
  if (!matchPattern(value, m_Constant(&constant)) || !constant.isSplat())
    return {};
  return constant.getSplatValue<APFloat>().convertToFloat();
}

// A sum over the last dimension from +0.
bool rowSum(linalg::ReduceOp op) {
  auto input = cast<RankedTensorType>(op.getInputs()[0].getType());
  if (op.getInputs().size() != 1 || op.getDimensions().size() != 1 ||
      op.getDimensions()[0] != input.getRank() - 1)
    return false;
  Block &body = op.getCombiner().front();
  if (body.getOperations().size() != 2 || !isa<arith::AddFOp>(body.front()))
    return false;
  auto fill = op.getInits()[0].getDefiningOp<linalg::FillOp>();
  auto zero = fill ? fill.getInputs()[0].getDefiningOp<arith::ConstantFloatOp>()
                   : arith::ConstantFloatOp();
  return zero && zero.value().isPosZero();
}

// The other input of a binary generic than `value`.
Value other(linalg::GenericOp op, Value value) {
  auto inputs = op.getDpsInputs();
  return inputs[0] == value ? inputs[1] : inputs[0];
}

// c * (the row sum `sum`, expanded to [..., 1]) with c = 1/w: the product.
linalg::GenericOp averaged(linalg::ReduceOp sum, int64_t w) {
  auto expand =
      dyn_cast_or_null<tensor::ExpandShapeOp>(soleUser(sum.getResult(0)));
  auto product = expand ? pointwise<arith::MulFOp>(soleUser(expand), 2)
                        : linalg::GenericOp();
  if (!product)
    return {};
  std::optional<float> c = splat(other(product, expand.getResult()));
  return c && *c == 1.0f / float(w) ? product : linalg::GenericOp();
}

// The [w] source of a row vector broadcast to the shape of the rows.
Value rowVector(Value value, int64_t w) {
  auto expand = value.getDefiningOp<tensor::ExpandShapeOp>();
  if (!expand)
    return {};
  auto source = cast<RankedTensorType>(expand.getSrc().getType());
  return source.getRank() == 1 && source.getDimSize(0) == w ? expand.getSrc()
                                                            : Value();
}
} // namespace

// Called by outlineOperators before it fuses row arithmetic.
void outlineLayerNorm(ModuleOp module) {
  SmallVector<linalg::ReduceOp> sums;
  module.walk([&](linalg::ReduceOp op) {
    if (rowSum(op))
      sums.push_back(op);
  });
  for (linalg::ReduceOp sum : sums) {
    if (!sum->getBlock())
      continue;
    Value x = sum.getInputs()[0];
    auto type = cast<RankedTensorType>(x.getType());
    int64_t w = type.getShape().back();
    // x may have other users: the residual stream reads it again.
    if (!type.hasStaticShape() || !type.getElementType().isF32() || w < 16 ||
        w > 2048)
      continue;
    linalg::GenericOp mean = averaged(sum, w);
    auto d = mean ? pointwise<arith::SubFOp>(soleUser(mean.getResult(0)), 2)
                  : linalg::GenericOp();
    if (!d || d.getDpsInputs()[0] != x ||
        d.getDpsInputs()[1] != mean.getResult(0))
      continue;
    // d feeds its square and the normalization, nothing else.
    linalg::GenericOp square, normalized;
    for (Operation *user : d.getResult(0).getUsers()) {
      auto product = pointwise<arith::MulFOp>(user, 2);
      if (product && product.getDpsInputs()[0] == d.getResult(0) &&
          product.getDpsInputs()[1] == d.getResult(0))
        square = product;
      else if (product)
        normalized = product;
    }
    if (!square || !normalized ||
        SmallPtrSet<Operation *, 2>(d.getResult(0).user_begin(),
                                    d.getResult(0).user_end())
                .size() != 2)
      continue;
    auto squares =
        dyn_cast_or_null<linalg::ReduceOp>(soleUser(square.getResult(0)));
    linalg::GenericOp variance =
        squares && rowSum(squares) ? averaged(squares, w) : linalg::GenericOp();
    auto shifted =
        variance ? pointwise<arith::AddFOp>(soleUser(variance.getResult(0)), 2)
                 : linalg::GenericOp();
    std::optional<float> eps =
        shifted ? splat(other(shifted, variance.getResult(0))) : std::nullopt;
    auto root =
        eps ? pointwise<math::RsqrtOp>(soleUser(shifted.getResult(0)), 1)
            : linalg::GenericOp();
    if (!root || soleUser(root.getResult(0)) != normalized.getOperation() ||
        normalized.getDpsInputs()[0] != d.getResult(0))
      continue;
    auto scaled =
        pointwise<arith::MulFOp>(soleUser(normalized.getResult(0)), 2);
    Value gamma =
        scaled ? rowVector(other(scaled, normalized.getResult(0)), w) : Value();
    auto y = gamma ? pointwise<arith::AddFOp>(soleUser(scaled.getResult(0)), 2)
                   : linalg::GenericOp();
    Value beta = y ? rowVector(other(y, scaled.getResult(0)), w) : Value();
    if (!beta)
      continue;

    OpBuilder b(y);
    Location loc = y.getLoc();
    SmallVector<ReassociationIndices> rows(2);
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis)
      rows[0].push_back(axis);
    rows[1].push_back(type.getRank() - 1);
    Value flat = tensor::CollapseShapeOp::create(b, loc, x, rows);
    auto flatType = cast<RankedTensorType>(flat.getType());
    auto norm =
        ::buddy::acc::CustomOp::create(b, loc, TypeRange{flatType}, "layernorm",
                                       ValueRange{flat, gamma, beta});
    norm->setAttr("eps", b.getF32FloatAttr(*eps));
    Value result =
        tensor::ExpandShapeOp::create(b, loc, type, norm.getResult(0), rows);
    y.getResult(0).replaceAllUsesWith(result);
    Operation *chain[] = {
        y,
        y.getDpsInputs()[0] == scaled.getResult(0)
            ? y.getDpsInputs()[1].getDefiningOp()
            : y.getDpsInputs()[0].getDefiningOp(),
        scaled,
        other(scaled, normalized.getResult(0)).getDefiningOp(),
        normalized,
        root,
        shifted,
        variance,
        squares.getResult(0).getUsers().empty()
            ? nullptr
            : *squares.getResult(0).getUsers().begin(),
        squares,
        square,
        d,
        mean,
        *sum.getResult(0).getUsers().begin(),
        sum};
    for (Operation *op : chain)
      if (op && op->use_empty())
        op->erase();
  }

  // RMSNorm.
  sums.clear();
  module.walk([&](linalg::ReduceOp op) {
    if (rowSum(op))
      sums.push_back(op);
  });
  for (linalg::ReduceOp squares : sums) {
    if (!squares->getBlock())
      continue;
    // x ^ 2: fpowi(x, 2) of one input, or x * x.
    auto square = squares.getInputs()[0].getDefiningOp<linalg::GenericOp>();
    if (!square || !square->hasOneUse() || square.getNumResults() != 1 ||
        square.getNumReductionLoops())
      continue;
    Block &body = square.getRegion().front();
    Value x;
    if (body.getOperations().size() == 2) {
      if (auto power = dyn_cast<math::FPowIOp>(body.front())) {
        auto two = power.getRhs().getDefiningOp<arith::ConstantIntOp>();
        if (square.getNumDpsInputs() == 1 && two && two.value() == 2 &&
            power.getLhs() == body.getArgument(0))
          x = square.getDpsInputs()[0];
      } else if (auto product = dyn_cast<arith::MulFOp>(body.front())) {
        if (square.getNumDpsInputs() == 2 &&
            square.getDpsInputs()[0] == square.getDpsInputs()[1] &&
            llvm::is_contained(body.getArguments(), product.getLhs()) &&
            llvm::is_contained(body.getArguments(), product.getRhs()))
          x = square.getDpsInputs()[0];
      }
    }
    if (!x)
      continue;
    auto type = cast<RankedTensorType>(x.getType());
    int64_t w = type.getShape().back();
    if (!type.hasStaticShape() || !type.getElementType().isF32() || w < 16 ||
        w > 2048 || square.getResult(0).getType() != type)
      continue;
    linalg::GenericOp variance = averaged(squares, w);
    auto shifted =
        variance ? pointwise<arith::AddFOp>(soleUser(variance.getResult(0)), 2)
                 : linalg::GenericOp();
    std::optional<float> eps =
        shifted ? splat(other(shifted, variance.getResult(0))) : std::nullopt;
    auto root =
        eps ? pointwise<math::RsqrtOp>(soleUser(shifted.getResult(0)), 1)
            : linalg::GenericOp();
    auto normalized =
        root ? pointwise<arith::MulFOp>(soleUser(root.getResult(0)), 2)
             : linalg::GenericOp();
    if (!normalized || other(normalized, root.getResult(0)) != x)
      continue;
    auto scaled =
        pointwise<arith::MulFOp>(soleUser(normalized.getResult(0)), 2);
    if (!scaled)
      continue;
    // gamma: a [w] parameter broadcast to the rows, or a computed row
    // [1, ..., 1, w] (the scale an adaptive RMSNorm takes from its condition).
    Value wide = other(scaled, normalized.getResult(0));
    Value gamma = rowVector(wide, w);
    auto wideType = cast<RankedTensorType>(wide.getType());
    if (!gamma &&
        !(wideType.hasStaticShape() && wideType.getNumElements() == w &&
          wideType.getShape().back() == w && wideType.getRank() > 1))
      continue;

    OpBuilder b(scaled);
    Location loc = scaled.getLoc();
    if (!gamma) {
      ReassociationIndices all;
      for (int64_t axis = 0; axis < wideType.getRank(); ++axis)
        all.push_back(axis);
      gamma = tensor::CollapseShapeOp::create(b, loc, wide, {all});
    }
    SmallVector<ReassociationIndices> rows(2);
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis)
      rows[0].push_back(axis);
    rows[1].push_back(type.getRank() - 1);
    Value flat = tensor::CollapseShapeOp::create(b, loc, x, rows);
    auto norm =
        ::buddy::acc::CustomOp::create(b, loc, TypeRange{flat.getType()},
                                       "layernorm", ValueRange{flat, gamma});
    norm->setAttr("eps", b.getF32FloatAttr(*eps));
    norm->setAttr("rms", b.getUnitAttr());
    Value result =
        tensor::ExpandShapeOp::create(b, loc, type, norm.getResult(0), rows);
    scaled.getResult(0).replaceAllUsesWith(result);
    Operation *chain[] = {
        scaled,
        other(scaled, normalized.getResult(0)).getDefiningOp(),
        normalized,
        root,
        shifted,
        variance,
        *squares.getResult(0).getUsers().begin(),
        squares,
        square};
    for (Operation *op : chain)
      if (op && op->use_empty())
        op->erase();
  }
}
