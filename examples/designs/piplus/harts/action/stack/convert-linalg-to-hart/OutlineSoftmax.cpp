//===- OutlineSoftmax.cpp - Softmax over rows as SoftmaxBall work ---------===//
//
// Part of convert-linalg-to-hart: finds the softmax of FP32 tensors over their
// last dimension in the form of the frontend's decomposition,
//
//   m = reduce max (x)         (init the lowest finite FP32 or -inf)
//   d = x - m                  (m broadcast along the row)
//   e = exp (d)
//   s = reduce sum (e)         (init 0)
//   y = e / s                  (s broadcast along the row)
//
// and replaces y by a bbacc.custom "softmax" on x collapsed to [rows, w]
// (16 <= w <= 2048), expanded back to the shape of y. An x that is only a
// pointwise product of a value by a power of two 2^e (the 1/sqrt(d) of an
// attention with d a power of four) gives the softmax that value and
// scale_log2 = e: SoftmaxBall scales its rows by 2^e itself, the same FP32
// multiplication.
// convert-linalg-to-hart then runs it on the thread of its hart, whose
// convert-linalg-to-acc lowers it to SoftmaxBall.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include <cmath>

#include <limits>

using namespace mlir;

namespace {
// The only user of `value`, if it has exactly one.
Operation *soleUser(Value value) {
  return value.hasOneUse() ? *value.getUsers().begin() : nullptr;
}

// A reduction over the last dimension combining with T from `init`.
template <typename T>
bool isRowReduction(linalg::ReduceOp op, function_ref<bool(float)> init) {
  auto input = cast<RankedTensorType>(op.getInputs()[0].getType());
  if (op.getInputs().size() != 1 || op.getDimensions().size() != 1 ||
      op.getDimensions()[0] != input.getRank() - 1)
    return false;
  Block &body = op.getCombiner().front();
  if (body.getOperations().size() != 2 || !isa<T>(body.front()))
    return false;
  auto fill = op.getInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill)
    return false;
  auto constant = fill.getInputs()[0].getDefiningOp<arith::ConstantFloatOp>();
  return constant && init(constant.value().convertToFloat());
}

// A pointwise generic whose body is a single T of its inputs in order.
template <typename T> bool isPointwise(linalg::GenericOp op, unsigned inputs) {
  if (op.getNumDpsInputs() != inputs || op.getNumResults() != 1 ||
      op.getNumReductionLoops())
    return false;
  Block &body = op.getRegion().front();
  if (body.getOperations().size() != 2 || !isa<T>(body.front()))
    return false;
  for (unsigned i = 0; i < inputs; ++i)
    if (body.front().getOperand(i) != body.getArgument(i))
      return false;
  return true;
}

// The row value of a reduction, broadcast along the row: its expand_shape to
// [..., 1], the operand `i` of the generic `use`.
bool broadcasts(linalg::ReduceOp reduction, linalg::GenericOp use, unsigned i) {
  auto expand = use.getDpsInputs()[i].getDefiningOp<tensor::ExpandShapeOp>();
  return expand && expand.getSrc() == reduction.getResult(0) &&
         expand->hasOneUse() && reduction->hasOneUse();
}

} // namespace

// Called by outlineOperators before it fuses row arithmetic.
void outlineSoftmax(ModuleOp module) {
  SmallVector<linalg::ReduceOp> maxima;
  module.walk([&](linalg::ReduceOp op) {
    if (isRowReduction<arith::MaximumFOp>(op, [](float value) {
          return value == -std::numeric_limits<float>::max() ||
                 value == -std::numeric_limits<float>::infinity();
        }))
      maxima.push_back(op);
  });
  for (linalg::ReduceOp maximum : maxima) {
    Value x = maximum.getInputs()[0];
    auto type = cast<RankedTensorType>(x.getType());
    int64_t w = type.getShape().back();
    if (!type.hasStaticShape() || !type.getElementType().isF32() || w < 16 ||
        w > 2048)
      continue;
    auto expand =
        dyn_cast_or_null<tensor::ExpandShapeOp>(soleUser(maximum.getResult(0)));
    auto difference =
        expand ? dyn_cast_or_null<linalg::GenericOp>(soleUser(expand))
               : nullptr;
    if (!difference || !isPointwise<arith::SubFOp>(difference, 2) ||
        difference.getDpsInputs()[0] != x ||
        !broadcasts(maximum, difference, 1))
      continue;
    auto exponential =
        dyn_cast_or_null<linalg::GenericOp>(soleUser(difference.getResult(0)));
    if (!exponential || !isPointwise<math::ExpOp>(exponential, 1))
      continue;
    // e feeds the sum and the division, nothing else.
    Value e = exponential.getResult(0);
    linalg::ReduceOp sum;
    linalg::GenericOp quotient;
    for (Operation *user : e.getUsers()) {
      if (auto reduce = dyn_cast<linalg::ReduceOp>(user))
        sum = reduce;
      else if (auto generic = dyn_cast<linalg::GenericOp>(user))
        quotient = generic;
    }
    if (!sum || !quotient || std::distance(e.use_begin(), e.use_end()) != 2 ||
        !isRowReduction<arith::AddFOp>(
            sum, [](float value) { return value == 0; }) ||
        !isPointwise<arith::DivFOp>(quotient, 2) ||
        quotient.getDpsInputs()[0] != e || !broadcasts(sum, quotient, 1))
      continue;

    // x = v * 2^e, read by the softmax alone: the Ball scales v.
    Value source = x;
    int64_t scale = 0;
    auto product = x.getDefiningOp<linalg::GenericOp>();
    DenseElementsAttr factor;
    if (product && isPointwise<arith::MulFOp>(product, 2) &&
        std::distance(x.use_begin(), x.use_end()) == 2 &&
        product.getIndexingMapsArray()[0].isIdentity() &&
        product.getIndexingMapsArray().back().isIdentity() &&
        matchPattern(product.getDpsInputs()[1], m_Constant(&factor)) &&
        factor.isSplat()) {
      int exponent;
      double fraction = std::frexp(
          factor.getSplatValue<APFloat>().convertToDouble(), &exponent);
      if (fraction == 0.5 && exponent - 1 >= -126 && exponent - 1 <= 127) {
        source = product.getDpsInputs()[0];
        scale = exponent - 1;
      }
    }

    OpBuilder b(quotient);
    Location loc = quotient.getLoc();
    SmallVector<ReassociationIndices> rows(2);
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis)
      rows[0].push_back(axis);
    rows[1].push_back(type.getRank() - 1);
    Value flat = tensor::CollapseShapeOp::create(b, loc, source, rows);
    auto flatType = cast<RankedTensorType>(flat.getType());
    auto softmax = ::buddy::acc::CustomOp::create(b, loc, TypeRange{flatType},
                                                  "softmax", ValueRange{flat});
    softmax->setAttr("scale_log2", b.getI64IntegerAttr(scale));
    Value y =
        tensor::ExpandShapeOp::create(b, loc, type, softmax.getResult(0), rows);
    quotient.getResult(0).replaceAllUsesWith(y);
    Operation *chain[] = {quotient,
                          quotient.getDpsInputs()[1].getDefiningOp(),
                          sum,
                          sum.getInits()[0].getDefiningOp(),
                          exponential,
                          difference,
                          expand,
                          maximum,
                          maximum.getInits()[0].getDefiningOp(),
                          source == x ? nullptr : product.getOperation()};
    for (Operation *op : chain)
      if (op && op->use_empty())
        op->erase();
  }
}
