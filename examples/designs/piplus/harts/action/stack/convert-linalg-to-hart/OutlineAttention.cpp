//===- OutlineAttention.cpp - An attention as one thread computation ------===//
//
// Part of convert-linalg-to-hart, after the softmax is outlined
// (OutlineSoftmax.cpp) and before the row arithmetic is fused
// (OutlineOperators.cpp): an attention,
//
//   S = batch_matmul (dec(quant(q)), dec(quant(k^T)))     [H, Sq, T]
//       (or of q, k^T, P and v as they are: a recipe may leave their
//       quantization to the attention, which reads them through MXINT8)
//   X = S, S + bias, or select (mask, S * 2^e, lowest)    (through reshapes)
//   P = softmax (X * 2^e)                                  (one of the 2^e)
//   O = batch_matmul (dec(quant(P)), dec(quant(v)))        [H, Sq, D]
//
// becomes O = bbacc.custom "attention" (dec(quant(q)), the parts of
// dec(quant(k^T)), the parts of dec(quant(v)), bias) {scale_log2 = e,
// segments, masked}: the thread's convert-linalg-to-acc keeps S and P in its
// banks (LowerAttention.cpp), so the scores never move to the DDR.
//
//   - The parts: keys inserted one after another into one value (the
//     prefix's keys before a step's own) are read where they are, each part
//     k^T and v sliced through its views to its source (whole blocks of 32
//     keys each, so v's blocks along T are those of the whole), and the
//     value they were inserted into is no longer built.
//   - The mask [.., Sb, T] (true: the key is seen), the same for every head,
//     becomes the bias [Sb, T] the scores start from: +0.0 where it is true
//     and -2^100 elsewhere, which the softmax gives +0.0 as it does the
//     lowest value of the graph. It is a pointwise region of its own, an
//     operator of the GP domain. A bias [.., Sb, T] the graph adds to the
//     scores is read as it is.
//
// The operands keep their views, decodings and quantizations, which the
// thread reads in place as for a product.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <cmath>

using namespace mlir;
using namespace ::buddy::frontend;

Value sliceThrough(OpBuilder &b, Location loc, Value value, int64_t axis,
                   int64_t first, int64_t length);

namespace {
// `value` back through the reshapes and transposes that make it, each read
// by the next alone; the operations walked, nearest `value` first.
Value throughViews(Value value, SmallVectorImpl<Operation *> &chain) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             linalg::TransposeOp>(op) ||
        !op->getResult(0).hasOneUse())
      return value;
    chain.push_back(op);
    value = op->getOperand(0);
  }
  return value;
}

// A pointwise generic whose body is a single T of its inputs in order.
template <typename T> bool isPointwise(linalg::GenericOp op, unsigned inputs) {
  if (!op || op.getNumDpsInputs() != inputs || op.getNumResults() != 1 ||
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

// The value of a tensor of one repeated constant, or of the pointwise sum of
// two such.
std::optional<double> splat(Value value) {
  DenseElementsAttr constant;
  if (matchPattern(value, m_Constant(&constant)))
    return constant.isSplat() && isa<FloatType>(constant.getElementType())
               ? std::optional<double>(
                     constant.getSplatValue<APFloat>().convertToDouble())
               : std::nullopt;
  auto sum = value.getDefiningOp<linalg::GenericOp>();
  if (!isPointwise<arith::AddFOp>(sum, 2))
    return std::nullopt;
  auto a = splat(sum.getDpsInputs()[0]), b = splat(sum.getDpsInputs()[1]);
  return a && b ? std::optional<double>(*a + *b) : std::nullopt;
}

// Erases `value`'s operation if no one reads it, then those of its operands.
void prune(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op || !op->use_empty() ||
      !(isOpTriviallyDead(op) || isa<MXINT8ToFP32Op, FP32ToMXINT8Op>(op)))
    return;
  SmallVector<Value> operands(op->getOperands());
  op->erase();
  for (Value operand : operands)
    prune(operand);
}
} // namespace

void outlineAttention(ModuleOp module) {
  SmallVector<linalg::BatchMatmulOp> outputs;
  module.walk([&](linalg::BatchMatmulOp op) { outputs.push_back(op); });
  for (linalg::BatchMatmulOp output : outputs) {
    // O's A: P, or the decoding of its quantization.
    SmallVector<Operation *> chain;
    Value p = throughViews(output.getInputs()[0], chain);
    if (auto decode = p.getDefiningOp<MXINT8ToFP32Op>()) {
      auto quant = decode->hasOneUse()
                       ? decode.getInput().getDefiningOp<FP32ToMXINT8Op>()
                       : FP32ToMXINT8Op();
      if (!quant || !quant->hasOneUse())
        continue;
      chain.append({decode, quant});
      p = throughViews(quant.getInput(), chain);
    }
    auto softmax = p.getDefiningOp<::buddy::acc::CustomOp>();
    if (!softmax || softmax.getKind() != "softmax" || !softmax->hasOneUse())
      continue;
    chain.push_back(softmax);
    int64_t scale = cast<IntegerAttr>(softmax->getAttr("scale_log2")).getInt();
    Value s = throughViews(softmax.getInputs()[0], chain);
    // The scores of the keys the mask sees, times a power of two, or the
    // scores plus a bias.
    Value mask, lowest, bias;
    if (auto sum = s.getDefiningOp<linalg::GenericOp>();
        isPointwise<arith::AddFOp>(sum, 2) && sum->hasOneUse()) {
      auto biasType = cast<RankedTensorType>(sum.getDpsInputs()[1].getType());
      auto type = cast<RankedTensorType>(sum.getResult(0).getType());
      int64_t rank = biasType.getRank();
      if (rank < 2 || rank != type.getRank() ||
          biasType.getShape().take_back(2) != type.getShape().take_back(2) ||
          biasType.getNumElements() !=
              biasType.getDimSize(rank - 2) * biasType.getDimSize(rank - 1))
        continue;
      bias = sum.getDpsInputs()[1];
      chain.push_back(sum);
      s = throughViews(sum.getDpsInputs()[0], chain);
    }
    if (auto select = s.getDefiningOp<linalg::GenericOp>();
        isPointwise<arith::SelectOp>(select, 3) && select->hasOneUse()) {
      auto low = splat(select.getDpsInputs()[2]);
      auto maskType =
          cast<RankedTensorType>(select.getDpsInputs()[0].getType());
      auto type = cast<RankedTensorType>(select.getResult(0).getType());
      int64_t rank = maskType.getRank();
      if (!low || *low > -1e30 || scale != 0 || rank < 2 ||
          rank != type.getRank() ||
          maskType.getShape().take_back(2) != type.getShape().take_back(2) ||
          maskType.getNumElements() !=
              maskType.getDimSize(rank - 2) * maskType.getDimSize(rank - 1))
        continue;
      mask = select.getDpsInputs()[0];
      lowest = select.getDpsInputs()[2];
      chain.push_back(select);
      s = throughViews(select.getDpsInputs()[1], chain);
      if (auto product = s.getDefiningOp<linalg::GenericOp>();
          isPointwise<arith::MulFOp>(product, 2) && product->hasOneUse()) {
        auto factor = splat(product.getDpsInputs()[1]);
        int exponent;
        if (!factor || std::frexp(*factor, &exponent) != 0.5 ||
            exponent - 1 < -126 || exponent - 1 > 127)
          continue;
        scale = exponent - 1;
        chain.push_back(product);
        s = throughViews(product.getDpsInputs()[0], chain);
      }
    }
    auto scores = s.getDefiningOp<linalg::BatchMatmulOp>();
    // Both products start from +0.0.
    auto zero = [](linalg::BatchMatmulOp product) {
      auto fill = product.getOutputs()[0].getDefiningOp<linalg::FillOp>();
      return fill && matchPattern(fill.getInputs()[0], m_PosZeroFloat());
    };
    if (!scores || !scores->hasOneUse() || !zero(scores) || !zero(output))
      continue;
    // Whole MXINT8 blocks along the head's values and the keys.
    auto scoreType = cast<RankedTensorType>(scores.getResult(0).getType());
    int64_t head =
        cast<RankedTensorType>(scores.getInputs()[0].getType()).getDimSize(2);
    int64_t keys = scoreType.getDimSize(2);
    if (head % 32 || keys % 32)
      continue;
    chain.push_back(scores);
    OpBuilder b(output);
    Location loc = output.getLoc();
    // An operand the graph does not quantize (a recipe that leaves the
    // attention to the device) is read through MXINT8 as the thread reads
    // it: q and k along the head's values, v along the keys.
    auto read = [&](Value value, int64_t axis) -> Value {
      SmallVector<Operation *> views;
      if (throughViews(value, views).getDefiningOp<MXINT8ToFP32Op>())
        return value;
      auto type = cast<RankedTensorType>(value.getType());
      SmallVector<int64_t> packed(type.getShape());
      packed[axis] += packed[axis] / 32;
      Value codes = FP32ToMXINT8Op::create(
          b, loc, RankedTensorType::get(packed, b.getI8Type()), value, axis);
      return MXINT8ToFP32Op::create(b, loc, type, codes, axis);
    };
    // The parts of the keys: the values inserted along one axis into the
    // value k^T is (or is the quantization of), (first key, keys) each;
    // found before the operands gain another reader below.
    SmallVector<std::pair<int64_t, int64_t>> parts;
    {
      SmallVector<Operation *> views;
      Value source = throughViews(scores.getInputs()[1], views);
      if (auto decoded = source.getDefiningOp<MXINT8ToFP32Op>())
        if (auto codes = decoded.getInput().getDefiningOp<FP32ToMXINT8Op>())
          source = throughViews(codes.getInput(), views);
      for (auto insert = source.getDefiningOp<tensor::InsertSliceOp>(); insert;
           insert = insert.getDest().getDefiningOp<tensor::InsertSliceOp>())
        for (int64_t axis = 0; axis < insert.getDestType().getRank(); ++axis)
          if (insert.getSourceType().getRank() ==
                  insert.getDestType().getRank() &&
              insert.getSourceType().getDimSize(axis) !=
                  insert.getDestType().getDimSize(axis))
            parts.push_back(
                {*getConstantIntValue(insert.getMixedOffsets()[axis]),
                 insert.getSourceType().getDimSize(axis)});
      llvm::sort(parts);
      int64_t next = 0;
      for (auto [first, length] : parts)
        next = first == next && length % 32 == 0 ? first + length : -1;
      if (next != keys)
        parts.clear();
    }
    Value q = read(scores.getInputs()[0], 2);
    Value k = read(scores.getInputs()[1], 1),
          v = read(output.getInputs()[1], 1);
    SmallVector<Value> inputs{q};
    for (auto [first, length] : parts)
      inputs.push_back(sliceThrough(b, loc, k, 2, first, length));
    for (auto [first, length] : parts)
      inputs.push_back(sliceThrough(b, loc, v, 1, first, length));
    if (parts.empty())
      inputs.append({k, v});
    // The mask or the bias as [Sb, T].
    auto rows = [&](Value value) -> Value {
      int64_t rank = cast<RankedTensorType>(value.getType()).getRank();
      if (rank == 2)
        return value;
      SmallVector<ReassociationIndices> groups(2);
      for (int64_t axis = 0; axis < rank - 1; ++axis)
        groups[0].push_back(axis);
      groups[1].push_back(rank - 1);
      return tensor::CollapseShapeOp::create(b, loc, value, groups);
    };
    if (bias)
      inputs.push_back(rows(bias));
    if (mask) {
      Value flat = rows(mask);
      auto type = RankedTensorType::get(
          cast<RankedTensorType>(flat.getType()).getShape(), b.getF32Type());
      // The bias is the compiler's, no value of the source graph: a named
      // location with no source under it.
      Location generated = NameLoc::get(b.getStringAttr("attention bias"));
      Value empty = tensor::EmptyOp::create(b, generated, type.getShape(),
                                            type.getElementType());
      SmallVector<AffineMap> maps(2, b.getMultiDimIdentityMap(2));
      SmallVector<utils::IteratorType> iterators(2,
                                                 utils::IteratorType::parallel);
      inputs.push_back(
          linalg::GenericOp::create(
              b, generated, TypeRange{type}, ValueRange{flat},
              ValueRange{empty}, maps, iterators,
              [](OpBuilder &b, Location loc, ValueRange arguments) {
                Value seen =
                    arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0f));
                Value unseen = arith::ConstantOp::create(
                    b, loc, b.getF32FloatAttr(-0x1p100f));
                linalg::YieldOp::create(
                    b, loc,
                    arith::SelectOp::create(b, loc, arguments[0], seen, unseen)
                        .getResult());
              })
              .getResult(0));
    }
    auto attention = ::buddy::acc::CustomOp::create(
        b, loc, TypeRange{output.getResult(0).getType()}, "attention", inputs);
    attention->setAttr("scale_log2", b.getI64IntegerAttr(scale));
    attention->setAttr("segments",
                       b.getI64IntegerAttr(std::max<size_t>(parts.size(), 1)));
    if (mask || bias)
      attention->setAttr("masked", b.getUnitAttr());
    output.getResult(0).replaceAllUsesWith(attention.getResult(0));
    Operation *fills[] = {output.getOutputs()[0].getDefiningOp(),
                          scores.getOutputs()[0].getDefiningOp()};
    output->erase();
    for (Operation *op : chain)
      op->erase();
    for (Operation *fill : fills)
      if (fill->use_empty())
        fill->erase();
    // The whole k^T and v and the value their parts were inserted into, and
    // the lowest value of the mask.
    prune(k);
    prune(v);
    if (lowest)
      prune(lowest);
  }
}
