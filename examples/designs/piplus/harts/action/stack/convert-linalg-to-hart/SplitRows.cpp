//===- SplitRows.cpp - The rows of a graph's work over the hart's threads -===//
//
// Part of convert-linalg-to-hart: work whose rows are independent splits into
// one part per thread of the hart, the parts' results inserted into the
// whole; the placement puts each part on its own thread. The rows split in
// units of the array rows of mxint8mm, the last part the rest; a product
// of few rows splits into fewer parts, each of which loads the weight.
//
//   - a linalg.matmul of an online quantized A over rows M (A the decoding of
//     the quantization of x [M, K], or of x [1, M, K] collapsed), with the
//     terms of its weight (FuseBias.cpp) and its start: each part quantizes
//     its rows of x, its products read their own copies of the weight views
//     (so the parts do not join through the weight) and start from their
//     rows of the start (zero, the bias, the residual or both);
//   - a linalg.batch_matmul (the heads of an attention) over its batches:
//     each part reads its batches of both operands through its own copies
//     of their views (quantization, decoding, transposes, reshapes), sliced
//     at their source;
//   - a bbacc.custom "attention" over its heads, as a batch product (its
//     bias whole in each part); of one key head, over its query rows, up to
//     four parts (each takes the keys and values whole, and its rows of the
//     bias);
//   - a bbacc.custom "softmax" or "layernorm" of [rows, w], of at least 32768
//     elements: each part takes its rows of every input of those rows;
//   - a bbacc.custom "row_operator", as large: each part takes its rows of
//     the first axis of the result that is not one (not the row; whole
//     launches of `rows` rows along their axis), so that the parts are
//     contiguous, up to `parts` of them, and of every input along it.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace ::buddy::frontend;

namespace {
// The [first, first + length) parts of `rows` rows.
SmallVector<std::pair<int64_t, int64_t>> partsOf(int64_t rows, int64_t parts,
                                                 int64_t unit) {
  int64_t units = (rows + unit - 1) / unit;
  SmallVector<std::pair<int64_t, int64_t>> result;
  for (int64_t part = 0; part < parts; ++part) {
    int64_t first = units * part / parts * unit;
    int64_t end = std::min(rows, units * (part + 1) / parts * unit);
    if (end > first)
      result.push_back({first, end - first});
  }
  return result;
}

// Rows [first, first + length) of `value` along `axis`.
Value rowsOf(OpBuilder &b, Location loc, Value value, int64_t axis,
             int64_t first, int64_t length) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0)), sizes,
      strides(type.getRank(), b.getIndexAttr(1));
  for (int64_t i = 0; i < type.getRank(); ++i)
    sizes.push_back(b.getIndexAttr(type.getDimSize(i)));
  offsets[axis] = b.getIndexAttr(first);
  sizes[axis] = b.getIndexAttr(length);
  return tensor::ExtractSliceOp::create(b, loc, value, offsets, sizes, strides);
}

// `type` with `length` along `axis`.
RankedTensorType withRows(Type type, int64_t axis, int64_t length) {
  auto ranked = cast<RankedTensorType>(type);
  SmallVector<int64_t> shape(ranked.getShape());
  shape[axis] = length;
  return RankedTensorType::get(shape, ranked.getElementType());
}

// The parts inserted along `axis` in place of `whole`: into an empty tensor,
// or into its place in the larger value it goes into (directly or through a
// reshape: a step's own keys and values after the prefix's, a copy of the
// prefix's), so the parts come back there and the whole is not copied into
// it. The larger value is there before the parts: an empty tensor, a value
// computed before them, or an empty tensor that takes another value first,
// elsewhere (the prefix's before a step's own): then the whole goes in
// first and the other after it.
Value assemble(OpBuilder &b, Location loc, Value whole, int64_t axis,
               ArrayRef<Value> parts,
               ArrayRef<std::pair<int64_t, int64_t>> ranges) {
  auto type = cast<RankedTensorType>(whole.getType());
  Operation *user = whole.hasOneUse() ? *whole.getUsers().begin() : nullptr;
  auto expand = dyn_cast_or_null<tensor::ExpandShapeOp>(user);
  if (expand && expand->hasOneUse())
    user = *expand->getUsers().begin();
  auto insert = dyn_cast_or_null<tensor::InsertSliceOp>(user);
  if (insert && insert.getSource() != (expand ? expand.getResult() : whole))
    insert = tensor::InsertSliceOp();
  Block *block = b.getInsertionBlock();
  Block::iterator point = b.getInsertionPoint();
  auto before = [&](Operation *op) {
    return op->getBlock() == block &&
           (point == block->end() || op->isBeforeInBlock(&*point));
  };
  Value destination;
  if (insert) {
    Operation *def = insert.getDest().getDefiningOp();
    auto other = dyn_cast_or_null<tensor::InsertSliceOp>(def);
    auto empty = dyn_cast_or_null<tensor::EmptyOp>(
        other ? other.getDest().getDefiningOp() : def);
    if (def && before(def)) {
      destination = insert.getDest();
    } else if (empty && !other) {
      empty->moveBefore(block, point);
      destination = insert.getDest();
    } else if (empty && other && other->hasOneUse() && empty->hasOneUse()) {
      // The two go into different rows of the empty tensor.
      bool apart = false;
      for (int64_t i = 0; i < insert.getDestType().getRank(); ++i) {
        auto first = getConstantIntValue(insert.getMixedOffsets()[i]);
        auto length = getConstantIntValue(insert.getMixedSizes()[i]);
        auto otherFirst = getConstantIntValue(other.getMixedOffsets()[i]);
        auto otherLength = getConstantIntValue(other.getMixedSizes()[i]);
        apart |= first && length && otherFirst && otherLength &&
                 (*first + *length <= *otherFirst ||
                  *otherFirst + *otherLength <= *first);
      }
      if (apart) {
        if (!before(empty))
          empty->moveBefore(block, point);
        insert.getResult().replaceAllUsesWith(other.getResult());
        insert.getDestMutable().assign(empty.getResult());
        other.getDestMutable().assign(insert.getResult());
        if (expand)
          expand->moveBefore(block, point);
        insert->moveBefore(block, point);
        b.setInsertionPoint(expand ? expand.getOperation()
                                   : insert.getOperation());
        destination = insert.getDest();
      }
    }
  }
  Value result;
  if (destination) {
    result = tensor::ExtractSliceOp::create(
        b, loc, cast<RankedTensorType>(insert.getSourceType()), destination,
        insert.getMixedOffsets(), insert.getMixedSizes(),
        insert.getMixedStrides());
    if (expand)
      result = tensor::CollapseShapeOp::create(
          b, loc, type, result, expand.getReassociationIndices());
  } else {
    result =
        tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType());
  }
  for (auto [part, range] : llvm::zip_equal(parts, ranges)) {
    SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0)), sizes,
        strides(type.getRank(), b.getIndexAttr(1));
    for (int64_t i = 0; i < type.getRank(); ++i)
      sizes.push_back(b.getIndexAttr(type.getDimSize(i)));
    offsets[axis] = b.getIndexAttr(range.first);
    sizes[axis] = b.getIndexAttr(range.second);
    // The part's place in the whole, for GVN: its consumers can read it
    // directly, the whole never built.
    SmallVector<int64_t> place(type.getRank(), 0);
    place[axis] = range.first;
    Operation *producer = part.getDefiningOp();
    producer->setLoc(
        FusedLoc::get(b.getContext(), {producer->getLoc()},
                      b.getDictionaryAttr(b.getNamedAttr(
                          "gvn.offsets", b.getDenseI64ArrayAttr(place)))));
    result = tensor::InsertSliceOp::create(b, loc, part, result, offsets, sizes,
                                           strides);
  }
  return result;
}

// The operations from a laid-out weight to the product: slice, decode,
// transpose, collapse (the nearest the product last), none if B is another.
SmallVector<Operation *> weightViews(Value value) {
  SmallVector<Operation *> chain;
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<linalg::TransposeOp, tensor::CollapseShapeOp,
             tensor::ExtractSliceOp, MXINT8ToFP32Op>(op))
      return {};
    chain.insert(chain.begin(), op);
    if (auto decode = dyn_cast<MXINT8ToFP32Op>(op)) {
      Value input = decode.getInput();
      if (auto slice = input.getDefiningOp<tensor::ExtractSliceOp>()) {
        chain.insert(chain.begin(), slice);
        input = slice.getSource();
      }
      return input.getDefiningOp<FP32ToMXINT8Op>() ? SmallVector<Operation *>()
                                                   : chain;
    }
    value = op->getOperand(0);
  }
  return {};
}
} // namespace

// `value` with `length` of its `axis` from `first`, sliced where its views
// (transposes, reshapes, the decoding of its quantization) start, the views
// cloned over the slice: the batches of a batch product's operand.
Value sliceThrough(OpBuilder &b, Location loc, Value value, int64_t axis,
                   int64_t first, int64_t length) {
  auto type = cast<RankedTensorType>(value.getType());
  auto sliced = withRows(type, axis, length);
  Operation *def = value.getDefiningOp();
  if (auto transpose = dyn_cast_or_null<linalg::TransposeOp>(def)) {
    Value input = sliceThrough(b, loc, transpose.getInput(),
                               transpose.getPermutation()[axis], first, length);
    Value init = tensor::EmptyOp::create(b, loc, sliced.getShape(),
                                         sliced.getElementType());
    return linalg::TransposeOp::create(b, loc, input, init,
                                       transpose.getPermutation())
        ->getResult(0);
  }
  if (auto collapse = dyn_cast_or_null<tensor::CollapseShapeOp>(def)) {
    // The outermost axis of the group that is not one, the axes inside it
    // whole.
    auto group = collapse.getReassociationIndices()[axis];
    auto source = collapse.getSrcType();
    size_t outer = 0;
    while (source.getDimSize(group[outer]) == 1 && outer + 1 < group.size())
      ++outer;
    int64_t inner = 1;
    for (size_t i = outer + 1; i < group.size(); ++i)
      inner *= source.getDimSize(group[i]);
    Value input;
    if (first % inner == 0 && length % inner == 0) {
      input = sliceThrough(b, loc, collapse.getSrc(), group[outer],
                           first / inner, length / inner);
    } else {
      // Rows within one index of the outer axis (some query rows of one
      // query head): that index, then the rows of the next axis.
      if (outer + 1 >= group.size() ||
          first / inner != (first + length - 1) / inner)
        llvm::report_fatal_error(
            "convert-linalg-to-hart: a batch splits a collapsed axis");
      int64_t next = inner / source.getDimSize(group[outer + 1]);
      if (first % inner % next || length % next)
        llvm::report_fatal_error(
            "convert-linalg-to-hart: a batch splits a collapsed axis");
      input = sliceThrough(b, loc, collapse.getSrc(), group[outer],
                           first / inner, 1);
      input = sliceThrough(b, loc, input, group[outer + 1],
                           first % inner / next, length / next);
    }
    return tensor::CollapseShapeOp::create(b, loc, sliced, input,
                                           collapse.getReassociationIndices());
  }
  if (auto expand = dyn_cast_or_null<tensor::ExpandShapeOp>(def)) {
    // The axis of the source whose group holds `axis`, the axes after it in
    // the group whole and the ones before it one.
    auto groups = expand.getReassociationIndices();
    int64_t source = 0, inner = 1;
    for (auto [index, group] : llvm::enumerate(groups))
      if (llvm::is_contained(group, axis)) {
        source = index;
        for (int64_t i : group) {
          if (i < axis && type.getDimSize(i) != 1)
            llvm::report_fatal_error(
                "convert-linalg-to-hart: a batch splits an expanded axis");
          if (i > axis)
            inner *= type.getDimSize(i);
        }
      }
    Value input = sliceThrough(b, loc, expand.getSrc(), source, first * inner,
                               length * inner);
    return tensor::ExpandShapeOp::create(b, loc, sliced, input, groups);
  }
  // The decoding of a quantization: the quantized values sliced, and
  // quantized again; of codes quantized elsewhere, their slice. Along the
  // quantization axis the slice is whole blocks of 32 values, each block 33
  // codes (its values and its exponent).
  if (auto decode = dyn_cast_or_null<MXINT8ToFP32Op>(def)) {
    auto quant = decode.getInput().getDefiningOp<FP32ToMXINT8Op>();
    int64_t quantAxis = decode.getAxis();
    if (quantAxis == axis && (!quant || first % 32 || length % 32))
      llvm::report_fatal_error(
          "convert-linalg-to-hart: a batch splits a quantization block");
    // Codes quantized elsewhere (an argument): their slice, decoded.
    if (!quant) {
      Value codes =
          sliceThrough(b, loc, decode.getInput(), axis, first, length);
      Operation *values = b.clone(*decode);
      values->setOperand(0, codes);
      values->getResult(0).setType(sliced);
      return values->getResult(0);
    }
    Value input = sliceThrough(b, loc, quant.getInput(), axis, first, length);
    Operation *codes = b.clone(*quant);
    codes->setOperand(0, input);
    codes->getResult(0).setType(
        withRows(quant.getResult().getType(), axis,
                 quantAxis == axis ? length / 32 * 33 : length));
    Operation *values = b.clone(*decode);
    values->setOperand(0, codes->getResult(0));
    values->getResult(0).setType(sliced);
    return values->getResult(0);
  }
  // Parts inserted along `axis` (assemble): the part of these rows.
  for (auto insert = dyn_cast_or_null<tensor::InsertSliceOp>(def); insert;
       insert = insert.getDest().getDefiningOp<tensor::InsertSliceOp>()) {
    auto at = getConstantIntValue(insert.getMixedOffsets()[axis]);
    auto size = getConstantIntValue(insert.getMixedSizes()[axis]);
    if (insert.getSourceType() == sliced && at == first && size == length)
      return insert.getSource();
  }
  return rowsOf(b, loc, value, axis, first, length);
}

// `value` through its own copies of its views, decoding and quantization.
Value copyThrough(OpBuilder &b, Location loc, Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || !isa<linalg::TransposeOp, tensor::CollapseShapeOp,
                   tensor::ExpandShapeOp, MXINT8ToFP32Op, FP32ToMXINT8Op>(def))
    return value;
  Value input = copyThrough(b, loc, def->getOperand(0));
  Operation *copy = b.clone(*def);
  copy->setOperand(0, input);
  if (auto transpose = dyn_cast<linalg::TransposeOp>(copy)) {
    auto type = cast<RankedTensorType>(transpose.getInit().getType());
    copy->setOperand(1, tensor::EmptyOp::create(b, loc, type.getShape(),
                                                type.getElementType()));
  }
  return copy->getResult(0);
}

void splitRows(ModuleOp module, int64_t parts) {
  if (parts < 2)
    return;
  // An empty tensor of its own for every value built in one (CSE merges
  // them: the caches of all layers), so that each can take its parts in
  // place (assemble).
  SmallVector<tensor::EmptyOp> empties;
  module.walk([&](tensor::EmptyOp empty) { empties.push_back(empty); });
  for (tensor::EmptyOp empty : empties) {
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : empty->getUses())
      uses.push_back(&use);
    for (size_t i = 1; i < uses.size(); ++i) {
      OpBuilder b(uses[i]->getOwner());
      uses[i]->set(b.clone(*empty)->getResult(0));
    }
  }
  // Products of one row that share its quantization (the projections of one
  // condition): each quantizes the row for itself, so that they do not join
  // through it and the threads take them apart.
  SmallVector<MXINT8ToFP32Op> rows;
  module.walk([&](MXINT8ToFP32Op decode) {
    auto type = dyn_cast<RankedTensorType>(decode->getResult(0).getType());
    if (!decode.getInput().getDefiningOp<FP32ToMXINT8Op>() ||
        !decode.getInput().hasOneUse() || !type || type.getRank() != 2 ||
        type.getDimSize(0) != 1 || decode->getResult(0).hasOneUse())
      return;
    for (OpOperand &use : decode->getResult(0).getUses())
      if (!isa<linalg::MatmulOp>(use.getOwner()) || use.getOperandNumber() != 0)
        return;
    rows.push_back(decode);
  });
  for (MXINT8ToFP32Op decode : rows) {
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : decode->getResult(0).getUses())
      uses.push_back(&use);
    for (size_t i = 1; i < uses.size(); ++i) {
      OpBuilder b(uses[i]->getOwner());
      Operation *codes = b.clone(*decode.getInput().getDefiningOp());
      Operation *values = b.clone(*decode);
      values->setOperand(0, codes->getResult(0));
      uses[i]->set(values->getResult(0));
    }
  }
  // Products, from the last term of each chain.
  SmallVector<linalg::MatmulOp> products;
  module.walk([&](linalg::MatmulOp op) {
    bool later = op->hasOneUse() &&
                 isa<linalg::MatmulOp>(*op->getUsers().begin()) &&
                 op->getUses().begin()->getOperandNumber() == 2;
    if (!later)
      products.push_back(op);
  });
  int64_t unit = target::getBallParam("MXINT8MM", "rows");
  for (linalg::MatmulOp last : products) {
    SmallVector<linalg::MatmulOp> terms{last};
    while (auto earlier =
               terms.front().getOutputs()[0].getDefiningOp<linalg::MatmulOp>())
      terms.insert(terms.begin(), earlier);
    Value a = last.getInputs()[0];
    auto type = cast<RankedTensorType>(last.getResult(0).getType());
    int64_t m = type.getDimSize(0), n = type.getDimSize(1);
    // Every part moves the weight from the DDR once, and the harts share
    // the DDR: p parts pay off while their p weights (N K / 16 cycles each)
    // take no longer than the rows of a part in the array (M / p rows, N K
    // / 16 cycles every `unit` rows), p * p <= M / unit.
    int64_t count = 1;
    while (count < parts && (count + 1) * (count + 1) * unit <= m)
      ++count;
    // A product of several K pieces (K in pieces of up to 1024) keeps a
    // group of A tiles in the thread's banks, every piece of them, while the
    // weight passes once (heldGroup of convert-linalg-to-acc: the pieces of
    // the tiles, a B tile a term and the tiles' C within the banks but two;
    // a tile the rows a bank takes). A part of more rows reads the weight
    // again for each further group: more parts where that makes the rows of
    // a part one group, as far as there are threads.
    int64_t codes =
        llvm::alignTo(cast<RankedTensorType>(a.getType()).getDimSize(1), 32);
    int64_t piece = 1024;
    while (codes % piece)
      piece -= 32;
    int64_t pieces = codes / piece;
    int64_t tiles = (target::getTarget().bankNum - 2 - int64_t(terms.size())) /
                    (pieces + 1);
    int64_t held = tiles * (target::getTarget().bankDepth /
                            (piece / 16 + (piece + 511) / 512));
    if (pieces > 1 && held >= unit) {
      int64_t units = (m + unit - 1) / unit;
      while (count < parts && (units + count - 1) / count * unit > held)
        ++count;
    }
    if (count < 2 || llvm::any_of(
                         terms,
                         [&](linalg::MatmulOp t) {
                           return t.getInputs()[0] != a ||
                                  weightViews(t.getInputs()[1]).empty();
                         }))
      continue;
    // A: decode(quant(x)) of x [M, K], or collapsed from x [1, M, K].
    auto collapse = a.getDefiningOp<tensor::CollapseShapeOp>();
    auto decode =
        (collapse ? collapse.getSrc() : a).getDefiningOp<MXINT8ToFP32Op>();
    auto quant = decode ? decode.getInput().getDefiningOp<FP32ToMXINT8Op>()
                        : FP32ToMXINT8Op();
    if (!quant)
      continue;
    auto xType = cast<RankedTensorType>(quant.getInput().getType());
    int64_t axis = xType.getRank() - 2;
    if (xType.getDimSize(axis) != m ||
        int64_t(quant.getAxis()) != xType.getRank() - 1 ||
        (xType.getRank() == 3 && xType.getDimSize(0) != 1) ||
        xType.getRank() > 3 || (xType.getRank() == 3) != bool(collapse))
      continue;
    // The start of the head term: zero, a broadcast bias, a residual, or both.
    Value init = terms.front().getOutputs()[0];
    auto fill = init.getDefiningOp<linalg::FillOp>();
    auto start = init.getDefiningOp<linalg::BroadcastOp>();
    auto both = init.getDefiningOp<linalg::GenericOp>();
    if (both && !(both.getNumDpsInputs() == 2 &&
                  both.getDpsInputs()[1].getDefiningOp<linalg::BroadcastOp>()))
      continue;
    OpBuilder b(last);
    Location loc = last.getLoc();
    auto ranges = partsOf(m, count, unit);
    SmallVector<Value> results;
    for (auto [first, length] : ranges) {
      auto partType = RankedTensorType::get({length, n}, type.getElementType());
      IRMapping mapping;
      Value x = rowsOf(b, loc, quant.getInput(), axis, first, length);
      Operation *codes = b.clone(*quant);
      codes->setOperand(0, x);
      codes->getResult(0).setType(
          withRows(quant->getResult(0).getType(), axis, length));
      Operation *values = b.clone(*decode);
      values->setOperand(0, codes->getResult(0));
      values->getResult(0).setType(
          withRows(decode->getResult(0).getType(), axis, length));
      Value rowsA = collapse ? Value(tensor::CollapseShapeOp::create(
                                   b, loc, values->getResult(0),
                                   collapse.getReassociationIndices()))
                             : values->getResult(0);
      // The start of the part.
      auto zero = [&](Value pattern) -> Value {
        Value empty = tensor::EmptyOp::create(b, loc, partType.getShape(),
                                              partType.getElementType());
        return linalg::FillOp::create(b, loc, pattern, empty).getResult(0);
      };
      auto bias = [&](linalg::BroadcastOp broadcast) -> Value {
        Value into = zero(arith::ConstantOp::create(
            b, loc, b.getZeroAttr(partType.getElementType())));
        return linalg::BroadcastOp::create(b, loc, broadcast.getInput(), into,
                                           broadcast.getDimensions())
            ->getResult(0);
      };
      Value partStart;
      if (fill) {
        partStart = zero(fill.getInputs()[0]);
      } else if (start) {
        partStart = bias(start);
      } else if (both) {
        Value residual =
            rowsOf(b, loc, both.getDpsInputs()[0], 0, first, length);
        Value broadcast =
            bias(both.getDpsInputs()[1].getDefiningOp<linalg::BroadcastOp>());
        IRMapping sum;
        sum.map(both.getDpsInputs()[0], residual);
        sum.map(both.getDpsInputs()[1], broadcast);
        sum.map(both.getDpsInits()[0],
                zero(arith::ConstantOp::create(
                    b, loc, b.getZeroAttr(partType.getElementType()))));
        Operation *added = b.clone(*both, sum);
        added->getResult(0).setType(partType);
        partStart = added->getResult(0);
      } else {
        partStart = rowsOf(b, loc, init, 0, first, length);
      }
      // The terms, each reading its own copy of its weight views.
      Value previous = partStart;
      for (linalg::MatmulOp term : terms) {
        IRMapping views;
        for (Operation *op : weightViews(term.getInputs()[1]))
          b.clone(*op, views);
        Value weight = views.lookup(term.getInputs()[1]);
        IRMapping product;
        product.map(term.getInputs()[0], rowsA);
        product.map(term.getInputs()[1], weight);
        product.map(term.getOutputs()[0], previous);
        Operation *cloned = b.clone(*term, product);
        cloned->getResult(0).setType(partType);
        previous = cloned->getResult(0);
      }
      results.push_back(previous);
    }
    last.getResult(0).replaceAllUsesWith(
        assemble(b, loc, last.getResult(0), 0, results, ranges));
    for (linalg::MatmulOp term : llvm::reverse(terms))
      term->erase();
    for (Operation *op :
         {init.getDefiningOp(), collapse ? collapse.getOperation() : nullptr,
          decode.getOperation(), quant.getOperation()})
      if (op && op->use_empty())
        op->erase();
  }

  // Softmax, layer normalization and row operators.
  SmallVector<::buddy::acc::CustomOp> customs;
  module.walk([&](::buddy::acc::CustomOp op) {
    if (op.getKind() == "softmax" || op.getKind() == "layernorm" ||
        op.getKind() == "row_operator")
      customs.push_back(op);
  });
  for (auto op : customs) {
    auto type = cast<RankedTensorType>(op.getResult(0).getType());
    // Small work goes whole to one thread: its parts would cost more
    // launches and their assembly (a copy, when the whole goes into a larger
    // value: a step's own keys into its cache) than their rows.
    if (type.getNumElements() < 32768)
      continue;
    int64_t rank = type.getRank(), axis = -1;
    bool rows = op.getKind() == "row_operator";
    // Along the axis of a row operator's launches the parts are whole
    // launches (`rows` rows each).
    int64_t launch = rows ? op->getAttrOfType<IntegerAttr>("rows").getInt() : 1;
    // The parts are contiguous in the whole: along its first axis that is
    // not one (not the row), as many as it has units up to `parts`.
    int64_t unit = 1;
    for (int64_t i = 0; i + 1 < rank && axis < 0; ++i)
      if (type.getDimSize(i) > 1)
        axis = i;
    if (axis < 0)
      continue;
    unit = launch > 1 && axis == rank - 2 ? launch : 1;
    int64_t count = std::min(parts, type.getDimSize(axis) / unit);
    if (count < 2 || type.getDimSize(axis) % unit)
      continue;
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto ranges = partsOf(type.getDimSize(axis), count, unit);
    SmallVector<AffineMap> maps;
    if (rows)
      for (Attribute map :
           op->getAttrOfType<ArrayAttr>("maps").getValue().drop_back())
        maps.push_back(cast<AffineMapAttr>(map).getValue());
    SmallVector<Value> results;
    for (auto [first, length] : ranges) {
      SmallVector<Value> inputs;
      for (auto [i, input] : llvm::enumerate(op.getInputs())) {
        auto inputType = cast<RankedTensorType>(input.getType());
        // The input axis along the split one, if any.
        int64_t along = -1;
        if (rows) {
          for (auto [j, e] : llvm::enumerate(maps[i].getResults()))
            if (e == getAffineDimExpr(axis, op.getContext()))
              along = j;
        } else if (inputType.getRank() == rank &&
                   inputType.getDimSize(0) == type.getDimSize(0)) {
          along = 0;
        }
        inputs.push_back(
            along < 0 ? input : rowsOf(b, loc, input, along, first, length));
      }
      auto part = ::buddy::acc::CustomOp::create(
          b, loc, TypeRange{withRows(type, axis, length)}, op.getKind(),
          inputs);
      part->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      results.push_back(part.getResult(0));
    }
    op.getResult(0).replaceAllUsesWith(
        assemble(b, loc, op.getResult(0), axis, results, ranges));
    op.erase();
  }

  // Batch products (the heads of an attention to its tokens): their batches
  // over the threads, each part reading its batches of both operands through
  // its own copies of their views.
  SmallVector<linalg::BatchMatmulOp> batches;
  module.walk([&](linalg::BatchMatmulOp op) { batches.push_back(op); });
  for (linalg::BatchMatmulOp op : batches) {
    auto type = cast<RankedTensorType>(op.getResult(0).getType());
    int64_t count = std::min(parts, type.getDimSize(0));
    auto fill = op.getOutputs()[0].getDefiningOp<linalg::FillOp>();
    if (count < 2 || !fill)
      continue;
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto ranges = partsOf(type.getDimSize(0), count, 1);
    SmallVector<Value> results;
    for (auto [first, length] : ranges) {
      auto partType = withRows(type, 0, length);
      Value lhs = sliceThrough(b, loc, op.getInputs()[0], 0, first, length);
      Value rhs = sliceThrough(b, loc, op.getInputs()[1], 0, first, length);
      Value empty = tensor::EmptyOp::create(b, loc, partType.getShape(),
                                            partType.getElementType());
      Value start = linalg::FillOp::create(b, loc, fill.getInputs()[0], empty)
                        .getResult(0);
      results.push_back(
          linalg::BatchMatmulOp::create(b, loc, TypeRange{partType},
                                        ValueRange{lhs, rhs}, ValueRange{start})
              ->getResult(0));
    }
    op.getResult(0).replaceAllUsesWith(
        assemble(b, loc, op.getResult(0), 0, results, ranges));
    op.erase();
  }

  // Attentions (OutlineAttention.cpp): their heads over the threads, each
  // part reading its heads of q, k and v through its own copies of their
  // views, as a batch product.
  SmallVector<::buddy::acc::CustomOp> attentions;
  module.walk([&](::buddy::acc::CustomOp op) {
    if (op.getKind() == "attention")
      attentions.push_back(op);
  });
  for (auto op : attentions) {
    auto type = cast<RankedTensorType>(op.getResult(0).getType());
    if (type.getDimSize(0) == 1) {
      // One key head: its query rows over up to four threads (each part
      // packs the keys and values for itself), each part rows of one group
      // (the query heads of the key head are groups of rows q's collapse
      // merges) with the bias rows of those rows, the keys and values whole.
      OpBuilder b(op);
      Location loc = op.getLoc();
      bool masked = op->hasAttr("masked");
      int64_t rows = type.getDimSize(1), perGroup = rows;
      for (Operation *def = op.getInputs()[0].getDefiningOp();
           def &&
           isa<MXINT8ToFP32Op, FP32ToMXINT8Op, tensor::CollapseShapeOp>(def);
           def = def->getOperand(0).getDefiningOp())
        if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(def)) {
          auto group = collapse.getReassociationIndices()[1];
          auto source = collapse.getSrcType();
          size_t outer = 0;
          while (source.getDimSize(group[outer]) == 1 &&
                 outer + 1 < group.size())
            ++outer;
          perGroup = 1;
          for (size_t i = outer + 1; i < group.size(); ++i)
            perGroup *= source.getDimSize(group[i]);
          break;
        }
      int64_t chunks = std::min<int64_t>(parts, 4) / (rows / perGroup);
      if (chunks * (rows / perGroup) < 2)
        continue;
      SmallVector<std::pair<int64_t, int64_t>> ranges;
      SmallVector<Value> results;
      for (int64_t group = 0; group < rows / perGroup; ++group)
        for (auto [first, length] :
             partsOf(perGroup, std::max<int64_t>(chunks, 1), 1)) {
          SmallVector<Value> inputs;
          inputs.push_back(sliceThrough(b, loc, op.getInputs()[0], 1,
                                        group * perGroup + first, length));
          for (Value input : op.getInputs().drop_front().drop_back(masked))
            inputs.push_back(copyThrough(b, loc, input));
          if (masked) {
            Value bias = op.getInputs().back();
            int64_t period =
                cast<RankedTensorType>(bias.getType()).getDimSize(0);
            // Query row r takes bias row r mod period: a part within one
            // period takes its rows of the bias.
            if (first % period || length % period) {
              if (first / period != (first + length - 1) / period)
                llvm::report_fatal_error(
                    "convert-linalg-to-hart: a part splits the rows of an "
                    "attention's bias");
              bias = rowsOf(b, loc, bias, 0, first % period, length);
            }
            inputs.push_back(bias);
          }
          auto part = ::buddy::acc::CustomOp::create(
              b, loc, TypeRange{withRows(type, 1, length)}, "attention",
              inputs);
          part->setDiscardableAttrs(op->getDiscardableAttrDictionary());
          results.push_back(part.getResult(0));
          ranges.push_back({group * perGroup + first, length});
        }
      op.getResult(0).replaceAllUsesWith(
          assemble(b, loc, op.getResult(0), 1, results, ranges));
      op.erase();
      continue;
    }
    int64_t count = std::min(parts, type.getDimSize(0));
    if (count < 2)
      continue;
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto ranges = partsOf(type.getDimSize(0), count, 1);
    SmallVector<Value> results;
    for (auto [first, length] : ranges) {
      // The bias is that of every head.
      SmallVector<Value> inputs;
      for (Value input : op.getInputs().drop_back(op->hasAttr("masked")))
        inputs.push_back(sliceThrough(b, loc, input, 0, first, length));
      if (op->hasAttr("masked"))
        inputs.push_back(op.getInputs().back());
      auto part = ::buddy::acc::CustomOp::create(
          b, loc, TypeRange{withRows(type, 0, length)}, "attention", inputs);
      part->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      results.push_back(part.getResult(0));
    }
    op.getResult(0).replaceAllUsesWith(
        assemble(b, loc, op.getResult(0), 0, results, ranges));
    op.erase();
  }

  // The whole operands' views, decodings and quantizations no one reads now.
  bool erased = true;
  while (erased) {
    erased = false;
    module.walk([&](Operation *op) {
      if (op->use_empty() && op->getNumResults() &&
          (isOpTriviallyDead(op) || isa<MXINT8ToFP32Op, FP32ToMXINT8Op>(op))) {
        op->erase();
        erased = true;
      }
    });
  }
}
