//===- ConvertLinalgToAcc.cpp - Piplus MXINT8 computations on banks -------===//
//
// convert-linalg-to-acc maps the MXINT8 products of a thread subgraph to
// tiles of mxint8mm and the online quantization of their operands to
// mxint8quant.
//
// K is split in blocks of kc values (a multiple of 32 dividing K). One row of
// a block is packed in S = kc/16 + ceil(kc/512) bank rows: the kc codes, the
// kc/32 exponent bytes, zero padding. An operand is held in memory as
// [blocks][rows][S * 16 bytes] (behind a batch axis for batch products), so
// that the rows of a block are contiguous and DMA moves a tile of them in
// place.
//
//   - A weight is laid out so offline by the recipe: [blocks][N][S * 16],
//     read in the graph through a slice dropping the padding, mxint8_to_fp32
//     along axis 2, a transpose [0, 2, 1] and a collapse to [K, N]. The
//     weight is DMA'd as it is; there is no copy of it on the CPU.
//   - An operand quantized online (the activations, both operands of the
//     attention products) is quantized by mxint8quant one row of a block at
//     a time into a buffer of that layout, its rows padded with zero rows
//     to whole tiles. An operand whose rows are the contiguous axis in
//     memory (V of attention, [K][N]) moves by strided DMA in [kc][4]
//     column slices, which mxint8quant quantizes with columns into four
//     packed rows each.
//
// A product computes Mt x Nt tiles of C, accumulating over the K blocks in
// its destination bank, within the limits of mxint8mm on the target's banks
// of D rows: Mt * S <= D, Nt * S <= D, Mt * Nt / 4 <= D, Nt a multiple of 4.
// Tiles go in groups of up to 8 A tiles, each with its own C bank. With a
// single K block the group's A tiles stay in their banks while every B tile
// of the row passes once; with several, for each B tile column the group
// accumulates over the K blocks, each block's B tiles moved once for the
// group and the A tiles once a block.
//
// C is row-major Mt x Nt in its bank. Each tile writes the rows and columns
// of it within [M, N] into the destination: the whole tile by mvout when it
// spans whole rows (Nt = N); else by strided mvout, in column chunks of
// blocks of up to sixteen 16-byte rows when a row of the destination is
// within the 4095-byte stride of the strided DMA or whole 16-byte rows (the
// block DMA), or one tile row at a time, the bytes after its last whole
// 16-byte row apart. The last tiles of M and N are
// apart, so no tile writes past M or N.
//
// The row_operators of convert-linalg-to-hart launch their operators on the GP
// domain (LowerRowOperators.cpp). The pass numbers the operators it launches
// and adds their entry `operator`, which runs the operator numbered by a7; the
// operators and the entry are built into the operator image by the recipe.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "layernormOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mxint8mmOps.h"
#include "mxint8quantOps.h"
#include "softmaxOps.h"
#include <cmath>
#include <map>
#include <tuple>

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

LogicalResult lowerRowOperators(ModuleOp module,
                                function_ref<int64_t(StringAttr)> number);
LogicalResult lowerSoftmax(ModuleOp module);
LogicalResult lowerLayerNorm(ModuleOp module);
LogicalResult lowerAttention(ModuleOp module);

// The memref of a tensor: an argument of a subgraph that the hart passes as
// a slice of a larger value (convert-linalg-to-hart) keeps that value's
// strides, which the rows of an operator, a softmax or a layer
// normalization move by.
MemRefType bufferType(Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  MemRefLayoutAttrInterface layout;
  if (auto argument = dyn_cast<BlockArgument>(tensor))
    if (auto function =
            dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp()))
      layout = function.getArgAttrOfType<MemRefLayoutAttrInterface>(
          argument.getArgNumber(), "bufferization.buffer_layout");
  return MemRefType::get(type.getShape(), type.getElementType(), layout);
}

// The strides, over the buffer of `source` (bufferType), of the value its
// views (transposes and reshapes, nearest the product first) make of it;
// none when a reshape merges axes that are not contiguous.
SmallVector<int64_t> viewStrides(Value source, ArrayRef<Operation *> views);

namespace {
Value index(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(b, loc, value);
}

// Walk layout-only operations back to the packed value. The same chain is
// used below to map product coordinates to the original tensor coordinates.
struct Operand {
  Value value;
  SmallVector<Operation *> views;
  MXINT8ToFP32Op decode;
  FP32ToMXINT8Op quant;
};
Operand operand(Value value) {
  Operand result{value, {}, {}, {}};
  while (Operation *op = value.getDefiningOp()) {
    if (auto decode = dyn_cast<MXINT8ToFP32Op>(op)) {
      result.decode = decode;
      result.quant = decode.getInput().getDefiningOp<FP32ToMXINT8Op>();
      return result;
    }
    if (!isa<linalg::TransposeOp, tensor::CollapseShapeOp,
             tensor::ExpandShapeOp>(op))
      break;
    result.views.push_back(op);
    value = op->getOperand(0);
  }
  return result;
}

// Build a loop nest, keeping large model dimensions in the IR as loops.
void loops(OpBuilder &b, Location loc, ArrayRef<int64_t> sizes, bool stream,
           function_ref<void(ValueRange)> body) {
  OpBuilder::InsertionGuard guard(b);
  SmallVector<Value> ivs;
  for (int64_t size : sizes) {
    auto loop = scf::ForOp::create(b, loc, index(b, loc, 0),
                                   index(b, loc, size), index(b, loc, 1));
    if (stream)
      loop->setAttr("hart.stream", b.getUnitAttr());
    ivs.push_back(loop.getInductionVar());
    b.setInsertionPointToStart(loop.getBody());
  }
  body(ivs);
}

} // namespace

SmallVector<int64_t> viewStrides(Value source, ArrayRef<Operation *> views) {
  MemRefType memory = bufferType(source);
  SmallVector<int64_t> shape(memory.getShape()),
      strides(memory.getStridesAndOffset().first);
  for (Operation *view : llvm::reverse(views)) {
    auto result = cast<RankedTensorType>(view->getResult(0).getType());
    SmallVector<int64_t> next(result.getRank());
    if (auto transpose = dyn_cast<linalg::TransposeOp>(view)) {
      for (auto [axis, from] : llvm::enumerate(transpose.getPermutation()))
        next[axis] = strides[from];
    } else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(view)) {
      for (auto [from, group] :
           llvm::enumerate(expand.getReassociationIndices())) {
        int64_t inner = strides[from];
        for (int64_t axis : llvm::reverse(group)) {
          next[axis] = inner;
          inner *= result.getDimSize(axis);
        }
      }
    } else {
      auto collapse = cast<tensor::CollapseShapeOp>(view);
      // A group collapses contiguously; its unit axes place nothing.
      for (auto [to, group] :
           llvm::enumerate(collapse.getReassociationIndices())) {
        SmallVector<int64_t> kept;
        for (int64_t axis : group)
          if (shape[axis] != 1)
            kept.push_back(axis);
        for (int64_t i = 0; i + 1 < int64_t(kept.size()); ++i)
          if (strides[kept[i]] != strides[kept[i + 1]] * shape[kept[i + 1]])
            return {};
        next[to] = kept.empty() ? strides[group.back()] : strides[kept.back()];
      }
    }
    shape.assign(result.getShape().begin(), result.getShape().end());
    strides = next;
  }
  return strides;
}

namespace {
Value buffer(OpBuilder &b, Location loc, Value tensor) {
  return bufferization::ToBufferOp::create(b, loc, bufferType(tensor), tensor);
}

// The offset of `memory`'s first element from its base.
Value offsetOf(OpBuilder &b, Location loc, Value memory) {
  return memref::ExtractStridedMetadataOp::create(b, loc, memory).getOffset();
}

// `count` rows of `width` elements from row `row` of the array at the outer
// coordinates `outer` of `memory`.
Value rows(OpBuilder &b, Location loc, Value memory, ValueRange outer,
           OpFoldResult row, int64_t count, int64_t width) {
  SmallVector<OpFoldResult> at, size,
      stride(outer.size() + 2, b.getIndexAttr(1));
  for (Value value : outer) {
    at.push_back(value);
    size.push_back(b.getIndexAttr(1));
  }
  at.append({row, b.getIndexAttr(0)});
  size.append({b.getIndexAttr(count), b.getIndexAttr(width)});
  return memref::SubViewOp::create(b, loc, memory, at, size, stride);
}

// The bank rows of one packed row of a block of kc values.
int64_t packedRows(int64_t kc) { return kc / 16 + (kc + 511) / 512; }

// `count` split in tiles of at most `limit`, as even as possible.
int64_t evenTile(int64_t count, int64_t limit) {
  int64_t tiles = (count + limit - 1) / limit;
  return (count + tiles - 1) / tiles;
}

class ConvertLinalgToAccPass
    : public PassWrapper<ConvertLinalgToAccPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToAccPass)
  StringRef getArgument() const final { return "convert-linalg-to-acc"; }
  StringRef getDescription() const final {
    return "Map piplus MXINT8 products and their quantization to banks";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<AccDialect, arith::ArithDialect,
             bufferization::BufferizationDialect, linalg::LinalgDialect,
             memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
             ::buddy::mxint8mm_ball::Mxint8mmBallDialect,
             ::buddy::mxint8quant_ball::Mxint8quantBallDialect,
             ::buddy::softmax_ball::SoftmaxBallDialect,
             ::buddy::layernorm_ball::LayerNormBallDialect>();
  }
  void runOnOperation() override {
    const auto &target = target::getTarget();
    int64_t depth = target.bankDepth;
    // A tile of A: the most rows, in whole passes of the array, that keep a
    // square tile of C within the bank (Mt * Nt / 4 <= D).
    int64_t arrayRows = target::getBallParam("MXINT8MM", "rows");
    int64_t side = int64_t(std::sqrt(double(4 * depth)));
    int64_t tileRows = side / arrayRows * arrayRows;
    if (target.bankWidthBits != 128 || target.bankNum < 4 || tileRows < 1) {
      getOperation().emitError(
          "piplus needs 128-bit banks, four banks and a tile of the array");
      return signalPassFailure();
    }
    SmallVector<Operation *> products;
    getOperation().walk([&](Operation *op) {
      if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op) &&
          operand(op->getOperand(0)).decode &&
          operand(op->getOperand(1)).decode)
        products.push_back(op);
    });
    // A product whose start is another product of the same A (the two
    // MXINT8 terms of a weight, convert-linalg-to-hart's FuseBias.cpp): the
    // terms of a chain go to the same C tiles, each K block of each term
    // accumulating onto the tile.
    DenseSet<Operation *> productSet(products.begin(), products.end());
    auto earlier = [&](Operation *product) -> Operation * {
      // A product of another A is a residual start (written first).
      Operation *def = product->getOperand(2).getDefiningOp();
      return def && productSet.contains(def) &&
                     def->getOperand(0) == product->getOperand(0)
                 ? def
                 : nullptr;
    };
    // The quantized operands, by value, side and padded rows: the q, k and
    // v projections of a layer quantize their input once.
    std::map<std::tuple<void *, bool, int64_t>, Value> quantized;
    for (Operation *op : products) {
      if (op->hasOneUse() && productSet.contains(*op->getUsers().begin()) &&
          earlier(*op->getUsers().begin()) == op)
        continue;
      SmallVector<Operation *> terms{op};
      while (Operation *term = earlier(terms.front()))
        terms.insert(terms.begin(), term);
      Operation *head = terms.front();
      OpBuilder b(op);
      Location loc = op->getLoc();
      Type i64 = b.getI64Type();
      Operand a = operand(op->getOperand(0)), w = operand(op->getOperand(1));
      auto outputType = cast<RankedTensorType>(op->getResult(0).getType());
      bool batched = isa<linalg::BatchMatmulOp>(op);
      int64_t batch = batched ? outputType.getDimSize(0) : 1;
      int64_t m = outputType.getDimSize(batched ? 1 : 0);
      int64_t n = outputType.getDimSize(batched ? 2 : 1);
      int64_t k = cast<RankedTensorType>(a.value.getType()).getShape().back();
      // The start of the sums (convert-linalg-to-hart's FuseBias.cpp): zero,
      // a bias row broadcast along the rows (each element starts at
      // bias[col]), a residual R (each C tile is loaded from it) or R and
      // the bias.
      Value init = head->getOperand(2), residual;
      auto residualSum = init.getDefiningOp<linalg::GenericOp>();
      if (residualSum && residualSum.getNumDpsInputs() == 2 &&
          residualSum.getDpsInputs()[1].getDefiningOp<linalg::BroadcastOp>()) {
        residual = residualSum.getDpsInputs()[0];
        init = residualSum.getDpsInputs()[1];
      } else {
        residualSum = {};
      }
      auto fill = init.getDefiningOp<linalg::FillOp>();
      bool zero =
          matchPattern(fill ? fill.getInputs()[0] : init, m_PosZeroFloat());
      auto start = init.getDefiningOp<linalg::BroadcastOp>();
      Value bias = start && !batched && n % 4 == 0 &&
                           start.getDimensions() == ArrayRef<int64_t>{0}
                       ? start.getInput()
                       : Value();
      if (!zero && !bias && !residual && !batched && n % 4 == 0)
        residual = init;
      // Or A in packed MXINT8 rows already (a softmax with mxint8): K one
      // block, a multiple of 512, so that a row of the tensor is a packed
      // row; DMA reads its rows in place.
      bool packedA = !a.quant && a.decode && k % 512 == 0 && k <= 1024;
      if (!outputType.hasStaticShape() || k <= 0 || k % 32 ||
          !(zero || bias || residual) || !(a.quant || packedA) ||
          (!w.quant && n % 4)) {
        op->emitError("piplus MXINT8 product needs static shapes, K multiple "
                      "of 32, a zero, bias or residual start, an online "
                      "quantized A and a weight N multiple of 4");
        return signalPassFailure();
      }
      auto users = op->getResult(0).getUsers();
      auto destination =
          op->getResult(0).hasOneUse()
              ? dyn_cast<bufferization::MaterializeInDestinationOp>(
                    *users.begin())
              : bufferization::MaterializeInDestinationOp();

      // B: a weight in the layout of the recipe, or quantized online. The K
      // block of an online pair is the recipe's: the largest multiple of 32
      // dividing K up to 1024 (frontend/quant/layout.py).
      // The terms of a chain read the same A, each its laid-out weight.
      for (Operation *term : terms)
        if (term != op && (operand(term->getOperand(0)).value != a.value ||
                           operand(term->getOperand(1)).quant || w.quant)) {
          term->emitError("piplus product terms must read the same A and "
                          "laid-out weights");
          return signalPassFailure();
        }
      SmallVector<Value> weights;
      Value weight;
      int64_t kc = 1024;
      while (k % kc)
        kc -= 32;
      for (Operation *term : terms) {
        if (w.quant)
          break;
        Operand w = operand(term->getOperand(1));
        // The laid-out parameter: read through a slice dropping the padding
        // of each packed row, or directly when a row has none.
        auto slice =
            w.decode.getInput().getDefiningOp<tensor::ExtractSliceOp>();
        Value parameter = slice ? slice.getSource() : w.decode.getInput();
        auto layout = cast<RankedTensorType>(parameter.getType());
        auto read = cast<RankedTensorType>(w.decode.getInput().getType());
        auto collapse = w.views.size() == 2
                            ? dyn_cast<tensor::CollapseShapeOp>(w.views[0])
                            : tensor::CollapseShapeOp();
        auto transpose = w.views.size() == 2
                             ? dyn_cast<linalg::TransposeOp>(w.views[1])
                             : linalg::TransposeOp();
        bool laid =
            !batched && w.decode.getAxis() == 2 && layout.getRank() == 3 &&
            layout.hasStaticShape() && layout.getDimSize(1) == n &&
            k % layout.getDimSize(0) == 0 && collapse && transpose &&
            transpose.getPermutation() == ArrayRef<int64_t>{0, 2, 1} &&
            (!slice || (slice.hasZeroOffset() && slice.hasUnitStride()));
        if (laid) {
          kc = k / layout.getDimSize(0);
          laid = kc % 32 == 0 && layout.getDimSize(2) == packedRows(kc) * 16 &&
                 read.getShape() ==
                     ArrayRef<int64_t>{layout.getDimSize(0), n, kc + kc / 32};
        }
        if (!laid) {
          op->emitError("piplus weight must be laid out [blocks][N][packed "
                        "block] and read through slice, mxint8_to_fp32 along "
                        "axis 2, transpose [0, 2, 1] and collapse to [K, N]");
          return signalPassFailure();
        }
        weight = buffer(b, loc, parameter);
        weight.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
        weights.push_back(weight);
      }
      int64_t blocks = k / kc, packed = packedRows(kc), width = packed * 16;

      // Tiles within mxint8mm: Mt * S, Nt * S and Mt * Nt / 4 bank rows. A
      // weight's N tile divides N; an online B is padded to whole tiles.
      int64_t limit = std::min<int64_t>(depth / packed, 1023);
      int64_t mt = evenTile(m, std::min(limit, tileRows));
      // Packed rows in place have no zero rows past M: tiles divide M.
      if (packedA)
        while (m % mt)
          --mt;
      // C: Mt * Nt / 4 bank rows, after the Nt / 4 of its bias.
      int64_t nLimit =
          std::min(limit, 4 * depth / (mt + (bias ? 1 : 0))) / 4 * 4;
      int64_t nt = nLimit;
      if (weight)
        while (nt >= 4 && n % nt)
          nt -= 4;
      else
        nt = (evenTile(n, nLimit) + 3) / 4 * 4;
      if (mt < 1 || nt < 4) {
        op->emitError("piplus product finds no tile within the banks");
        return signalPassFailure();
      }
      int64_t mTiles = (m + mt - 1) / mt, nTiles = (n + nt - 1) / nt;
      int64_t mPadded = mTiles * mt, nPadded = nTiles * nt;

      // An operand quantized online into [batch][blocks][padded][S * 16].
      auto quantize = [&](Operand input, int64_t count, int64_t padded,
                          bool rhs) -> Value {
        Value &result =
            quantized[{input.value.getAsOpaquePointer(), rhs, padded}];
        if (result)
          return result;
        Value source = input.quant.getInput();
        auto sourceType = cast<RankedTensorType>(source.getType());
        // DMA reads each block of a row in place when the views of the
        // operand keep it a strided view of the source whose K axis is
        // contiguous in the source's last (the quantization axis), whole
        // K rows of it.
        SmallVector<int64_t> strides = viewStrides(source, input.views);
        int64_t rank = cast<RankedTensorType>(input.value.getType()).getRank();
        int64_t kAxis = rank - (rhs ? 2 : 1);
        int64_t countAxis = rank - (rhs ? 1 : 2);
        // Or the count axis a collapse of two strided axes (the query heads
        // of a key head and the tokens, grouped-query attention): DMA reads
        // the rows of each outer index in turn, `outer` of them `outerStride`
        // apart, each of count / outer rows.
        int64_t outer = 1, outerStride = 0;
        auto collapse =
            input.views.empty()
                ? tensor::CollapseShapeOp()
                : dyn_cast<tensor::CollapseShapeOp>(input.views.front());
        SmallVector<int64_t> before =
            collapse && strides.empty()
                ? viewStrides(source,
                              ArrayRef<Operation *>(input.views).drop_front())
                : SmallVector<int64_t>();
        if (!before.empty()) {
          auto shape = collapse.getSrcType().getShape();
          SmallVector<int64_t> merged;
          bool split = false;
          for (auto [to, group] :
               llvm::enumerate(collapse.getReassociationIndices())) {
            SmallVector<int64_t> kept;
            for (int64_t axis : group)
              if (shape[axis] != 1)
                kept.push_back(axis);
            bool contiguous = true;
            for (int64_t i = 0; i + 1 < int64_t(kept.size()); ++i)
              contiguous &=
                  before[kept[i]] == before[kept[i + 1]] * shape[kept[i + 1]];
            if (!contiguous && int64_t(to) == countAxis && kept.size() == 2) {
              outer = shape[kept[0]];
              outerStride = before[kept[0]];
              split = true;
            } else if (!contiguous) {
              merged.clear();
              break;
            }
            merged.push_back(kept.empty() ? before[group.back()]
                                          : before[kept.back()]);
          }
          if (split && int64_t(merged.size()) == rank)
            strides = merged;
          else
            outer = 1;
        }
        bool direct =
            !strides.empty() &&
            int64_t(input.quant.getAxis()) == sourceType.getRank() - 1 &&
            sourceType.getShape().back() % k == 0 && strides[kAxis] == 1;
        // Or, the count axis contiguous and K the whole quantization axis
        // (V of attention, [K][N] in memory): strided DMA moves [kc][4]
        // column slices, which mxint8quant quantizes along kc with columns
        // into four packed rows each, so no transpose on the CPU.
        int64_t quantAxis = input.quant.getAxis();
        int64_t quantStride =
            bufferType(source).getStridesAndOffset().first[quantAxis];
        int64_t kBytes = strides.empty() ? 0 : strides[kAxis] * 4;
        bool sliced = !direct && !strides.empty() && strides[countAxis] == 1 &&
                      sourceType.getDimSize(quantAxis) == k &&
                      strides[kAxis] == quantStride &&
                      (kBytes <= 4095 || (count % 4 == 0 && kBytes % 16 == 0));
        // The source's first element: a slice the hart passes in place starts
        // where its buffer says.
        Value sourceBuffer;
        int64_t start = 0;
        OpFoldResult first = b.getIndexAttr(0);
        if (sliced || direct) {
          sourceBuffer = buffer(b, loc, source);
          sourceBuffer.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
          start = bufferType(source).getStridesAndOffset().second;
          if (ShapedType::isDynamic(start))
            first = offsetOf(b, loc, sourceBuffer);
        }
        if (sliced) {
          int64_t batchStride = batched ? strides[0] : count * k;
          Value view = memref::ReinterpretCastOp::create(
              b, loc,
              MemRefType::get(
                  {batch, k, count}, b.getF32Type(),
                  StridedLayoutAttr::get(b.getContext(), start,
                                         {batchStride, strides[kAxis], 1})),
              sourceBuffer, first,
              ArrayRef<OpFoldResult>{b.getIndexAttr(batch), b.getIndexAttr(k),
                                     b.getIndexAttr(count)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(batchStride),
                                     b.getIndexAttr(strides[kAxis]),
                                     b.getIndexAttr(1)});
          result = memref::AllocOp::create(
              b, loc,
              MemRefType::get({batch, blocks, padded, width}, b.getI8Type()));
          // Padded is whole tiles of a multiple of four rows: the four rows
          // of a group are within it. Rows past the last group are zero.
          int64_t groups = (count + 3) / 4;
          if (padded != groups * 4)
            linalg::FillOp::create(
                b, loc,
                ValueRange{arith::ConstantOp::create(
                    b, loc, b.getZeroAttr(b.getI8Type()))},
                ValueRange{memref::SubViewOp::create(
                    b, loc, result,
                    ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(0),
                                           b.getIndexAttr(groups * 4),
                                           b.getIndexAttr(0)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(batch),
                                           b.getIndexAttr(blocks),
                                           b.getIndexAttr(padded - groups * 4),
                                           b.getIndexAttr(width)},
                    ArrayRef<OpFoldResult>(
                        SmallVector<OpFoldResult>(4, b.getIndexAttr(1))))});
          // Group `group` (of `columns` columns) of block `block`: a tail
          // group starts from a cleared bank, its missing columns +0.0.
          auto slice = [&](Value at, Value block, Value group,
                           int64_t columns) {
            Value in = BankAllocOp::create(b, loc, i64, 1, 1);
            Value out = BankAllocOp::create(b, loc, i64, 1, 1);
            if (columns < 4)
              in = BankClearOp::create(b, loc, i64, in, 1, 1);
            Value part = memref::SubViewOp::create(
                b, loc, view,
                ArrayRef<OpFoldResult>{
                    at,
                    arith::MulIOp::create(b, loc, block, index(b, loc, kc))
                        .getResult(),
                    arith::MulIOp::create(b, loc, group, index(b, loc, 4))
                        .getResult()},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(kc),
                                       b.getIndexAttr(columns)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                       b.getIndexAttr(1)});
            in = BankMvinStridedOp::create(
                b, loc, i64, part, in, index(b, loc, 0), index(b, loc, kc),
                index(b, loc, kBytes), 0, 1, index(b, loc, columns * 4),
                index(b, loc, 0));
            out = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                b, loc, i64, in, out, 4, kc, true, 0, 0, 1, false);
            out = BankMvoutOp::create(
                b, loc, i64,
                rows(b, loc, result, ValueRange{at, block},
                     arith::MulIOp::create(b, loc, group, index(b, loc, 4))
                         .getResult(),
                     4, width),
                out, 4 * packed, 1);
            BankReleaseOp::create(b, loc, in);
            BankReleaseOp::create(b, loc, out);
          };
          if (count / 4)
            loops(b, loc, {batch, blocks, count / 4}, true,
                  [&](ValueRange at) { slice(at[0], at[1], at[2], 4); });
          if (count % 4)
            loops(b, loc, {batch, blocks}, true, [&](ValueRange at) {
              slice(at[0], at[1], index(b, loc, count / 4), count % 4);
            });
          return result;
        }
        Value values;
        if (direct) {
          int64_t batchStride = batched ? strides[0] : count * k;
          int64_t countStride = strides[countAxis];
          // [batch][blocks][outer][count / outer][kc].
          auto layout = StridedLayoutAttr::get(
              b.getContext(), start,
              {batchStride, kc, outerStride, countStride, 1});
          values = memref::ReinterpretCastOp::create(
              b, loc,
              MemRefType::get({batch, blocks, outer, count / outer, kc},
                              b.getF32Type(), layout),
              sourceBuffer, first,
              ArrayRef<OpFoldResult>{
                  b.getIndexAttr(batch), b.getIndexAttr(blocks),
                  b.getIndexAttr(outer), b.getIndexAttr(count / outer),
                  b.getIndexAttr(kc)},
              ArrayRef<OpFoldResult>{
                  b.getIndexAttr(batchStride), b.getIndexAttr(kc),
                  b.getIndexAttr(outerStride), b.getIndexAttr(countStride),
                  b.getIndexAttr(1)});
        } else {
          // The FP32 values gathered through the views, block by block.
          values = memref::AllocOp::create(
              b, loc,
              MemRefType::get({batch, blocks, count, kc}, b.getF32Type()));
          loops(b, loc, {batch, blocks, count, kc}, false, [&](ValueRange at) {
            SmallVector<Value> coords;
            if (batched)
              coords.push_back(at[0]);
            Value position = arith::AddIOp::create(
                b, loc, arith::MulIOp::create(b, loc, at[1], index(b, loc, kc)),
                at[3]);
            coords.push_back(rhs ? position : at[2]);
            coords.push_back(rhs ? at[2] : position);
            for (Operation *view : input.views) {
              if (auto transpose = dyn_cast<linalg::TransposeOp>(view)) {
                SmallVector<Value> reordered(coords.size());
                for (auto [i, axis] :
                     llvm::enumerate(transpose.getPermutation()))
                  reordered[axis] = coords[i];
                coords = reordered;
                continue;
              }
              auto original =
                  cast<RankedTensorType>(view->getOperand(0).getType());
              Value flat = index(b, loc, 0);
              for (auto [axis, coord] : llvm::enumerate(coords))
                flat = arith::AddIOp::create(
                    b, loc,
                    arith::MulIOp::create(
                        b, loc, flat,
                        tensor::DimOp::create(b, loc, view->getResult(0),
                                              axis)),
                    coord);
              coords.resize(original.getRank());
              for (int64_t axis = original.getRank() - 1; axis >= 0; --axis) {
                Value size =
                    tensor::DimOp::create(b, loc, view->getOperand(0), axis);
                coords[axis] = arith::RemUIOp::create(b, loc, flat, size);
                flat = arith::DivUIOp::create(b, loc, flat, size);
              }
            }
            Value value = tensor::ExtractOp::create(b, loc, source, coords);
            memref::StoreOp::create(b, loc, value, values, at);
          });
        }
        result = memref::AllocOp::create(
            b, loc,
            MemRefType::get({batch, blocks, padded, width}, b.getI8Type()));
        // Padding rows are zero codes with exponent byte 0: their products
        // are +0.0, finite as mxint8mm requires.
        if (padded != count)
          linalg::FillOp::create(
              b, loc,
              ValueRange{arith::ConstantOp::create(
                  b, loc, b.getZeroAttr(b.getI8Type()))},
              ValueRange{memref::SubViewOp::create(
                  b, loc, result,
                  ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(0),
                                         b.getIndexAttr(count),
                                         b.getIndexAttr(0)},
                  ArrayRef<OpFoldResult>{
                      b.getIndexAttr(batch), b.getIndexAttr(blocks),
                      b.getIndexAttr(padded - count), b.getIndexAttr(width)},
                  ArrayRef<OpFoldResult>(
                      SmallVector<OpFoldResult>(4, b.getIndexAttr(1))))});
        // Row `row` of the source rows `source` to row `target` of the result.
        auto row = [&](Value source, ValueRange at, Value target) {
          Value in = BankAllocOp::create(b, loc, i64, 1, 1);
          Value out = BankAllocOp::create(b, loc, i64, 1, 1);
          in = BankMvinOp::create(b, loc, i64, source, in, kc / 4, 1);
          out = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
              b, loc, i64, in, out, 1, kc, false, 0, 0, kc / 4, false);
          out = BankMvoutOp::create(
              b, loc, i64,
              rows(b, loc, result, at.take_front(2), target, 1, width), out,
              packed, 1);
          BankReleaseOp::create(b, loc, in);
          BankReleaseOp::create(b, loc, out);
        };
        if (direct)
          loops(b, loc, {batch, blocks, outer, count / outer}, true,
                [&](ValueRange at) {
                  row(rows(b, loc, values, at.take_front(3), at[3], 1, kc), at,
                      arith::AddIOp::create(
                          b, loc,
                          arith::MulIOp::create(b, loc, at[2],
                                                index(b, loc, count / outer)),
                          at[3]));
                });
        else
          loops(b, loc, {batch, blocks, count}, true, [&](ValueRange at) {
            row(rows(b, loc, values, at.take_front(2), at[2], 1, kc), at,
                at[2]);
          });
        return result;
      };
      Value lhs;
      if (packedA) {
        Value source = a.decode.getInput();
        auto sourceType = cast<RankedTensorType>(source.getType());
        SmallVector<int64_t> strides = viewStrides(source, a.views);
        int64_t rank = cast<RankedTensorType>(a.value.getType()).getRank();
        if (strides.empty() || strides[rank - 1] != 1 || blocks != 1 ||
            sourceType.getShape().back() != width) {
          op->emitError(
              "piplus packed A must be whole packed rows of one block");
          return signalPassFailure();
        }
        int64_t rowStride = strides[rank - 2];
        int64_t batchStride = batched ? strides[0] : m * rowStride;
        Value sourceBuffer = buffer(b, loc, source);
        sourceBuffer.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
        lhs = memref::ReinterpretCastOp::create(
            b, loc,
            MemRefType::get({batch, 1, m, width}, b.getI8Type(),
                            StridedLayoutAttr::get(
                                b.getContext(), 0,
                                {batchStride, m * rowStride, rowStride, 1})),
            sourceBuffer, b.getIndexAttr(0),
            ArrayRef<OpFoldResult>{b.getIndexAttr(batch), b.getIndexAttr(1),
                                   b.getIndexAttr(m), b.getIndexAttr(width)},
            ArrayRef<OpFoldResult>{
                b.getIndexAttr(batchStride), b.getIndexAttr(m * rowStride),
                b.getIndexAttr(rowStride), b.getIndexAttr(1)});
      } else {
        lhs = quantize(a, m, mPadded, false);
      }
      if (!weight)
        weights.push_back(quantize(w, n, nPadded, true));

      // C: each tile writes its rows and columns within [M, N] into the
      // destination (writeTile).
      // A product read within the subgraph (a projection its attention
      // product quantizes) writes a buffer of its own.
      Value destinationBuffer =
          destination ? buffer(b, loc, destination.getDest())
                      : memref::AllocOp::create(
                            b, loc,
                            MemRefType::get(outputType.getShape(),
                                            outputType.getElementType()));
      Value biasBuffer = bias ? buffer(b, loc, bias) : Value();
      if (biasBuffer)
        biasBuffer.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      Value residualBuffer = residual ? buffer(b, loc, residual) : Value();
      if (residualBuffer)
        residualBuffer.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      // The destination from row `row` and column `column`.
      auto window = [&](Value memory, Value at, Value row, Value column,
                        int64_t rowCount, int64_t columnCount) -> Value {
        SmallVector<OpFoldResult> offsets, sizes;
        if (batched) {
          offsets.push_back(at);
          sizes.push_back(b.getIndexAttr(1));
        }
        offsets.append({row, column});
        sizes.append({b.getIndexAttr(rowCount), b.getIndexAttr(columnCount)});
        return memref::SubViewOp::create(
            b, loc, memory, offsets, sizes,
            SmallVector<OpFoldResult>(offsets.size(), b.getIndexAttr(1)));
      };
      auto target = [&](Value at, Value row, Value column, int64_t rowCount,
                        int64_t columnCount) -> Value {
        return window(destinationBuffer, at, row, column, rowCount,
                      columnCount);
      };
      // The most 16-byte rows, up to sixteen, a strided DMA block of
      // `rowCount` can take.
      auto blockOf = [](int64_t rowCount) {
        int64_t block = 16;
        while (rowCount % block)
          block /= 2;
        return block;
      };
      // `count` 16-byte columns as strided DMA pieces of up to sixteen, a
      // power of two each: (first column, columns).
      auto piecesOf = [](int64_t count) {
        SmallVector<std::pair<int64_t, int64_t>> pieces;
        for (int64_t at = 0, block = 16; at < count; block /= 2)
          for (; count - at >= block; at += block)
            pieces.push_back({at, block});
        return pieces;
      };
      // The first `rowsOut` rows and `columnsOut` columns of the Mt x Nt tile
      // in bank `c` to the destination at (row, column): whole rows of the
      // destination by mvout; column chunks of blocks of 16-byte rows by
      // strided mvout when a row of the destination is within the 4095-byte
      // stride of the strided DMA or whole rows (the block DMA); else a tile
      // row at a time, its last bytes apart.
      // The instructions writeTile issues for a tile of `columnsOut` columns.
      auto columnsOf = [&](int64_t columnsOut) -> int64_t {
        if (columnsOut == n && nt == n)
          return 1;
        if (columnsOut % 4 == 0 && (n * 4 <= 4095 || n % 4 == 0))
          return piecesOf(columnsOut / 4).size();
        return 4;
      };
      auto writeTile = [&](Value c, Value at, Value row, Value column,
                           int64_t rowsOut, int64_t columnsOut,
                           int64_t offset) {
        if (columnsOut == n && nt == n && offset == 0) {
          BankMvoutOp::create(b, loc, i64, target(at, row, column, rowsOut, n),
                              c, rowsOut * n / 4, 1);
          return;
        }
        // Whole 16-byte columns in blocks, then the last columns of every
        // row by one strided DMA of their bytes.
        if ((columnsOut % 4 == 0 && n % 4 == 0) || n * 4 <= 4095) {
          int64_t whole = columnsOut / 4;
          if (columnsOut % 4)
            BankMvoutStridedOp::create(
                b, loc, i64,
                target(at, row,
                       arith::AddIOp::create(b, loc, column,
                                             index(b, loc, whole * 4)),
                       rowsOut, columnsOut % 4),
                c, index(b, loc, offset + whole), index(b, loc, rowsOut),
                index(b, loc, n * 4), 0, nt / 4,
                index(b, loc, columnsOut % 4 * 4), index(b, loc, 0));
          if (!whole)
            return;
          for (auto [first, block] : piecesOf(whole))
            c = BankMvoutStridedOp::create(
                b, loc, i64,
                target(at, row,
                       arith::AddIOp::create(b, loc, column,
                                             index(b, loc, first * 4)),
                       rowsOut, block * 4),
                c, index(b, loc, offset + first),
                index(b, loc, rowsOut * block), index(b, loc, n * 4), 0, nt / 4,
                index(b, loc, 16), index(b, loc, llvm::Log2_64(block)));
          return;
        }
        loops(b, loc, {rowsOut}, true, [&](ValueRange r) {
          Value at0 = arith::AddIOp::create(b, loc, row, r[0]);
          Value first = arith::AddIOp::create(
              b, loc,
              arith::MulIOp::create(b, loc, r[0], index(b, loc, nt / 4)),
              index(b, loc, offset));
          int64_t whole = columnsOut / 4;
          if (whole) {
            int64_t block = blockOf(whole);
            BankMvoutStridedOp::create(
                b, loc, i64, target(at, at0, column, 1, whole * 4), c, first,
                index(b, loc, whole), index(b, loc, block * 16), 0, block,
                index(b, loc, 16), index(b, loc, llvm::Log2_64(block)));
          }
          if (columnsOut % 4)
            BankMvoutStridedOp::create(
                b, loc, i64,
                target(at, at0,
                       arith::AddIOp::create(b, loc, column,
                                             index(b, loc, whole * 4)),
                       1, columnsOut % 4),
                c, arith::AddIOp::create(b, loc, first, index(b, loc, whole)),
                index(b, loc, 1), index(b, loc, 16), 0, 1,
                index(b, loc, columnsOut % 4 * 4), index(b, loc, 0));
        });
      };
      // The bank of a C tile at (row, column) before its products: the bias
      // of the tile's columns in the Nt / 4 bank rows before C's window, then
      // the residual's rows and columns of the tile in the window (the
      // strided mvin mirroring writeTile). A whole tile's first mvin defines
      // the bank, so the next tile's start and A load while this tile
      // computes; a part of a tile starts from a cleared bank, the rows and
      // columns past M and N at zero.
      int64_t offset = biasBuffer ? nt / 4 : 0;
      auto startTile = [&](Value at, Value row, Value column, int64_t rowsOut,
                           int64_t columnsOut) -> Value {
        Value c = BankAllocOp::create(b, loc, i64, 1, 1);
        bool partial = columnsOut < nt || (residualBuffer && rowsOut < mt);
        if (partial)
          c = BankClearOp::create(b, loc, i64, c, 1, 1);
        if (biasBuffer) {
          Value part = memref::SubViewOp::create(
              b, loc, biasBuffer, ArrayRef<OpFoldResult>{column},
              ArrayRef<OpFoldResult>{b.getIndexAttr(columnsOut)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
          c = partial
                  ? Value(BankMvinStridedOp::create(
                        b, loc, i64, part, c, index(b, loc, 0),
                        index(b, loc, columnsOut / 4), index(b, loc, 16), 0, 1,
                        index(b, loc, 16), index(b, loc, 0)))
                  : Value(BankMvinOp::create(b, loc, i64, part, c, nt / 4, 1));
        }
        if (!residualBuffer)
          return c;
        if (columnsOut == n && nt == n && !partial && !offset)
          return BankMvinOp::create(
              b, loc, i64, window(residualBuffer, at, row, column, rowsOut, n),
              c, rowsOut * n / 4, 1);
        if (columnsOut % 4 || !(n * 4 <= 4095 || n % 4 == 0))
          llvm::report_fatal_error(
              "piplus residual start needs whole bank rows");
        for (auto [first, block] : piecesOf(columnsOut / 4))
          c = BankMvinStridedOp::create(
              b, loc, i64,
              window(residualBuffer, at, row,
                     arith::AddIOp::create(b, loc, column,
                                           index(b, loc, first * 4)),
                     rowsOut, block * 4),
              c, index(b, loc, offset + first), index(b, loc, rowsOut * block),
              index(b, loc, n * 4), 0, nt / 4, index(b, loc, 16),
              index(b, loc, llvm::Log2_64(block)));
        return c;
      };
      // The B tile of term `term` at block `block` and column `column`.
      auto rhsTile = [&](int64_t term, Value at, Value block, Value column) {
        Value bank = BankAllocOp::create(b, loc, i64, 1, 1);
        Value tile = weight
                         ? rows(b, loc, weights[term], block, column, nt, width)
                         : rows(b, loc, weights[term], ValueRange{at, block},
                                column, nt, width);
        return BankMvinOp::create(b, loc, i64, tile, bank, nt * packed, 1)
            .getResult();
      };
      int64_t termCount = terms.size();
      // The C tile at (row, column) of the A tile in bank `ab` and the B
      // tiles of the terms in banks `wbs`, one K block.
      auto groupTile = [&](Value at, Value ab, ArrayRef<Value> wbs, Value row,
                           Value column, int64_t rowsOut, int64_t columnsOut) {
        Value c = startTile(at, row, column, rowsOut, columnsOut);
        for (auto [term, wb] : llvm::enumerate(wbs))
          ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
              b, loc, i64, ab, wb, c, mt, nt, kc, term > 0 || residualBuffer,
              term == 0 && biasBuffer, 0, nt / 4, offset, false);
        writeTile(c, at, row, column, rowsOut, columnsOut, offset);
        BankReleaseOp::create(b, loc, c);
      };
      // A single K block: a group of A tiles stays in its banks while every
      // B tile of the row passes once, so A is moved once and B once per
      // group, not per tile.
      auto group = [&](Value at, Value first, ArrayRef<int64_t> rowsOuts) {
        SmallVector<Value> as, rowsAt;
        for (auto [i, rowsOut] : llvm::enumerate(rowsOuts)) {
          Value row = arith::AddIOp::create(b, loc, first,
                                            index(b, loc, int64_t(i) * mt));
          Value ab = BankAllocOp::create(b, loc, i64, 1, 1);
          as.push_back(BankMvinOp::create(b, loc, i64,
                                          rows(b, loc, lhs,
                                               ValueRange{at, index(b, loc, 0)},
                                               row, mt, width),
                                          ab, mt * packed, 1));
          rowsAt.push_back(row);
        }
        auto columns = [&](Value column, int64_t columnsOut) {
          SmallVector<Value> wbs;
          for (int64_t term = 0; term < termCount; ++term)
            wbs.push_back(rhsTile(term, at, index(b, loc, 0), column));
          for (auto [i, rowsOut] : llvm::enumerate(rowsOuts))
            groupTile(at, as[i], wbs, rowsAt[i], column, rowsOut, columnsOut);
          for (Value wb : wbs)
            BankReleaseOp::create(b, loc, wb);
        };
        if (n / nt)
          loops(b, loc, {n / nt}, true, [&](ValueRange ni) {
            columns(arith::MulIOp::create(b, loc, ni[0], index(b, loc, nt)),
                    nt);
          });
        if (n % nt)
          columns(index(b, loc, n / nt * nt), n % nt);
        for (Value ab : as)
          BankReleaseOp::create(b, loc, ab);
      };
      // Up to 8 A tiles a group, as many as the thread's instruction stream
      // holds: each tile of a group is its own copy of the A load, the start
      // of C (startTile), a product per term, the B tiles of the terms and
      // the writes of C (writeTile), and a product has up to six copies of
      // the group (tails of M and N); the stream holds 640.
      int64_t writes = columnsOf(nt);
      int64_t groupTiles = std::max<int64_t>(
          1, std::min<int64_t>({8, std::max<int64_t>(m / mt, 1),
                                100 / (6 + 2 * termCount + writes)}));
      // Several K blocks: a group of A tiles accumulates in its C banks while
      // each K block's B tiles load once for the group (B moved once a
      // group, not once a tile; A once a block).
      auto blockGroup = [&](Value at, Value column, int64_t columnsOut,
                            Value first, ArrayRef<int64_t> rowsOuts) {
        SmallVector<Value> cs, rowsAt;
        for (auto [i, rowsOut] : llvm::enumerate(rowsOuts)) {
          Value row = arith::AddIOp::create(b, loc, first,
                                            index(b, loc, int64_t(i) * mt));
          cs.push_back(startTile(at, row, column, rowsOut, columnsOut));
          rowsAt.push_back(row);
        }
        auto block = [&](Value k, bool opening) {
          SmallVector<Value> wbs;
          for (int64_t term = 0; term < termCount; ++term)
            wbs.push_back(rhsTile(term, at, k, column));
          for (auto [i, c] : llvm::enumerate(cs)) {
            Value ab = BankAllocOp::create(b, loc, i64, 1, 1);
            ab = BankMvinOp::create(
                b, loc, i64,
                rows(b, loc, lhs, ValueRange{at, k}, rowsAt[i], mt, width), ab,
                mt * packed, 1);
            for (auto [term, wb] : llvm::enumerate(wbs)) {
              bool start = opening && term == 0;
              ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
                  b, loc, i64, ab, wb, c, mt, nt, kc, !start || residualBuffer,
                  start && biasBuffer, 0, nt / 4, offset, false);
            }
            BankReleaseOp::create(b, loc, ab);
          }
          for (Value wb : wbs)
            BankReleaseOp::create(b, loc, wb);
        };
        block(index(b, loc, 0), true);
        loops(b, loc, {blocks - 1}, true, [&](ValueRange part) {
          block(arith::AddIOp::create(b, loc, part[0], index(b, loc, 1)),
                false);
        });
        for (auto [i, rowsOut] : llvm::enumerate(rowsOuts)) {
          writeTile(cs[i], at, rowsAt[i], column, rowsOut, columnsOut, offset);
          BankReleaseOp::create(b, loc, cs[i]);
        }
      };
      // The groups of A tiles of a column of B tiles, `columnsOut` columns of
      // which are within N; the last tile of M writes the rows within M.
      auto columnGroups = [&](Value at, Value column, int64_t columnsOut) {
        int64_t full = m / mt;
        if (full / groupTiles)
          loops(b, loc, {full / groupTiles}, true, [&](ValueRange gi) {
            blockGroup(at, column, columnsOut,
                       arith::MulIOp::create(b, loc, gi[0],
                                             index(b, loc, groupTiles * mt)),
                       SmallVector<int64_t>(groupTiles, mt));
          });
        if (full % groupTiles)
          blockGroup(at, column, columnsOut,
                     index(b, loc, full / groupTiles * groupTiles * mt),
                     SmallVector<int64_t>(full % groupTiles, mt));
        if (m % mt)
          blockGroup(at, column, columnsOut, index(b, loc, full * mt),
                     {m % mt});
      };
      // Several K blocks whose A tiles the banks hold all at once: a group
      // of A tiles is moved once, each K block of a tile into a bank of its
      // own, and stays while every column of B tiles passes and accumulates
      // over the blocks in the group's C banks — A once a group and B once a
      // group, where blockGroup moves A again for every column. The first
      // column moves the A tiles as blockGroup does, each right before its
      // products, so the array starts after one B and one A tile; the other
      // columns find them in their banks. The sums are blockGroup's: block
      // after block, term after term.
      auto heldGroup = [&](Value at, Value first, ArrayRef<int64_t> rowsOuts) {
        SmallVector<SmallVector<Value>> as(rowsOuts.size());
        SmallVector<Value> rowsAt;
        for (size_t i = 0; i < rowsOuts.size(); ++i)
          rowsAt.push_back(arith::AddIOp::create(
              b, loc, first, index(b, loc, int64_t(i) * mt)));
        auto columns = [&](Value column, int64_t columnsOut, bool opening) {
          SmallVector<Value> cs;
          for (auto [i, rowsOut] : llvm::enumerate(rowsOuts))
            cs.push_back(startTile(at, rowsAt[i], column, rowsOut, columnsOut));
          for (int64_t k = 0; k < blocks; ++k) {
            SmallVector<Value> wbs;
            for (int64_t term = 0; term < termCount; ++term)
              wbs.push_back(rhsTile(term, at, index(b, loc, k), column));
            for (auto [i, c] : llvm::enumerate(cs)) {
              if (opening) {
                Value ab = BankAllocOp::create(b, loc, i64, 1, 1);
                as[i].push_back(BankMvinOp::create(
                    b, loc, i64,
                    rows(b, loc, lhs, ValueRange{at, index(b, loc, k)},
                         rowsAt[i], mt, width),
                    ab, mt * packed, 1));
              }
              for (auto [term, wb] : llvm::enumerate(wbs)) {
                bool start = k == 0 && term == 0;
                ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
                    b, loc, i64, as[i][k], wb, c, mt, nt, kc,
                    !start || residualBuffer, start && biasBuffer, 0, nt / 4,
                    offset, false);
              }
            }
            for (Value wb : wbs)
              BankReleaseOp::create(b, loc, wb);
          }
          for (auto [i, rowsOut] : llvm::enumerate(rowsOuts)) {
            writeTile(cs[i], at, rowsAt[i], column, rowsOut, columnsOut,
                      offset);
            BankReleaseOp::create(b, loc, cs[i]);
          }
        };
        int64_t whole = n / nt;
        columns(index(b, loc, 0), whole ? nt : n % nt, true);
        if (whole > 1)
          loops(b, loc, {whole - 1}, true, [&](ValueRange ni) {
            columns(arith::MulIOp::create(
                        b, loc,
                        arith::AddIOp::create(b, loc, ni[0], index(b, loc, 1)),
                        index(b, loc, nt)),
                    nt, false);
          });
        if (whole && n % nt)
          columns(index(b, loc, whole * nt), n % nt, false);
        for (auto &held : as)
          for (Value ab : held)
            BankReleaseOp::create(b, loc, ab);
      };
      // The A tiles of a held group: their blocks, a B tile a term and their
      // C tiles within the banks (two are left for the rest of the
      // subgraph), and its instructions (the first column with the moves of
      // A, a column of the loop, one of the last columns of N, the releases)
      // within a third of what groupTiles leaves a product: a product has
      // the whole groups, the last tiles and the last rows of M.
      int64_t heldTiles = std::min<int64_t>(
          (target::getTarget().bankNum - 2 - termCount) / (blocks + 1),
          std::max<int64_t>(m / mt, 1));
      auto heldCost = [&](int64_t tiles) {
        int64_t column =
            2 * tiles + blocks * termCount * (3 + tiles) + tiles * (writes + 1);
        return 3 * column + 3 * tiles * blocks;
      };
      while (heldTiles > 1 && heldCost(heldTiles) > 200)
        --heldTiles;
      if (blocks == 1) {
        int64_t full = m / mt;
        loops(b, loc, {batch}, true, [&](ValueRange bt) {
          if (full / groupTiles)
            loops(b, loc, {full / groupTiles}, true, [&](ValueRange gi) {
              group(bt[0],
                    arith::MulIOp::create(b, loc, gi[0],
                                          index(b, loc, groupTiles * mt)),
                    SmallVector<int64_t>(groupTiles, mt));
            });
          if (full % groupTiles)
            group(bt[0], index(b, loc, full / groupTiles * groupTiles * mt),
                  SmallVector<int64_t>(full % groupTiles, mt));
          if (m % mt)
            group(bt[0], index(b, loc, full * mt), {m % mt});
        });
      } else if (heldTiles >= 1) {
        int64_t full = m / mt;
        loops(b, loc, {batch}, true, [&](ValueRange bt) {
          if (full / heldTiles)
            loops(b, loc, {full / heldTiles}, true, [&](ValueRange gi) {
              heldGroup(bt[0],
                        arith::MulIOp::create(b, loc, gi[0],
                                              index(b, loc, heldTiles * mt)),
                        SmallVector<int64_t>(heldTiles, mt));
            });
          if (full % heldTiles)
            heldGroup(bt[0], index(b, loc, full / heldTiles * heldTiles * mt),
                      SmallVector<int64_t>(full % heldTiles, mt));
          if (m % mt)
            heldGroup(bt[0], index(b, loc, full * mt), {m % mt});
        });
      } else {
        if (n / nt)
          loops(b, loc, {batch, n / nt}, true, [&](ValueRange bn) {
            columnGroups(
                bn[0], arith::MulIOp::create(b, loc, bn[1], index(b, loc, nt)),
                nt);
          });
        if (n % nt)
          loops(b, loc, {batch}, true, [&](ValueRange bt) {
            columnGroups(bt[0], index(b, loc, n / nt * nt), n % nt);
          });
      }
      Value written = bufferization::ToTensorOp::create(
          b, loc, outputType, destinationBuffer, true, true);
      if (destination) {
        destination.getResult().replaceAllUsesWith(written);
        destination.erase();
      } else {
        op->getResult(0).replaceAllUsesWith(written);
      }
      for (Operation *term : llvm::reverse(terms))
        term->erase();
      if (residualSum && residualSum->use_empty())
        residualSum->erase();
      if (start && start->use_empty())
        start->erase();
    }
    // Delete the Q/DQ and layout chains consumed by the Ball lowering.
    RewritePatternSet patterns(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        failed(lowerRowOperators(
            getOperation(), [&](StringAttr name) { return number(name); })) ||
        failed(lowerAttention(getOperation())) ||
        failed(lowerSoftmax(getOperation())) ||
        failed(lowerLayerNorm(getOperation())))
      return signalPassFailure();
    if (operators.empty())
      return;
    // The entry of the operator image: runs operator a7 with a0-a7.
    ModuleOp module = getOperation();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = module.getLoc();
    SmallVector<Type> arguments(8, b.getI64Type());
    auto entry = func::FuncOp::create(b, loc, "operator",
                                      b.getFunctionType(arguments, {}));
    b.setInsertionPointToStart(entry.addEntryBlock());
    auto cases = llvm::to_vector(llvm::seq<int64_t>(0, operators.size()));
    auto select = scf::IndexSwitchOp::create(
        b, loc, TypeRange{},
        arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                   entry.getArgument(7)),
        cases, cases.size());
    for (auto [region, name] :
         llvm::zip_equal(select.getCaseRegions(), operators)) {
      b.setInsertionPointToStart(&region.emplaceBlock());
      func::CallOp::create(b, loc, name.getValue(), TypeRange{},
                           entry.getArguments());
      scf::YieldOp::create(b, loc);
    }
    b.setInsertionPointToStart(&select.getDefaultRegion().emplaceBlock());
    scf::YieldOp::create(b, loc);
    b.setInsertionPointAfter(select);
    func::ReturnOp::create(b, loc);
  }
  // The operators this module launches, in the order of their numbers.
  SmallVector<StringAttr> operators;
  int64_t number(StringAttr name) {
    auto found = llvm::find(operators, name);
    if (found != operators.end())
      return found - operators.begin();
    operators.push_back(name);
    return operators.size() - 1;
  }
};
PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
