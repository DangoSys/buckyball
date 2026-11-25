//===- ConvertLinalgToAcc.cpp - Poly MXINT8 computations on banks -------===//
//
// Every product of a thread subgraph becomes tiles of Ball instructions:
// Mt x Nt output tiles, each the Ball's own M rows times N columns, within
// the limits of the Ball and the bank depth of the target (design.pb).
//
//   - MXINT8 products (both operands decoded MXINT8) run on
//     Mxint8matmulBall over packed K blocks (codes, exponents and padding
//     to 16 bytes), chained with `accumulate`. Static K blocks divide K
//     within bank capacity; dynamic K uses blocks of 32.
//
// Operands: a packed operand (a weight, or an activation the reduce hart
// quantized) is read in place when K fits, otherwise repacked into blocks.
// It must be in the Ball layout, rows of
// 16 * S(K) bytes (the recipe packs weights so offline). An operand
// quantized online is quantized by Mxint8quantizeBall straight into the
// [batch][block][row][S] layout the product reads, several rows per
// instruction.
//
// Loops: batch, row tiles, column tiles, K blocks. A row tile of A is moved
// into available banks once; retained K blocks serve every column tile.
// The remaining K blocks stream through one bank. Each weight tile serves all
// its rows through the Ball's M. Row counts that are no multiple of Mt end with
// tiles of halving height down to single rows: the
// command fields of a recorded stream loop are constant, so each tile height
// is its own loop.
//
// Results: a C tile is Mt x Nt row-major in its bank and mvout moves rows
// from bank row 0 with one stride, so it lands in place only when its rows
// are whole output rows (Nt = N) or a single row. Other tiles go to a
// tile-major scratch [batch][column tile][row][Nt], copied into the output
// by the hart after completion, one 2-D copy per column tile.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Conversion/Buckyball.h"
#include "Frontend/FrontendOps.h"
#include "Hart/HartOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mxint8actOps.h"
#include "mxint8matmulOps.h"
#include "mxint8quantizeOps.h"
#include "transposeOps.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>
#include <numeric>

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

LogicalResult lowerPolyQuantizedOutputs(ModuleOp module);
LogicalResult lowerPolyRowOperators(ModuleOp module);

void preparePolyActivations(ModuleOp module);
LogicalResult lowerPolyActivations(ModuleOp module);
void preparePolyIntegerSoftmax(ModuleOp module);
LogicalResult lowerPolyIntegerSoftmax(ModuleOp module);

namespace {
Value index(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(b, loc, value);
}

// The bank rows of one packed MXINT8 row of k values.
int64_t packedRows(int64_t k) { return k / 16 + (k + 511) / 512; }

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
             tensor::ExpandShapeOp, tensor::ExtractSliceOp>(op))
      break;
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
        slice && slice.getType().getRank() != slice.getSourceType().getRank())
      break;
    result.views.push_back(op);
    value = op->getOperand(0);
  }
  return result;
}

// Build a loop nest, keeping large model dimensions in the IR as loops.
void loops(OpBuilder &b, Location loc, ValueRange sizes, bool stream,
           function_ref<void(ValueRange)> body) {
  OpBuilder::InsertionGuard guard(b);
  SmallVector<Value> ivs;
  for (Value size : sizes) {
    auto loop =
        scf::ForOp::create(b, loc, index(b, loc, 0), size, index(b, loc, 1));
    if (stream)
      loop->setAttr("hart.stream", b.getUnitAttr());
    ivs.push_back(loop.getInductionVar());
    b.setInsertionPointToStart(loop.getBody());
  }
  body(ivs);
}

// `total` rows in recorded tiles of `tile` rows, then the rest in tiles of
// halving height down to `least` rows, then single rows. body(first row,
// height).
void rowTiles(OpBuilder &b, Location loc, Value total, int64_t tile,
              int64_t least, function_ref<void(Value, int64_t)> body) {
  Value count = b.createOrFold<arith::DivUIOp>(loc, total, index(b, loc, tile));
  Value rest = b.createOrFold<arith::RemUIOp>(loc, total, index(b, loc, tile));
  Value done = b.createOrFold<arith::MulIOp>(loc, count, index(b, loc, tile));
  auto tiles = [&](Value trips, Value start, int64_t rows) {
    if (matchPattern(trips, m_Zero()))
      return;
    loops(b, loc, trips, true, [&](ValueRange at) {
      body(arith::AddIOp::create(
               b, loc, start,
               arith::MulIOp::create(b, loc, at[0], index(b, loc, rows))),
           rows);
    });
  };
  tiles(count, index(b, loc, 0), tile);
  if (tile == 1)
    return;
  int64_t part = 1;
  while (part * 2 < tile)
    part *= 2;
  for (; part >= least; part /= 2) {
    Value has = b.createOrFold<arith::RemUIOp>(
        loc, b.createOrFold<arith::DivUIOp>(loc, rest, index(b, loc, part)),
        index(b, loc, 2));
    Value start = b.createOrFold<arith::AddIOp>(
        loc, done,
        b.createOrFold<arith::MulIOp>(
            loc,
            b.createOrFold<arith::DivUIOp>(loc, rest, index(b, loc, 2 * part)),
            index(b, loc, 2 * part)));
    tiles(has, start, part);
  }
  if (least > 1) {
    Value singles =
        b.createOrFold<arith::RemUIOp>(loc, rest, index(b, loc, least));
    tiles(singles, b.createOrFold<arith::SubIOp>(loc, total, singles), 1);
  }
}

Value buffer(OpBuilder &b, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> strides(type.getRank());
  int64_t pitch = 1;
  for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
    strides[axis] = pitch;
    pitch = pitch < 0 || type.isDynamicDim(axis)
                ? ShapedType::kDynamic
                : pitch * type.getDimSize(axis);
  }
  auto layout =
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, strides);
  return bufferization::ToBufferOp::create(
      b, loc, MemRefType::get(type.getShape(), type.getElementType(), layout),
      tensor);
}

// Rows `first` .. `first + rows` of a [batch][block][row][width] staging
// array at (batch, block): contiguous bank rows for DMA.
Value rowsOf(OpBuilder &b, Location loc, Value memory, Value batch, Value block,
             Value first, int64_t rows, int64_t width) {
  SmallVector<OpFoldResult> at{batch, block, first, b.getIndexAttr(0)};
  SmallVector<OpFoldResult> size{b.getIndexAttr(1), b.getIndexAttr(1),
                                 b.getIndexAttr(rows), b.getIndexAttr(width)};
  SmallVector<OpFoldResult> stride(4, b.getIndexAttr(1));
  return memref::SubViewOp::create(b, loc, memory, at, size, stride);
}

// The largest multiple of four dividing `n`, at most `limit`.
int64_t divisor(int64_t n, int64_t limit) {
  for (int64_t d = limit - limit % 4; d >= 4; d -= 4)
    if (n % d == 0)
      return d;
  return 0;
}

class ConvertLinalgToAccPass
    : public PassWrapper<ConvertLinalgToAccPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToAccPass)
  StringRef getArgument() const final { return "convert-linalg-to-acc"; }
  StringRef getDescription() const final {
    return "Map poly MXINT8 products and their quantization to banks";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    mlir::buddy::registerBallDialects(r);
    r.insert<::buddy::mxint8act_ball::Mxint8actBallDialect, AccDialect,
             cf::ControlFlowDialect, arith::ArithDialect,
             bufferization::BufferizationDialect, memref::MemRefDialect,
             scf::SCFDialect, tensor::TensorDialect,
             ::buddy::mxint8matmul_ball::Mxint8matmulBallDialect,
             ::buddy::mxint8quantize_ball::Mxint8quantizeBallDialect,
             ::buddy::transpose_ball::TransposeBallDialect>();
  }
  void runOnOperation() override {
    preparePolyActivations(getOperation());
    preparePolyIntegerSoftmax(getOperation());
    const auto &target = target::getTarget();
    const int64_t depth = target.bankDepth;
    const int64_t banks = std::min(target.bankNum, target.privateBankMax);
    if (target.bankWidthBits != 128 || banks < 3) {
      getOperation().emitError("poly needs 128-bit banks and three banks");
      return signalPassFailure();
    }
    // Remove unit source dimensions before composing a flatten/expand pair.
    // This exposes head splits without mixing their independent K dimension.
    getOperation().walk([&](tensor::ExpandShapeOp expand) {
      auto function = expand->getParentOfType<func::FuncOp>();
      auto profile = function
                         ? function->getAttrOfType<StringAttr>("bbacc.target")
                         : StringAttr();
      auto collapse = expand.getSrc().getDefiningOp<tensor::CollapseShapeOp>();
      if (!profile || profile.getValue() != target.hart || !collapse)
        return;
      auto sourceType = collapse.getSrcType();
      auto resultType = expand.getResultType();
      if (sourceType.getNumDynamicDims() > 1 ||
          resultType.getNumDynamicDims() > 1 ||
          llvm::is_contained(sourceType.getShape(), 0) ||
          llvm::is_contained(resultType.getShape(), 0))
        return;
      SmallVector<int64_t> shape;
      for (int64_t dim : sourceType.getShape())
        if (dim != 1)
          shape.push_back(dim);
      if (shape.empty() || shape.size() == size_t(sourceType.getRank()) ||
          shape.size() >= size_t(expand.getResultType().getRank()))
        return;
      auto squeezedType =
          RankedTensorType::get(shape, sourceType.getElementType());
      auto squeeze =
          getReassociationIndicesForReshape(sourceType, squeezedType);
      auto split = getReassociationIndicesForReshape(squeezedType,
                                                     expand.getResultType());
      if (!squeeze || !split)
        return;
      OpBuilder b(expand);
      Value squeezed = tensor::CollapseShapeOp::create(
          b, expand.getLoc(), squeezedType, collapse.getSrc(), *squeeze);
      Value result = tensor::ExpandShapeOp::create(
          b, expand.getLoc(), expand.getResultType(), squeezed, *split,
          expand.getMixedOutputShape());
      expand.getResult().replaceAllUsesWith(result);
      expand.erase();
    });
    // Column tiles of at most the square the accumulator bank holds
    // (Mt * Nt / 4 <= depth), so rows and columns both get reuse.
    int64_t square = int64_t(std::sqrt(double(4 * depth)));
    square -= square % 4;
    SmallVector<Operation *> products;
    getOperation().walk([&](Operation *op) {
      auto function = op->getParentOfType<func::FuncOp>();
      auto profile = function
                         ? function->getAttrOfType<StringAttr>("bbacc.target")
                         : StringAttr();
      if (profile && profile.getValue() == target.hart &&
          isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
        products.push_back(op);
    });
    for (Operation *op : products) {
      OpBuilder b(op);
      Location loc = op->getLoc();
      Operand a = operand(op->getOperand(0)), w = operand(op->getOperand(1));
      if (!a.decode || !w.decode) {
        op->emitError("poly matrix products require MXINT8 operands");
        return signalPassFailure();
      }
      auto outputType = cast<RankedTensorType>(op->getResult(0).getType());
      auto aType = cast<RankedTensorType>(a.value.getType());
      bool batched = isa<linalg::BatchMatmulOp>(op);
      int64_t batch = batched ? outputType.getDimSize(0) : 1;
      int64_t n = outputType.getDimSize(batched ? 2 : 1);
      int64_t k = aType.getShape().back();
      auto fill = op->getOperand(2).getDefiningOp<linalg::FillOp>();
      bool zero = matchPattern(fill ? fill.getInputs()[0] : op->getOperand(2),
                               m_PosZeroFloat());
      if (batch < 0 || k == 0 || (k >= 0 && k % 32) || !zero) {
        op->emitError("poly product needs a static batch, aligned positive K "
                      "and zero initialization");
        return signalPassFailure();
      }
      auto users = op->getResult(0).getUsers();
      auto destination =
          op->getResult(0).hasOneUse()
              ? dyn_cast<bufferization::MaterializeInDestinationOp>(
                    *users.begin())
              : bufferization::MaterializeInDestinationOp();
      if (!destination) {
        op->emitError("poly product must write its hart subgraph destination");
        return signalPassFailure();
      }
      MemRefLayoutAttrInterface outputLayout;
      if (auto columns = op->getAttrOfType<IntegerAttr>("hart.columns"))
        outputLayout = StridedLayoutAttr::get(
            b.getContext(), ShapedType::kDynamic, {columns.getInt(), 1});
      Value output = bufferization::ToBufferOp::create(
          b, loc,
          MemRefType::get(outputType.getShape(), outputType.getElementType(),
                          outputLayout),
          destination.getDest());
      auto size = [&](int64_t axis) -> Value {
        if (!outputType.isDynamicDim(axis))
          return index(b, loc, outputType.getDimSize(axis));
        return tensor::DimOp::create(b, loc, destination.getDest(), axis);
      };
      Value mSize = size(batched ? 1 : 0);
      Value nSize = size(outputType.getRank() - 1);
      Value kSize;
      if (k >= 0) {
        kSize = index(b, loc, k);
      } else {
        Value source = a.quant ? a.quant.getInput() : a.decode.getInput();
        Value extent =
            tensor::DimOp::create(b, loc, source, a.decode.getAxis());
        kSize = a.quant ? b.createOrFold<arith::MulIOp>(
                              loc,
                              b.createOrFold<arith::DivUIOp>(
                                  loc,
                                  b.createOrFold<arith::AddIOp>(
                                      loc, extent, index(b, loc, 31)),
                                  index(b, loc, 32)),
                              index(b, loc, 32))
                        : b.createOrFold<arith::MulIOp>(
                              loc,
                              b.createOrFold<arith::DivUIOp>(loc, extent,
                                                             index(b, loc, 33)),
                              index(b, loc, 32));
      }
      auto productOp = cast<linalg::LinalgOp>(op);
      bool transposedRhs =
          cast<AffineDimExpr>(
              productOp.getIndexingMapsArray()[1].getResult(batched ? 1 : 0))
              .getPosition() == (batched ? 2 : 1);

      // K blocks stay aligned to the quantization groups and bank capacity.
      int64_t chunk = 32;
      if (k >= 0) {
        int64_t limit = (depth / 4 * 16 / 33) * 32;
        if (a.quant || w.quant)
          limit = std::min(limit, depth * 4);
        chunk = 8 * divisor(k / 8, limit / 8);
      }
      if (!w.quant) {
        auto packed = cast<RankedTensorType>(w.decode.getInput().getType());
        if (packed.getRank() == 3 && w.decode.getAxis() == 2)
          chunk = packed.getDimSize(2) / 33 * 32;
      }
      if (chunk <= 0 || chunk % 32 || (k >= 0 && k % chunk)) {
        op->emitError("poly product cannot fit an aligned K block in a bank");
        return signalPassFailure();
      }
      int64_t chunks = k < 0 ? ShapedType::kDynamic : k / chunk;
      Value chunkCount = b.createOrFold<arith::DivUIOp>(
          loc,
          b.createOrFold<arith::AddIOp>(loc, kSize, index(b, loc, chunk - 1)),
          index(b, loc, chunk));
      // Bank rows and bytes per packed operand row.
      int64_t rowDepth = packedRows(chunk);
      int64_t width = rowDepth * 16;
      // Operand rows a bank holds.
      int64_t fit = std::min<int64_t>(depth / rowDepth, 1023);
      // Rows one quantization takes: its FP32 input and packed output.
      int64_t quantRows = std::min(depth / (chunk / 4), depth / rowDepth);
      if (fit < 1 || quantRows < 1) {
        op->emitError("poly product: one row of K exceeds a bank");
        return signalPassFailure();
      }

      // A packed operand read in place: rows of 16 * S(K) bytes, the slices
      // of a padded weight seen through; a weight [N, K] read as B[n, k],
      // directly or through one transpose.
      auto inPlace = [&](Operand input, bool rhs) -> Value {
        bool rows = input.views.empty() && (!rhs || transposedRhs);
        if (rhs && input.views.size() == 1 && !transposedRhs)
          if (auto transpose = dyn_cast<linalg::TransposeOp>(input.views[0]))
            rows = transpose.getPermutation() == ArrayRef<int64_t>{1, 0};
        Value source = input.decode.getInput();
        while (auto slice = source.getDefiningOp<tensor::ExtractSliceOp>()) {
          if (!slice.hasUnitStride())
            return Value();
          if (slice->hasAttr("hart.columns"))
            break;
          if (!slice.hasZeroOffset())
            return Value();
          source = slice.getSource();
        }
        auto type = cast<RankedTensorType>(source.getType());
        if (rhs && type.getRank() == 3 && type.hasStaticShape() &&
            input.decode.getAxis() == 2 && type.getDimSize(0) == chunks &&
            type.getDimSize(2) == width) {
          ArrayRef<Operation *> views = input.views;
          if (!transposedRhs && !views.empty()) {
            auto transpose = dyn_cast<linalg::TransposeOp>(views.front());
            if (!transpose ||
                transpose.getPermutation() != ArrayRef<int64_t>{1, 0})
              return Value();
            views = views.drop_front();
          }
          if (views.size() != 2)
            return Value();
          auto collapse = dyn_cast<tensor::CollapseShapeOp>(views[0]);
          auto transpose = dyn_cast<linalg::TransposeOp>(views[1]);
          if (collapse && transpose &&
              (collapse.getReassociationIndices() ==
                   SmallVector<ReassociationIndices>{{0}, {1, 2}} ||
               (chunks == 1 &&
                collapse.getReassociationIndices() ==
                    SmallVector<ReassociationIndices>{{0, 1}, {2}})) &&
              transpose.getPermutation() == ArrayRef<int64_t>{1, 0, 2})
            return source;
          return Value();
        }
        if (k < 0 || type.getRank() != 2 || !rows ||
            int64_t(input.decode.getAxis()) != 1 ||
            type.getDimSize(1) != packedRows(k) * 16)
          return Value();
        return source;
      };
      Value aSource, wSource;
      if (!a.quant)
        aSource = inPlace(a, false);
      if (!w.quant)
        wSource = inPlace(w, true);

      // Column tiles: a divisor of the rows of a packed weight (no reads
      // past it), else as wide as the bank allows, balanced over N.
      int64_t columns;
      if (wSource) {
        auto type = cast<RankedTensorType>(wSource.getType());
        columns =
            divisor(type.getDimSize(type.getRank() - 2), std::min(fit, square));
      } else {
        columns = std::min(fit, square);
        columns -= columns % 4;
        if (n >= 0) {
          int64_t padded = (n + 3) / 4 * 4;
          int64_t count = (padded + columns - 1) / columns;
          columns = ((padded + count - 1) / count + 3) / 4 * 4;
        }
      }
      if (columns < 4) {
        op->emitError("poly product cannot fit a column tile in a bank");
        return signalPassFailure();
      }
      // Row tiles: what the A bank and the accumulator bank (Mt * Nt FP32)
      // hold.
      int64_t rows = std::min(fit, 4 * depth / columns);
      if (columns < 4 || rows < 1) {
        op->emitError("poly product cannot fit a tile in a bank");
        return signalPassFailure();
      }
      Value columnCount = b.createOrFold<arith::DivUIOp>(
          loc,
          b.createOrFold<arith::AddIOp>(loc, nSize, index(b, loc, columns - 1)),
          index(b, loc, columns));
      Value paddedSize = b.createOrFold<arith::MulIOp>(loc, columnCount,
                                                       index(b, loc, columns));
      int64_t paddedN =
          n < 0 ? ShapedType::kDynamic : (n + columns - 1) / columns * columns;
      bool padded = n < 0 || paddedN != n;
      bool residentWeights = !batched && chunks == 1 && wSource && !padded;

      // An operand quantized online: staged as
      // [batch][block][row][chunk] (in place when contiguous), then
      // quantized by the Ball into [batch][block][row][width].
      auto coordinates = [&](Operand input, bool rhs, Value batchIndex,
                             Value row, Value reduction) {
        SmallVector<Value> coords;
        if (batched)
          coords.push_back(batchIndex);
        coords.push_back(rhs && !transposedRhs ? reduction : row);
        coords.push_back(rhs && !transposedRhs ? row : reduction);
        for (Operation *view : input.views) {
          if (auto slice = dyn_cast<tensor::ExtractSliceOp>(view)) {
            auto offsets = slice.getMixedOffsets();
            auto strides = slice.getMixedStrides();
            for (auto [axis, coord] : llvm::enumerate(coords))
              coords[axis] = b.createOrFold<arith::AddIOp>(
                  loc, getValueOrCreateConstantIndexOp(b, loc, offsets[axis]),
                  b.createOrFold<arith::MulIOp>(
                      loc, coord,
                      getValueOrCreateConstantIndexOp(b, loc, strides[axis])));
          } else if (auto transpose = dyn_cast<linalg::TransposeOp>(view)) {
            SmallVector<Value> reordered(coords.size());
            for (auto [i, axis] : llvm::enumerate(transpose.getPermutation()))
              reordered[axis] = coords[i];
            coords = reordered;
          } else {
            auto original =
                cast<RankedTensorType>(view->getOperand(0).getType());
            bool collapse = isa<tensor::CollapseShapeOp>(view);
            auto groups = collapse ? cast<tensor::CollapseShapeOp>(view)
                                         .getReassociationIndices()
                                   : cast<tensor::ExpandShapeOp>(view)
                                         .getReassociationIndices();
            auto split =
                collapse ? original
                         : cast<RankedTensorType>(view->getResult(0).getType());
            SmallVector<Value> mapped(original.getRank(), index(b, loc, 0));
            bool unitOnly = true;
            for (auto [group, dimensions] : llvm::enumerate(groups)) {
              int64_t varying = -1;
              for (int64_t axis : dimensions) {
                if (split.getDimSize(axis) == 1)
                  continue;
                if (varying != -1) {
                  unitOnly = false;
                  break;
                }
                varying = axis;
              }
              if (!unitOnly)
                break;
              if (varying != -1)
                mapped[collapse ? varying : group] =
                    coords[collapse ? group : varying];
            }
            // Unit dimensions do not change reductions. In
            // particular, do not wrap K by a dynamic cache extent.
            if (unitOnly) {
              coords = std::move(mapped);
              continue;
            }
            // Reassociation only mixes coordinates within each group. Keep
            // independent axes intact instead of flattening the whole tensor.
            for (auto [group, dimensions] : llvm::enumerate(groups)) {
              if (collapse) {
                Value flat = coords[group];
                for (int64_t i = dimensions.size() - 1; i > 0; --i) {
                  int64_t axis = dimensions[i];
                  Value size =
                      tensor::DimOp::create(b, loc, view->getOperand(0), axis);
                  mapped[axis] =
                      b.createOrFold<arith::RemUIOp>(loc, flat, size);
                  flat = b.createOrFold<arith::DivUIOp>(loc, flat, size);
                }
                mapped[dimensions.front()] = flat;
              } else {
                Value flat = coords[dimensions.front()];
                for (int64_t axis : llvm::drop_begin(dimensions)) {
                  Value size =
                      tensor::DimOp::create(b, loc, view->getResult(0), axis);
                  flat = b.createOrFold<arith::AddIOp>(
                      loc, b.createOrFold<arith::MulIOp>(loc, flat, size),
                      coords[axis]);
                }
                mapped[group] = flat;
              }
            }
            coords = std::move(mapped);
          }
        }
        return coords;
      };
      Value onlineLhs, quantizeInBank;
      auto stage = [&](Operand input, bool rhs) -> Value {
        Value source =
            input.quant ? input.quant.getInput() : input.decode.getInput();
        auto sourceType = cast<RankedTensorType>(source.getType());
        Type element = sourceType.getElementType();
        int64_t rowCount = rhs ? n : outputType.getDimSize(batched ? 1 : 0);
        int64_t storedRows = rhs ? paddedN : rowCount;
        Value extent = rhs ? nSize : mSize;
        Value storedExtent = rhs ? paddedSize : mSize;
        SmallVector<Value> dynamic;
        if (chunks < 0)
          dynamic.push_back(chunkCount);
        if (storedRows < 0)
          dynamic.push_back(storedExtent);
        Value storage;
        // Reshaping non-K dimensions and moving K past them preserves the
        // flattened row order. Such views need no physical transpose.
        int64_t axis = cast<RankedTensorType>(input.value.getType()).getRank() -
                       (rhs && !transposedRhs ? 2 : 1);
        bool ordered = true;
        for (Operation *view : input.views) {
          if (isa<tensor::ExtractSliceOp>(view)) {
            ordered = false;
            break;
          }
          if (auto transpose = dyn_cast<linalg::TransposeOp>(view)) {
            axis = transpose.getPermutation()[axis];
            auto shape = cast<RankedTensorType>(transpose.getInput().getType());
            int64_t previous = -1;
            for (int64_t dim : transpose.getPermutation())
              if (dim != axis && shape.getDimSize(dim) != 1) {
                ordered &= dim > previous;
                previous = dim;
              }
          } else {
            auto groups = isa<tensor::CollapseShapeOp>(view)
                              ? cast<tensor::CollapseShapeOp>(view)
                                    .getReassociationIndices()
                              : cast<tensor::ExpandShapeOp>(view)
                                    .getReassociationIndices();
            if (isa<tensor::CollapseShapeOp>(view)) {
              auto original =
                  cast<RankedTensorType>(view->getOperand(0).getType());
              SmallVector<int64_t> dimensions;
              for (int64_t dim : groups[axis])
                if (original.getDimSize(dim) != 1)
                  dimensions.push_back(dim);
              if (dimensions.size() != 1) {
                ordered = false;
                break;
              }
              axis = dimensions[0];
            } else {
              auto group = llvm::find_if(groups, [&](const auto &indices) {
                return llvm::is_contained(indices, axis);
              });
              auto expanded =
                  cast<RankedTensorType>(view->getResult(0).getType());
              if (llvm::any_of(*group, [&](int64_t dim) {
                    return dim != axis && expanded.getDimSize(dim) != 1;
                  })) {
                ordered = false;
                break;
              }
              axis = group - groups.begin();
            }
          }
        }
        if (!input.quant) {
          if (!rhs && ordered && chunks == 1 && k + k / 32 == width &&
              axis == sourceType.getRank() - 1 &&
              int64_t(input.decode.getAxis()) == axis &&
              sourceType.getShape().back() == width) {
            // The compact row already matches the bank layout, including its
            // exponent bytes. Keep the original read-only storage.
            Value memory = buffer(b, loc, source);
            memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
            OpFoldResult rows = rowCount < 0
                                    ? OpFoldResult(extent)
                                    : OpFoldResult(b.getIndexAttr(rowCount));
            OpFoldResult stride =
                rowCount < 0 ? OpFoldResult(b.createOrFold<arith::MulIOp>(
                                   loc, extent, index(b, loc, width)))
                             : OpFoldResult(b.getIndexAttr(rowCount * width));
            return memref::ReinterpretCastOp::create(
                b, loc,
                MemRefType::get(
                    {batch, 1, rowCount, width}, element,
                    StridedLayoutAttr::get(
                        b.getContext(), ShapedType::kDynamic,
                        {rowCount < 0 ? ShapedType::kDynamic : rowCount * width,
                         rowCount < 0 ? ShapedType::kDynamic : rowCount * width,
                         width, 1})),
                memory,
                memref::ExtractStridedMetadataOp::create(b, loc, memory)
                    .getOffset(),
                ArrayRef<OpFoldResult>{b.getIndexAttr(batch), b.getIndexAttr(1),
                                       rows, b.getIndexAttr(width)},
                ArrayRef<OpFoldResult>{stride, stride, b.getIndexAttr(width),
                                       b.getIndexAttr(1)});
          }
          if (ordered && chunks == 1 && k <= 512 &&
              axis == sourceType.getRank() - 1 &&
              int64_t(input.decode.getAxis()) == axis &&
              sourceType.getShape().back() == k + k / 32) {
            // Gather compact codes and exponent bytes with DMA. Bank padding
            // and padded output columns are initialized once by bank clear.
            Value blocks = memref::AllocOp::create(
                b, loc,
                MemRefType::get({batch, chunks, storedRows, width}, element),
                dynamic);
            Value memory = buffer(b, loc, source);
            memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
            ReassociationIndices axes(sourceType.getRank());
            std::iota(axes.begin(), axes.end(), 0);
            Value flat = memref::CollapseShapeOp::create(
                b, loc, memory, ArrayRef<ReassociationIndices>{axes});
            int64_t tile = std::min<int64_t>(
                fit, 1 + ((target.dmaMaxRanges - 1) * 4096 - chunk - 64) /
                             (k + k / 32));
            if (!llvm::isPowerOf2_64(k))
              tile = 1;
            loops(b, loc, {index(b, loc, batch)}, true, [&](ValueRange bt) {
              for (bool padding : {false, true}) {
                Value total =
                    padding
                        ? arith::SubIOp::create(b, loc, storedExtent, extent)
                              .getResult()
                        : extent;
                rowTiles(
                    b, loc, total, tile, 1, [&](Value start, int64_t count) {
                      Value first =
                          padding ? arith::AddIOp::create(b, loc, extent, start)
                                        .getResult()
                                  : start;
                      Value bank =
                          BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                      bank = BankClearOp::create(b, loc, b.getI64Type(), bank,
                                                 1, 1);
                      if (!padding) {
                        Value row = arith::AddIOp::create(
                            b, loc,
                            arith::MulIOp::create(b, loc, bt[0], extent),
                            first);
                        Value offset = arith::MulIOp::create(
                            b, loc, row, index(b, loc, k + k / 32));
                        if (!llvm::isPowerOf2_64(k)) {
                          for (int part = 0; part < 2; ++part) {
                            int64_t bytes = part ? k / 32 : k;
                            Value from = arith::AddIOp::create(
                                b, loc, offset, index(b, loc, part ? k : 0));
                            Value slice = memref::SubViewOp::create(
                                b, loc, flat, ArrayRef<OpFoldResult>{from},
                                ArrayRef<OpFoldResult>{b.getIndexAttr(bytes)},
                                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
                            bank = BankMvinStridedOp::create(
                                b, loc, b.getI64Type(), slice, bank,
                                index(b, loc, part ? k / 16 : 0),
                                index(b, loc, part ? 1 : k / 16),
                                index(b, loc, 16), 0, 1,
                                index(b, loc, part ? bytes : 16),
                                index(b, loc, 0));
                          }
                        } else {
                          int64_t codeBytes = std::min<int64_t>(k, 256);
                          for (int part = 0; part <= k / codeBytes; ++part) {
                            bool exponent = part == k / codeBytes;
                            int64_t firstByte = exponent ? k : part * codeBytes;
                            int64_t bytes = exponent ? k / 32 : codeBytes;
                            Value from = arith::AddIOp::create(
                                b, loc, offset, index(b, loc, firstByte));
                            Value slice = memref::SubViewOp::create(
                                b, loc, flat, ArrayRef<OpFoldResult>{from},
                                ArrayRef<OpFoldResult>{b.getIndexAttr(bytes)},
                                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
                            bank = BankMvinStridedOp::create(
                                b, loc, b.getI64Type(), slice, bank,
                                index(b, loc, firstByte / 16),
                                index(b, loc,
                                      count * (exponent ? 1 : bytes / 16)),
                                index(b, loc, k + k / 32), 0, rowDepth,
                                index(b, loc, exponent ? bytes : 16),
                                index(b, loc,
                                      exponent ? 0
                                               : llvm::Log2_64(bytes / 16)));
                          }
                        }
                      }
                      bank = BankMvoutOp::create(b, loc, b.getI64Type(),
                                                 rowsOf(b, loc, blocks, bt[0],
                                                        index(b, loc, 0), first,
                                                        count, width),
                                                 bank, count * rowDepth, 1);
                      BankReleaseOp::create(b, loc, bank);
                    });
              }
            });
            return blocks;
          }
          if (rhs && ordered && chunk <= 512 &&
              axis == sourceType.getRank() - 2 &&
              int64_t(input.decode.getAxis()) == axis &&
              (k < 0 || sourceType.getDimSize(axis) == k + k / 32)) {
            Value blocks = memref::AllocOp::create(
                b, loc,
                MemRefType::get({batch, chunks, storedRows, width}, element),
                dynamic);
            Value memory = buffer(b, loc, source);
            memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
            ReassociationIndices axes(sourceType.getRank());
            std::iota(axes.begin(), axes.end(), 0);
            Value flat = memref::CollapseShapeOp::create(
                b, loc, memory, ArrayRef<ReassociationIndices>{axes});
            Value packedFlat = memref::CollapseShapeOp::create(
                b, loc, blocks, ArrayRef<ReassociationIndices>{{0, 1, 2, 3}});
            Value stride =
                tensor::DimOp::create(b, loc, source, sourceType.getRank() - 1);
            Value packedExtent = tensor::DimOp::create(b, loc, source, axis);
            Value sourceK = arith::MulIOp::create(
                b, loc,
                arith::DivUIOp::create(b, loc, packedExtent, index(b, loc, 33)),
                index(b, loc, 32));
            Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            bank = BankClearOp::create(b, loc, b.getI64Type(), bank, 1, 1);
            loops(b, loc, {index(b, loc, batch), chunkCount}, true,
                  [&](ValueRange bt) {
                    int64_t zeroRows = std::min<int64_t>(
                        fit, ((target.dmaMaxRanges - 1) * 4096 - 64) / width);
                    rowTiles(b, loc, storedExtent, zeroRows, 1,
                             [&](Value first, int64_t count) {
                               BankMvoutOp::create(b, loc, b.getI64Type(),
                                                   rowsOf(b, loc, blocks, bt[0],
                                                          bt[1], first, count,
                                                          width),
                                                   bank, count * rowDepth, 1);
                             });
                  });
            for (int part = 0; part < 2; ++part) {
              int64_t count = part ? chunk / 32 : chunk;
              Value dmaRows = index(b, loc, 1);
              for (int64_t height = 2;
                   height <= count && count % height == 0 && height <= depth;
                   height *= 2) {
                int64_t maxStride = std::min<int64_t>(
                    4095,
                    ((target.dmaMaxRanges - 1) * 4096 - 65) / (height - 1));
                Value fits =
                    arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule,
                                          stride, index(b, loc, maxStride));
                dmaRows = arith::SelectOp::create(
                    b, loc, fits, index(b, loc, height), dmaRows);
              }
              Value loads =
                  arith::DivUIOp::create(b, loc, index(b, loc, count), dmaRows);
              Value multi = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::ugt, dmaRows, index(b, loc, 1));
              Value memoryStride = arith::SelectOp::create(
                  b, loc, multi, stride, index(b, loc, 1));
              loops(b, loc, {index(b, loc, batch), chunkCount, extent, loads},
                    true, [&](ValueRange at) {
                      Value first =
                          arith::MulIOp::create(b, loc, at[3], dmaRows);
                      auto coords = coordinates(input, rhs, at[0], at[2],
                                                index(b, loc, 0));
                      coords[axis] = arith::AddIOp::create(
                          b, loc, part ? sourceK : index(b, loc, 0),
                          arith::AddIOp::create(
                              b, loc,
                              arith::MulIOp::create(b, loc, at[1],
                                                    index(b, loc, count)),
                              first));
                      Value from = index(b, loc, 0);
                      for (auto [dim, coord] : llvm::enumerate(coords))
                        from = arith::AddIOp::create(
                            b, loc,
                            arith::MulIOp::create(
                                b, loc, from,
                                tensor::DimOp::create(b, loc, source, dim)),
                            coord);
                      Value row = arith::AddIOp::create(
                          b, loc,
                          arith::MulIOp::create(
                              b, loc,
                              arith::AddIOp::create(
                                  b, loc,
                                  arith::MulIOp::create(b, loc, at[0],
                                                        chunkCount),
                                  at[1]),
                              storedExtent),
                          at[2]);
                      Value to = arith::AddIOp::create(
                          b, loc,
                          arith::MulIOp::create(b, loc, row,
                                                index(b, loc, width)),
                          arith::AddIOp::create(
                              b, loc, first, index(b, loc, part ? chunk : 0)));
                      Value sourceByte = memref::SubViewOp::create(
                          b, loc, flat, ArrayRef<OpFoldResult>{from},
                          ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
                          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
                      Value targetByte = memref::SubViewOp::create(
                          b, loc, packedFlat, ArrayRef<OpFoldResult>{to},
                          ArrayRef<OpFoldResult>{dmaRows},
                          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
                      Value loaded = BankMvinStridedOp::create(
                          b, loc, b.getI64Type(), sourceByte, bank,
                          index(b, loc, 0), dmaRows, memoryStride, 0, 1,
                          index(b, loc, 1), index(b, loc, 0));
                      BankMvoutStridedOp::create(
                          b, loc, b.getI64Type(), targetByte, loaded,
                          index(b, loc, 0), dmaRows, index(b, loc, 1), 0, 1,
                          index(b, loc, 1), index(b, loc, 0));
                    });
            }
            BankReleaseOp::create(b, loc, bank);
            return blocks;
          }
          // Shared quantization returns compact codes and exponents. Repack
          // their bytes through the operand's views without dequantizing them.
          Value packedExtent =
              tensor::DimOp::create(b, loc, source, input.decode.getAxis());
          Value sourceK = arith::MulIOp::create(
              b, loc,
              arith::DivUIOp::create(b, loc, packedExtent, index(b, loc, 33)),
              index(b, loc, 32));
          Value blocks = memref::AllocOp::create(
              b, loc,
              MemRefType::get({batch, chunks, storedRows, width}, element),
              dynamic);
          linalg::FillOp::create(
              b, loc, ValueRange{arith::ConstantIntOp::create(b, loc, 0, 8)},
              blocks);
          loops(
              b, loc, {index(b, loc, batch), chunkCount, extent}, false,
              [&](ValueRange at) {
                Value start =
                    arith::MulIOp::create(b, loc, at[1], index(b, loc, chunk));
                for (bool exponent : {false, true})
                  loops(b, loc, {index(b, loc, exponent ? chunk / 32 : chunk)},
                        false, [&](ValueRange offset) {
                          Value inner =
                              exponent
                                  ? arith::MulIOp::create(b, loc, offset[0],
                                                          index(b, loc, 32))
                                        .getResult()
                                  : offset[0];
                          Value reduction =
                              arith::AddIOp::create(b, loc, start, inner);
                          auto coords =
                              coordinates(input, rhs, at[0], at[2], reduction);
                          if (exponent)
                            coords[input.decode.getAxis()] =
                                arith::AddIOp::create(
                                    b, loc, sourceK,
                                    arith::DivUIOp::create(
                                        b, loc, coords[input.decode.getAxis()],
                                        index(b, loc, 32)));
                          Value to =
                              exponent
                                  ? arith::AddIOp::create(b, loc, offset[0],
                                                          index(b, loc, chunk))
                                        .getResult()
                                  : offset[0];
                          Value byte =
                              tensor::ExtractOp::create(b, loc, source, coords);
                          memref::StoreOp::create(
                              b, loc, byte, blocks,
                              ValueRange{at[0], at[1], at[2], to});
                        });
              });
          return blocks;
        }
        bool contiguous = k >= 0 && chunks == 1 && ordered &&
                          axis == sourceType.getRank() - 1 &&
                          (int64_t(input.quant.getAxis()) == axis) &&
                          sourceType.getShape().back() == k;
        Value sourceRows;
        if (contiguous) {
          // View the original rows without materializing their layout chain.
          Value sourceBuffer = buffer(b, loc, source);
          sourceBuffer.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
          auto layout = StridedLayoutAttr::get(
              b.getContext(), ShapedType::kDynamic,
              {rowCount < 0 ? ShapedType::kDynamic : rowCount * k, chunk, k,
               1});
          sourceRows = memref::ReinterpretCastOp::create(
              b, loc,
              MemRefType::get({batch, chunks, rowCount, chunk}, element,
                              layout),
              sourceBuffer,
              memref::ExtractStridedMetadataOp::create(b, loc, sourceBuffer)
                  .getOffset(),
              ArrayRef<OpFoldResult>{
                  b.getIndexAttr(batch), b.getIndexAttr(chunks),
                  rowCount < 0 ? OpFoldResult(extent)
                               : OpFoldResult(b.getIndexAttr(rowCount)),
                  b.getIndexAttr(chunk)},
              ArrayRef<OpFoldResult>{
                  rowCount < 0
                      ? OpFoldResult(arith::MulIOp::create(b, loc, extent,
                                                           index(b, loc, k))
                                         .getResult())
                      : OpFoldResult(b.getIndexAttr(rowCount * k)),
                  b.getIndexAttr(chunk), b.getIndexAttr(k), b.getIndexAttr(1)});
        }
        if (contiguous) {
          storage = sourceRows;
        } else {
          // The host gathers a transposed or padded operand once before
          // recording; padding rows and K are zeros.
          storage = memref::AllocOp::create(
              b, loc,
              MemRefType::get({batch, chunks, storedRows, chunk}, element),
              dynamic);
          // Gathering writes every K position of each valid row, including
          // K padding. Only the extra output rows need separate zeroing.
          if (rhs && padded) {
            Value tail =
                b.createOrFold<arith::SubIOp>(loc, storedExtent, extent);
            Value padding = memref::SubViewOp::create(
                b, loc, storage,
                ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(0),
                                       extent, b.getIndexAttr(0)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(batch), chunkCount, tail,
                                       b.getIndexAttr(chunk)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                       b.getIndexAttr(1), b.getIndexAttr(1)});
            linalg::FillOp::create(b, loc,
                                   ValueRange{arith::ConstantOp::create(
                                       b, loc, b.getZeroAttr(element))},
                                   padding);
          }
          {
            Value sourceK =
                tensor::DimOp::create(b, loc, source, input.quant.getAxis());
            Value transposedChunks = index(b, loc, 0);
            int64_t rowWidth = 1;
            for (int64_t dim = axis + 1; dim < sourceType.getRank(); ++dim)
              rowWidth = rowWidth < 0 || sourceType.isDynamicDim(dim)
                             ? ShapedType::kDynamic
                             : rowWidth * sourceType.getDimSize(dim);
            // A head can occupy only part of a source row. Use the complete
            // source row stride when gathering four adjacent head columns.
            if (rhs && ordered && axis < sourceType.getRank() - 1 &&
                int64_t(input.quant.getAxis()) == axis && n > 0 && n % 4 == 0 &&
                sourceType.getShape().back() >= 4 &&
                sourceType.getShape().back() % 4 == 0 && rowWidth > 0 &&
                rowWidth / 4 <= 4095 &&
                chunk <= std::min<int64_t>(depth, 1023)) {
              transposedChunks =
                  arith::DivUIOp::create(b, loc, sourceK, index(b, loc, chunk));
              Value memory = buffer(b, loc, source);
              memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
              loops(b, loc,
                    {index(b, loc, batch), transposedChunks,
                     index(b, loc, n / 4)},
                    true, [&](ValueRange at) {
                      Value row = arith::MulIOp::create(b, loc, at[2],
                                                        index(b, loc, 4));
                      Value reduction = arith::MulIOp::create(
                          b, loc, at[1], index(b, loc, chunk));
                      SmallVector<OpFoldResult> offsets;
                      for (Value coord :
                           coordinates(input, rhs, at[0], row, reduction))
                        offsets.push_back(coord);
                      SmallVector<OpFoldResult> sizes(sourceType.getRank(),
                                                      b.getIndexAttr(1));
                      SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                                        b.getIndexAttr(1));
                      sizes[axis] = b.getIndexAttr(chunk);
                      sizes.back() = b.getIndexAttr(4);
                      Value slice = memref::SubViewOp::create(
                          b, loc, memory, offsets, sizes, strides);
                      Value in =
                          BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                      Value out =
                          BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                      in = BankMvinOp::create(b, loc, b.getI64Type(), slice, in,
                                              chunk, rowWidth / 4);
                      out = ::buddy::transpose_ball::BankTransposeF32Op::create(
                          b, loc, b.getI64Type(), in, out, chunk, 4);
                      out = BankMvoutOp::create(
                          b, loc, b.getI64Type(),
                          rowsOf(b, loc, storage, at[0], at[1], row, 4, chunk),
                          out, chunk, 1);
                      BankReleaseOp::create(b, loc, in);
                      BankReleaseOp::create(b, loc, out);
                    });
              if (rowWidth * 4 <= 4095) {
                OpBuilder::InsertionGuard tailInsertion(b);
                Value tail = arith::RemUIOp::create(b, loc, sourceK,
                                                    index(b, loc, chunk));
                Value nonempty = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::ugt, tail, index(b, loc, 0));
                auto active = scf::IfOp::create(b, loc, nonempty, false);
                b.setInsertionPointToStart(&active.getThenRegion().front());
                Value in = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                in = BankClearOp::create(b, loc, b.getI64Type(), in, 1, 1);
                Value out = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                // Every column overwrites the valid prefix. The same cleared
                // bank supplies the zero K tail for all columns and batches.
                loops(b, loc, {index(b, loc, batch), index(b, loc, n / 4)},
                      true, [&](ValueRange at) {
                        Value row = arith::MulIOp::create(b, loc, at[1],
                                                          index(b, loc, 4));
                        Value reduction = arith::MulIOp::create(
                            b, loc, transposedChunks, index(b, loc, chunk));
                        SmallVector<OpFoldResult> offsets;
                        for (Value coord :
                             coordinates(input, rhs, at[0], row, reduction))
                          offsets.push_back(coord);
                        SmallVector<OpFoldResult> sizes(sourceType.getRank(),
                                                        b.getIndexAttr(1));
                        SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                                          b.getIndexAttr(1));
                        sizes[axis] = tail;
                        sizes.back() = b.getIndexAttr(4);
                        Value memoryRow = memref::SubViewOp::create(
                            b, loc, memory, offsets, sizes, strides);
                        Value loaded = BankMvinStridedOp::create(
                            b, loc, b.getI64Type(), memoryRow, in,
                            index(b, loc, 0), tail, index(b, loc, rowWidth * 4),
                            0, 1, index(b, loc, 16), index(b, loc, 0));
                        Value result =
                            ::buddy::transpose_ball::BankTransposeF32Op::create(
                                b, loc, b.getI64Type(), loaded, out, chunk, 4);
                        BankMvoutOp::create(b, loc, b.getI64Type(),
                                            rowsOf(b, loc, storage, at[0],
                                                   transposedChunks, row, 4,
                                                   chunk),
                                            result, chunk, 1);
                      });
                BankReleaseOp::create(b, loc, in);
                BankReleaseOp::create(b, loc, out);
                transposedChunks = chunkCount;
              }
            }
            if (ordered && axis == sourceType.getRank() - 1 &&
                int64_t(input.quant.getAxis()) == axis) {
              Value memory = buffer(b, loc, source);
              memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
              ReassociationIndices axes;
              for (int64_t i = 0; i < sourceType.getRank(); ++i)
                axes.push_back(i);
              Value flat = memref::CollapseShapeOp::create(
                  b, loc, memory, ArrayRef<ReassociationIndices>{axes});
              transposedChunks =
                  arith::DivUIOp::create(b, loc, sourceK, index(b, loc, chunk));
              Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
              // Whole K blocks need no padding or host element copies.
              loops(b, loc, {index(b, loc, batch), transposedChunks, extent},
                    true, [&](ValueRange at) {
                      Value row = arith::AddIOp::create(
                          b, loc, arith::MulIOp::create(b, loc, at[0], extent),
                          at[2]);
                      Value offset = arith::AddIOp::create(
                          b, loc, arith::MulIOp::create(b, loc, row, sourceK),
                          arith::MulIOp::create(b, loc, at[1],
                                                index(b, loc, chunk)));
                      Value block = memref::SubViewOp::create(
                          b, loc, flat, ArrayRef<OpFoldResult>{offset},
                          ArrayRef<OpFoldResult>{b.getIndexAttr(chunk)},
                          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
                      Value loaded = BankMvinOp::create(
                          b, loc, b.getI64Type(), block, bank, chunk / 4, 1);
                      BankMvoutOp::create(b, loc, b.getI64Type(),
                                          rowsOf(b, loc, storage, at[0], at[1],
                                                 at[2], 1, chunk),
                                          loaded, chunk / 4, 1);
                    });
              BankReleaseOp::create(b, loc, bank);
            }
            if (rhs && ordered && n < 0 && k > 0 && chunks == 1 &&
                axis == sourceType.getRank() - 2 &&
                int64_t(input.quant.getAxis()) == axis &&
                chunk <= std::min<int64_t>(depth, 1023)) {
              Value memory = buffer(b, loc, source);
              memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
              Value stride = tensor::DimOp::create(b, loc, source,
                                                   sourceType.getRank() - 1);
              Value groups =
                  arith::DivUIOp::create(b, loc, nSize, index(b, loc, 4));
              Value tail =
                  arith::RemUIOp::create(b, loc, nSize, index(b, loc, 4));
              Value dmaRows = index(b, loc, 1);
              for (int64_t height = 2; height <= chunk && chunk % height == 0;
                   height *= 2) {
                int64_t maxStride = std::min<int64_t>(
                    1023, ((target.dmaMaxRanges - 1) * 4096 - 80) /
                              (4 * (height - 1)));
                Value fits =
                    arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule,
                                          stride, index(b, loc, maxStride));
                dmaRows = arith::SelectOp::create(
                    b, loc, fits, index(b, loc, height), dmaRows);
              }
              Value loads =
                  arith::DivUIOp::create(b, loc, index(b, loc, chunk), dmaRows);
              Value multi = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::ugt, dmaRows, index(b, loc, 1));
              Value memoryStride = arith::SelectOp::create(
                  b, loc, multi,
                  arith::MulIOp::create(b, loc, stride, index(b, loc, 4)),
                  index(b, loc, 16));
              for (bool partial : {false, true}) {
                OpBuilder::InsertionGuard insertion(b);
                if (partial) {
                  Value nonempty =
                      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                            tail, index(b, loc, 0));
                  auto active = scf::IfOp::create(b, loc, nonempty, false);
                  b.setInsertionPointToStart(&active.getThenRegion().front());
                }
                Value in = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                Value out = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                if (partial)
                  in = BankClearOp::create(b, loc, b.getI64Type(), in, 1, 1);
                Value columns = partial ? tail : index(b, loc, 4);
                Value bytes =
                    arith::MulIOp::create(b, loc, columns, index(b, loc, 4));
                loops(
                    b, loc,
                    {index(b, loc, batch), partial ? index(b, loc, 1) : groups},
                    true, [&](ValueRange at) {
                      Value row = arith::MulIOp::create(
                          b, loc, partial ? groups : at[1], index(b, loc, 4));
                      loops(b, loc, {loads}, true, [&](ValueRange tile) {
                        Value first =
                            arith::MulIOp::create(b, loc, tile[0], dmaRows);
                        SmallVector<OpFoldResult> offsets;
                        for (Value coord :
                             coordinates(input, rhs, at[0], row, first))
                          offsets.push_back(coord);
                        SmallVector<OpFoldResult> sizes(sourceType.getRank(),
                                                        b.getIndexAttr(1));
                        SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                                          b.getIndexAttr(1));
                        sizes[axis] = dmaRows;
                        sizes.back() = columns;
                        Value part = memref::SubViewOp::create(
                            b, loc, memory, offsets, sizes, strides);
                        BankMvinStridedOp::create(
                            b, loc, b.getI64Type(), part, in, first, dmaRows,
                            memoryStride, 0, 1, bytes, index(b, loc, 0));
                      });
                      Value result =
                          ::buddy::transpose_ball::BankTransposeF32Op::create(
                              b, loc, b.getI64Type(), in, out, chunk, 4);
                      BankMvoutOp::create(b, loc, b.getI64Type(),
                                          rowsOf(b, loc, storage, at[0],
                                                 index(b, loc, 0), row, 4,
                                                 chunk),
                                          result, chunk, 1);
                    });
                BankReleaseOp::create(b, loc, in);
                BankReleaseOp::create(b, loc, out);
              }
              transposedChunks = chunkCount;
            }
            Value remaining =
                arith::SubIOp::create(b, loc, chunkCount, transposedChunks);
            loops(
                b, loc,
                {index(b, loc, batch), remaining, extent, index(b, loc, chunk)},
                false, [&](ValueRange at) {
                  Value block =
                      arith::AddIOp::create(b, loc, at[1], transposedChunks);
                  Value coordinate = arith::AddIOp::create(
                      b, loc,
                      arith::MulIOp::create(b, loc, block,
                                            index(b, loc, chunk)),
                      at[3]);
                  SmallVector<Value> coords =
                      coordinates(input, rhs, at[0], at[2], coordinate);
                  Value valid = arith::CmpIOp::create(
                      b, loc, arith::CmpIPredicate::ult,
                      coords[input.quant.getAxis()], sourceK);
                  auto choose = scf::IfOp::create(b, loc, TypeRange{element},
                                                  valid, true);
                  {
                    OpBuilder::InsertionGuard guard(b);
                    b.setInsertionPointToStart(&choose.getThenRegion().front());
                    scf::YieldOp::create(b, loc,
                                         ValueRange{tensor::ExtractOp::create(
                                             b, loc, source, coords)});
                    b.setInsertionPointToStart(&choose.getElseRegion().front());
                    scf::YieldOp::create(b, loc,
                                         ValueRange{arith::ConstantOp::create(
                                             b, loc, b.getZeroAttr(element))});
                  }
                  memref::StoreOp::create(
                      b, loc, choose.getResult(0), storage,
                      ValueRange{at[0], block, at[2], at[3]});
                });
          }
        }
        // Quantized rows go straight where the product reads them: as many
        // rows per instruction as the input and output banks hold.
        Value quantized = memref::AllocOp::create(
            b, loc,
            MemRefType::get({batch, chunks, storedRows, width}, b.getI8Type()),
            dynamic);
        OpBuilder::InsertionGuard insertion(b);
        if (!rhs && chunks == 1 && !residentWeights) {
          // Small activation matrices go straight from quantization to matmul.
          // Keep larger row tiles to preserve their weight reuse.
          onlineLhs = storage;
          // Row tiles are `rows` high, then halving powers of two (rowTiles).
          // A tile no higher than quantRows quantizes in its own bank; spill
          // only when M reaches the first tile height above quantRows.
          int64_t spillRows = rows > quantRows ? rows : 0;
          for (int64_t part = 1; spillRows && part < rows; part *= 2)
            if (part > quantRows) {
              spillRows = part;
              break;
            }
          quantizeInBank =
              spillRows ? Value(arith::CmpIOp::create(
                              b, loc, arith::CmpIPredicate::ult, mSize,
                              index(b, loc, spillRows)))
                        : Value(arith::ConstantIntOp::create(b, loc, 1, 1));
          Value spill =
              arith::XOrIOp::create(b, loc, quantizeInBank,
                                    arith::ConstantIntOp::create(b, loc, 1, 1));
          auto prepare = scf::IfOp::create(b, loc, spill, false);
          b.setInsertionPointToStart(&prepare.getThenRegion().front());
        }
        loops(
            b, loc, {index(b, loc, batch), chunkCount}, true,
            [&](ValueRange at) {
              rowTiles(
                  b, loc, contiguous ? extent : storedExtent, quantRows, 1,
                  [&](Value first, int64_t count) {
                    Value in =
                        BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                    Value out =
                        BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                    in = BankMvinOp::create(b, loc, b.getI64Type(),
                                            rowsOf(b, loc, storage, at[0],
                                                   at[1], first, count, chunk),
                                            in, count * chunk / 4, 1);
                    out =
                        ::buddy::mxint8quantize_ball::BankQuantMxint8Op::create(
                            b, loc, b.getI64Type(), in, out, count, chunk);
                    out =
                        BankMvoutOp::create(b, loc, b.getI64Type(),
                                            rowsOf(b, loc, quantized, at[0],
                                                   at[1], first, count, width),
                                            out, count * rowDepth, 1);
                    BankReleaseOp::create(b, loc, in);
                    BankReleaseOp::create(b, loc, out);
                  });
            });
        if (contiguous && rhs && padded) {
          // Padding is outside the logical operand. Zero its packed rows
          // without copying or quantizing a padded FP32 matrix on the host.
          Value tail = arith::SubIOp::create(b, loc, storedExtent, extent);
          loops(b, loc, {index(b, loc, batch)}, true, [&](ValueRange bt) {
            rowTiles(b, loc, tail, fit, 1, [&](Value start, int64_t count) {
              Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
              bank = BankClearOp::create(b, loc, b.getI64Type(), bank, 1, 1);
              Value first = arith::AddIOp::create(b, loc, extent, start);
              bank = BankMvoutOp::create(b, loc, b.getI64Type(),
                                         rowsOf(b, loc, quantized, bt[0],
                                                index(b, loc, 0), first, count,
                                                width),
                                         bank, count * rowDepth, 1);
              BankReleaseOp::create(b, loc, bank);
            });
          });
        }
        return quantized;
      };
      // A packed operand in place as [1][1][row][width].
      auto view = [&](Value source) -> Value {
        auto sourceType = cast<RankedTensorType>(source.getType());
        auto columns = op->getAttrOfType<IntegerAttr>("hart.columns");
        Value memory;
        if (columns && sourceType.getRank() == 3) {
          int64_t row = sourceType.getDimSize(2);
          auto layout =
              StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                                     {columns.getInt() * row, row, 1});
          memory = bufferization::ToBufferOp::create(
              b, loc,
              MemRefType::get(sourceType.getShape(),
                              sourceType.getElementType(), layout),
              source);
        } else {
          memory = buffer(b, loc, source);
        }
        memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
        auto type = cast<MemRefType>(memory.getType());
        if (type.getRank() == 3) {
          int64_t height = type.getDimSize(1);
          auto strides = type.getStridesAndOffset().first;
          auto metadata =
              memref::ExtractStridedMetadataOp::create(b, loc, memory);
          auto stride = [&](int64_t axis) -> OpFoldResult {
            return ShapedType::isDynamic(strides[axis])
                       ? OpFoldResult(metadata.getStrides()[axis])
                       : OpFoldResult(b.getIndexAttr(strides[axis]));
          };
          int64_t batchStride = ShapedType::isDynamic(strides[0])
                                    ? ShapedType::kDynamic
                                    : chunks * strides[0];
          OpFoldResult batch =
              ShapedType::isDynamic(batchStride)
                  ? OpFoldResult(arith::MulIOp::create(b, loc,
                                                       metadata.getStrides()[0],
                                                       index(b, loc, chunks))
                                     .getResult())
                  : OpFoldResult(b.getIndexAttr(batchStride));
          return memref::ReinterpretCastOp::create(
              b, loc,
              MemRefType::get({1, chunks, height, width}, type.getElementType(),
                              StridedLayoutAttr::get(b.getContext(),
                                                     ShapedType::kDynamic,
                                                     {batchStride, strides[0],
                                                      strides[1], strides[2]})),
              memory, metadata.getOffset(),
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(chunks),
                                     b.getIndexAttr(height),
                                     b.getIndexAttr(width)},
              ArrayRef<OpFoldResult>{batch, stride(0), stride(1), stride(2)});
        }
        int64_t height = type.getDimSize(0);
        Value extent = memref::DimOp::create(b, loc, memory, 0);
        if (chunks > 1) {
          // Keep each block's codes and exponents together in the Ball layout.
          SmallVector<Value> dynamic;
          if (height < 0)
            dynamic.push_back(extent);
          Value blocks = memref::AllocOp::create(
              b, loc,
              MemRefType::get({1, chunks, height, width}, b.getI8Type()),
              dynamic);
          linalg::FillOp::create(
              b, loc, ValueRange{arith::ConstantIntOp::create(b, loc, 0, 8)},
              blocks);
          loops(b, loc, {index(b, loc, chunks), extent}, false,
                [&](ValueRange at) {
                  for (bool exponent : {false, true}) {
                    int64_t count = exponent ? chunk / 32 : chunk;
                    Value start = arith::MulIOp::create(b, loc, at[0],
                                                        index(b, loc, count));
                    if (exponent)
                      start = arith::AddIOp::create(b, loc, start,
                                                    index(b, loc, k));
                    loops(b, loc, {index(b, loc, count)}, false,
                          [&](ValueRange offset) {
                            Value from =
                                arith::AddIOp::create(b, loc, start, offset[0]);
                            Value to = exponent ? arith::AddIOp::create(
                                                      b, loc, offset[0],
                                                      index(b, loc, chunk))
                                                      .getResult()
                                                : offset[0];
                            Value byte = memref::LoadOp::create(
                                b, loc, memory, ValueRange{at[1], from});
                            memref::StoreOp::create(
                                b, loc, byte, blocks,
                                ValueRange{index(b, loc, 0), at[0], at[1], to});
                          });
                  }
                });
          return blocks;
        }
        auto layout = StridedLayoutAttr::get(
            b.getContext(), ShapedType::kDynamic,
            {height < 0 ? ShapedType::kDynamic : height * width,
             height < 0 ? ShapedType::kDynamic : height * width, width, 1});
        OpFoldResult rowsSize = height < 0
                                    ? OpFoldResult(extent)
                                    : OpFoldResult(b.getIndexAttr(height));
        OpFoldResult plane =
            height < 0 ? OpFoldResult(arith::MulIOp::create(
                                          b, loc, extent, index(b, loc, width))
                                          .getResult())
                       : OpFoldResult(b.getIndexAttr(height * width));
        return memref::ReinterpretCastOp::create(
            b, loc,
            MemRefType::get({1, 1, height, width}, type.getElementType(),
                            layout),
            memory,
            memref::ExtractStridedMetadataOp::create(b, loc, memory)
                .getOffset(),
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                   rowsSize, b.getIndexAttr(width)},
            ArrayRef<OpFoldResult>{plane, plane, b.getIndexAttr(width),
                                   b.getIndexAttr(1)});
      };
      // Split already quantized rows directly into private banks. Their codes
      // and exponents are separate contiguous ranges in the source row.
      bool directInput = aSource && wSource && !batched && !padded &&
                         chunks > 1 && chunk % 512 == 0 && !onlineLhs &&
                         !matchPattern(mSize, m_Zero());
      int64_t sourceWidth = directInput ? packedRows(k) * 16 : 0;
      Value lhs = directInput ? buffer(b, loc, aSource)
                  : aSource   ? view(aSource)
                              : stage(a, false);
      if (directInput)
        lhs.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      Value rhs = wSource ? view(wSource) : stage(w, true);

      // Unpadded tiles write final DDR rows directly. Padded column tails
      // retain scratch storage until their logical extent is trimmed.
      auto outputStrides =
          cast<MemRefType>(output.getType()).getStridesAndOffset().first;
      int64_t outputRowStride = outputStrides[outputStrides.size() - 2];
      bool whole = !padded && columns == n && outputStrides.back() == 1 &&
                   outputRowStride == n;
      // Row-wise DMA encodes its first bank row in ten bits. Bound the
      // compute tile as well as its SRAM footprint before selecting rows.
      if (!padded && !whole)
        rows = std::min<int64_t>(rows, 1 + 1023 / (columns / 4));
      bool blockedOutput = llvm::isPowerOf2_64(columns / 4) && columns <= 64 &&
                           outputStrides.back() == 1 && outputRowStride > 0;
      blockedOutput &= rows > 1;
      bool single = !padded;
      bool scratchUsed = padded;
      Value scratch;
      if (scratchUsed) {
        SmallVector<Value> dynamic;
        if (paddedN < 0)
          dynamic.push_back(columnCount);
        int64_t m = outputType.getDimSize(batched ? 1 : 0);
        if (m < 0)
          dynamic.push_back(mSize);
        scratch = memref::AllocOp::create(
            b, loc,
            MemRefType::get(
                {batch, paddedN < 0 ? ShapedType::kDynamic : paddedN / columns,
                 m, columns},
                b.getF32Type()),
            dynamic);
      }
      Value zeroIndex = index(b, loc, 0);
      auto groupedStore = [&](Value batchIndex, Value first, Value column,
                              Value height, Value bank) {
        auto emit = [&](Value row, Value count) {
          SmallVector<OpFoldResult> offsets, sizes, strides;
          if (batched) {
            offsets.push_back(batchIndex);
            sizes.push_back(b.getIndexAttr(1));
          }
          offsets.append(
              {b.createOrFold<arith::AddIOp>(loc, first, row), column});
          sizes.append({count, b.getIndexAttr(columns)});
          strides.assign(offsets.size(), b.getIndexAttr(1));
          Value target = memref::SubViewOp::create(b, loc, output, offsets,
                                                   sizes, strides);
          BankMvoutStridedOp::create(
              b, loc, b.getI64Type(), target, bank,
              b.createOrFold<arith::MulIOp>(loc, row,
                                            index(b, loc, columns / 4)),
              arith::MulIOp::create(b, loc, count, index(b, loc, columns / 4)),
              index(b, loc, outputRowStride * 4), 0, columns / 4,
              index(b, loc, 16), index(b, loc, llvm::Log2_64(columns / 4)));
        };
        emit(zeroIndex, height);
      };
      if (wSource && !batched && !padded && chunks > 0 && !onlineLhs &&
          !matchPattern(mSize, m_Zero())) {
        // One runtime row tail replaces the binary decomposition of its height.
        // Both its operand footprint and its output footprint fit the same
        // banks as a full tile; only independent output rows change their
        // grouping.
        auto tiles = [&](function_ref<void(Value, Value)> tile) {
          Value full =
              b.createOrFold<arith::DivUIOp>(loc, mSize, index(b, loc, rows));
          Value tail =
              b.createOrFold<arith::RemUIOp>(loc, mSize, index(b, loc, rows));
          loops(b, loc, {full}, true, [&](ValueRange at) {
            tile(arith::MulIOp::create(b, loc, at[0], index(b, loc, rows)),
                 index(b, loc, rows));
          });
          if (!matchPattern(tail, m_Zero())) {
            Value active = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::ugt, tail, zeroIndex);
            auto last = scf::IfOp::create(b, loc, active, false);
            OpBuilder::InsertionGuard insertion(b);
            b.setInsertionPointToStart(&last.getThenRegion().front());
            tile(arith::MulIOp::create(b, loc, full, index(b, loc, rows)),
                 tail);
          }
        };
        auto inputRows = [&](Value part, Value first, Value height) -> Value {
          return memref::SubViewOp::create(
              b, loc, lhs,
              ArrayRef<OpFoldResult>{b.getIndexAttr(0), part, first,
                                     b.getIndexAttr(0)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                     height, b.getIndexAttr(width)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                     b.getIndexAttr(1), b.getIndexAttr(1)});
        };
        auto loadInput = [&](Value part, Value first, Value height) -> Value {
          Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
          if (directInput) {
            for (bool exponent : {false, true}) {
              int64_t count = exponent ? chunk / 32 : chunk;
              int64_t bytes = exponent ? 16 : 256;
              Value start =
                  arith::MulIOp::create(b, loc, part, index(b, loc, count));
              for (int64_t offset = 0; offset < count; offset += bytes) {
                Value column = arith::AddIOp::create(
                    b, loc, start, index(b, loc, offset + (exponent ? k : 0)));
                Value source = memref::SubViewOp::create(
                    b, loc, lhs, ArrayRef<OpFoldResult>{first, column},
                    ArrayRef<OpFoldResult>{height, b.getIndexAttr(bytes)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                           b.getIndexAttr(1)});
                bank = BankMvinStridedOp::create(
                    b, loc, b.getI64Type(), source, bank,
                    index(b, loc, (offset + (exponent ? chunk : 0)) / 16),
                    arith::MulIOp::create(b, loc, height,
                                          index(b, loc, bytes / 16)),
                    index(b, loc, sourceWidth), 0, rowDepth, index(b, loc, 16),
                    index(b, loc, llvm::Log2_64(bytes / 16)));
              }
            }
            return bank;
          }
          return BankMvinStridedOp::create(
              b, loc, b.getI64Type(), inputRows(part, first, height), bank,
              zeroIndex,
              arith::MulIOp::create(b, loc, height, index(b, loc, rowDepth)),
              index(b, loc, 16), 0, 1, index(b, loc, 16), zeroIndex);
        };
        auto store = [&](Value first, Value column, Value height,
                         Value result) {
          if (whole) {
            Value target = memref::SubViewOp::create(
                b, loc, output, ArrayRef<OpFoldResult>{first, column},
                ArrayRef<OpFoldResult>{height, b.getIndexAttr(columns)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
            BankMvoutStridedOp::create(
                b, loc, b.getI64Type(), target, result, zeroIndex,
                arith::MulIOp::create(b, loc, height,
                                      index(b, loc, columns / 4)),
                index(b, loc, 16), 0, 1, index(b, loc, 16), zeroIndex);
          } else if (blockedOutput) {
            groupedStore(zeroIndex, first, column, height, result);
          } else {
            loops(b, loc, {height}, true, [&](ValueRange row) {
              Value target = memref::SubViewOp::create(
                  b, loc, output,
                  ArrayRef<OpFoldResult>{
                      arith::AddIOp::create(b, loc, first, row[0]).getResult(),
                      column},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                         b.getIndexAttr(columns)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
              BankMvoutStridedOp::create(
                  b, loc, b.getI64Type(), target, result,
                  arith::MulIOp::create(b, loc, row[0],
                                        index(b, loc, columns / 4)),
                  index(b, loc, columns / 4), index(b, loc, 16), 0, 1,
                  index(b, loc, 16), zeroIndex);
            });
          }
        };
        if (residentWeights) {
          // Retain one or two A tiles when the complete input fits. This
          // removes repeated A reads without increasing the weight traffic.
          auto residentInput = [&](int64_t count) {
            SmallVector<Value> inputs, heights;
            for (int64_t tile = 0; tile < count; ++tile) {
              Value first = index(b, loc, tile * rows);
              Value height =
                  tile + 1 == count
                      ? b.createOrFold<arith::SubIOp>(loc, mSize, first)
                      : index(b, loc, rows);
              inputs.push_back(loadInput(zeroIndex, first, height));
              heights.push_back(height);
            }
            loops(b, loc, {columnCount}, true, [&](ValueRange at) {
              Value column =
                  arith::MulIOp::create(b, loc, at[0], index(b, loc, columns));
              Value weight = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
              weight =
                  BankMvinOp::create(b, loc, b.getI64Type(),
                                     rowsOf(b, loc, rhs, zeroIndex, zeroIndex,
                                            column, columns, width),
                                     weight, columns * rowDepth, 1);
              Value result = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
              for (int64_t tile = 0; tile < count; ++tile) {
                result = ::buddy::mxint8matmul_ball::BankMatmulMxint8Op::create(
                    b, loc, b.getI64Type(), inputs[tile], weight, result,
                    heights[tile], columns, chunk, false);
                store(index(b, loc, tile * rows), column, heights[tile],
                      result);
              }
              BankReleaseOp::create(b, loc, result);
              BankReleaseOp::create(b, loc, weight);
            });
            for (Value input : inputs)
              BankReleaseOp::create(b, loc, input);
          };
          // Larger inputs retain W across row tiles rather than rereading W.
          auto residentWeight = [&] {
            loops(b, loc, {columnCount}, true, [&](ValueRange at) {
              Value column =
                  arith::MulIOp::create(b, loc, at[0], index(b, loc, columns));
              Value weight = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
              weight =
                  BankMvinOp::create(b, loc, b.getI64Type(),
                                     rowsOf(b, loc, rhs, zeroIndex, zeroIndex,
                                            column, columns, width),
                                     weight, columns * rowDepth, 1);
              tiles([&](Value first, Value height) {
                Value input = loadInput(zeroIndex, first, height);
                Value result =
                    BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                result = ::buddy::mxint8matmul_ball::BankMatmulMxint8Op::create(
                    b, loc, b.getI64Type(), input, weight, result, height,
                    columns, chunk, false);
                store(first, column, height, result);
                BankReleaseOp::create(b, loc, input);
                BankReleaseOp::create(b, loc, result);
              });
              BankReleaseOp::create(b, loc, weight);
            });
          };
          if (whole) {
            residentWeight();
          } else if (auto height = getConstantIntValue(mSize)) {
            if (*height <= rows)
              residentInput(1);
            else if (banks >= 4 && *height <= 2 * rows)
              residentInput(2);
            else
              residentWeight();
          } else {
            Value positive = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::ugt, mSize, zeroIndex);
            Value one = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule,
                                              mSize, index(b, loc, rows));
            auto choose = scf::IfOp::create(
                b, loc, arith::AndIOp::create(b, loc, positive, one), true);
            OpBuilder::InsertionGuard insertion(b);
            b.setInsertionPointToStart(&choose.getThenRegion().front());
            residentInput(1);
            b.setInsertionPointToStart(&choose.getElseRegion().front());
            if (banks >= 4) {
              Value more =
                  arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                        mSize, index(b, loc, rows));
              Value two =
                  arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule,
                                        mSize, index(b, loc, 2 * rows));
              auto second = scf::IfOp::create(
                  b, loc, arith::AndIOp::create(b, loc, more, two), true);
              b.setInsertionPointToStart(&second.getThenRegion().front());
              residentInput(2);
              b.setInsertionPointToStart(&second.getElseRegion().front());
              residentWeight();
            } else {
              residentWeight();
            }
          }
        } else {
          // Multiple K blocks retain A across columns, preserving accumulation
          // order.
          tiles([&](Value first, Value height) {
            int64_t keep = chunks + 2 <= banks ? chunks : banks - 3;
            SmallVector<Value> inputs;
            for (int64_t part = 0; part < keep; ++part)
              inputs.push_back(loadInput(index(b, loc, part), first, height));
            loops(b, loc, {columnCount}, true, [&](ValueRange at) {
              Value column =
                  arith::MulIOp::create(b, loc, at[0], index(b, loc, columns));
              Value result = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
              auto product = [&](Value part, Value input, bool accumulate) {
                bool streamed = !input;
                if (streamed)
                  input = loadInput(part, first, height);
                Value weight =
                    BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                weight = BankMvinOp::create(b, loc, b.getI64Type(),
                                            rowsOf(b, loc, rhs, zeroIndex, part,
                                                   column, columns, width),
                                            weight, columns * rowDepth, 1);
                ::buddy::mxint8matmul_ball::BankMatmulMxint8Op::create(
                    b, loc, b.getI64Type(), input, weight, result, height,
                    columns, chunk, accumulate);
                if (streamed)
                  BankReleaseOp::create(b, loc, input);
                BankReleaseOp::create(b, loc, weight);
              };
              if (keep) {
                for (int64_t part = 0; part < keep; ++part)
                  product(index(b, loc, part), inputs[part], part != 0);
                if (keep < chunks) {
                  loops(b, loc, {index(b, loc, chunks - keep - 1)}, true,
                        [&](ValueRange part) {
                          product(arith::AddIOp::create(b, loc, part[0],
                                                        index(b, loc, keep)),
                                  Value(), true);
                        });
                  product(index(b, loc, chunks - 1), Value(), true);
                }
              } else {
                product(zeroIndex, Value(), false);
                loops(b, loc, {index(b, loc, chunks - 1)}, true,
                      [&](ValueRange part) {
                        product(arith::AddIOp::create(b, loc, part[0],
                                                      index(b, loc, 1)),
                                Value(), true);
                      });
              }
              store(first, column, height, result);
              BankReleaseOp::create(b, loc, result);
            });
            for (Value input : inputs)
              BankReleaseOp::create(b, loc, input);
          });
        }
      } else {
        loops(b, loc, {index(b, loc, batch)}, true, [&](ValueRange outer) {
          Value bi = outer[0];
          Value bl = aSource ? zeroIndex : bi;
          Value br = wSource ? zeroIndex : bi;
          rowTiles(b, loc, mSize, rows, 1, [&](Value first, int64_t height) {
            // B and C need two banks; streamed A blocks need one more.
            int64_t keep =
                chunks > 0 ? (chunks + 2 <= banks ? chunks : banks - 3) : 0;
            SmallVector<Value> inputs;
            if (keep) {
              for (int64_t part = 0; part < keep; ++part) {
                Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                Value residentBank = bank;
                if (onlineLhs && height <= quantRows) {
                  auto choose = scf::IfOp::create(b, loc, quantizeInBank, true);
                  OpBuilder::InsertionGuard insertion(b);
                  b.setInsertionPointToStart(&choose.getThenRegion().front());
                  Value in = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                  in = BankMvinOp::create(b, loc, b.getI64Type(),
                                          rowsOf(b, loc, onlineLhs, bl,
                                                 zeroIndex, first, height,
                                                 chunk),
                                          in, height * chunk / 4, 1);
                  ::buddy::mxint8quantize_ball::BankQuantMxint8Op::create(
                      b, loc, b.getI64Type(), in, bank, height, chunk);
                  BankReleaseOp::create(b, loc, in);
                  b.setInsertionPointToStart(&choose.getElseRegion().front());
                  BankMvinOp::create(
                      b, loc, b.getI64Type(),
                      rowsOf(b, loc, lhs, bl, zeroIndex, first, height, width),
                      bank, height * rowDepth, 1);
                } else {
                  residentBank = BankMvinOp::create(
                      b, loc, b.getI64Type(),
                      rowsOf(b, loc, lhs, bl, index(b, loc, part), first,
                             height, width),
                      bank, height * rowDepth, 1);
                }
                inputs.push_back(residentBank);
              }
            }
            loops(b, loc, {columnCount}, true, [&](ValueRange at) {
              Value column =
                  arith::MulIOp::create(b, loc, at[0], index(b, loc, columns));
              Value c = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
              auto product = [&](Value part, Value inputBank, bool accumulate) {
                Value lb = inputBank;
                if (!inputBank) {
                  lb = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                  lb = BankMvinOp::create(
                      b, loc, b.getI64Type(),
                      rowsOf(b, loc, lhs, bl, part, first, height, width), lb,
                      height * rowDepth, 1);
                }
                Value wb = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                wb = BankMvinOp::create(
                    b, loc, b.getI64Type(),
                    rowsOf(b, loc, rhs, br, part, column, columns, width), wb,
                    columns * rowDepth, 1);
                ::buddy::mxint8matmul_ball::BankMatmulMxint8Op::create(
                    b, loc, b.getI64Type(), lb, wb, c, index(b, loc, height),
                    columns, chunk, accumulate);
                if (!inputBank)
                  BankReleaseOp::create(b, loc, lb);
                BankReleaseOp::create(b, loc, wb);
              };
              if (keep) {
                for (int64_t part = 0; part < keep; ++part)
                  product(index(b, loc, part), inputs[part], part != 0);
                if (keep < chunks) {
                  loops(b, loc, {index(b, loc, chunks - keep - 1)}, true,
                        [&](ValueRange part) {
                          product(arith::AddIOp::create(b, loc, part[0],
                                                        index(b, loc, keep)),
                                  Value(), true);
                        });
                  product(index(b, loc, chunks - 1), Value(), true);
                }
              } else {
                product(zeroIndex, Value(), false);
                loops(b, loc,
                      {arith::SubIOp::create(b, loc, chunkCount,
                                             index(b, loc, 1))},
                      true, [&](ValueRange part) {
                        product(arith::AddIOp::create(b, loc, part[0],
                                                      index(b, loc, 1)),
                                Value(), true);
                      });
              }
              if (!padded && !whole && height > 1 && blockedOutput) {
                groupedStore(bi, first, column, index(b, loc, height), c);
              } else if (!padded && !whole && height > 1) {
                // Write each computed row directly to its final DDR columns.
                // A row spans contiguous bank lines even when DDR rows are far
                // apart.
                auto writeRow = [&](Value row) {
                  SmallVector<OpFoldResult> offsets;
                  if (batched)
                    offsets.push_back(bi);
                  offsets.append(
                      {arith::AddIOp::create(b, loc, first, row).getResult(),
                       column});
                  SmallVector<OpFoldResult> ones(offsets.size(),
                                                 b.getIndexAttr(1));
                  Value memory = memref::SubViewOp::create(b, loc, output,
                                                           offsets, ones, ones);
                  BankMvoutStridedOp::create(
                      b, loc, b.getI64Type(), memory, c,
                      arith::MulIOp::create(b, loc, row,
                                            index(b, loc, columns / 4)),
                      index(b, loc, columns / 4), index(b, loc, 16), 0, 1,
                      index(b, loc, 16), zeroIndex);
                };
                // A stream loop adds setup and offset commands. Small tiles
                // need fewer commands when their rows are emitted directly.
                if (height >= 8)
                  loops(b, loc, {index(b, loc, height)}, true,
                        [&](ValueRange row) { writeRow(row[0]); });
                else
                  for (int64_t row = 0; row < height; ++row)
                    writeRow(index(b, loc, row));
              } else {
                Value out;
                if (whole || (height == 1 && single)) {
                  SmallVector<OpFoldResult> offsets, sizes, strides;
                  if (batched) {
                    offsets.push_back(bi);
                    sizes.push_back(b.getIndexAttr(1));
                  }
                  offsets.append({first, column});
                  sizes.append(
                      {b.getIndexAttr(height), b.getIndexAttr(columns)});
                  strides.assign(offsets.size(), b.getIndexAttr(1));
                  out = memref::SubViewOp::create(b, loc, output, offsets,
                                                  sizes, strides);
                } else {
                  out = rowsOf(b, loc, scratch, bi, at[0], first, height,
                               columns);
                }
                BankMvoutOp::create(b, loc, b.getI64Type(), out, c,
                                    height * columns / 4, 1);
              }
              BankReleaseOp::create(b, loc, c);
            });
            for (Value bank : inputs)
              BankReleaseOp::create(b, loc, bank);
          });
        });
      }
      if (scratchUsed) {
        // Trim the padded columns from scratch.
        Value copied = mSize;
        OpBuilder::InsertionGuard insertion(b);
        if (!matchPattern(copied, m_Constant())) {
          // No copy when every row landed in place (a single token).
          Value any = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                            copied, zeroIndex);
          auto guard = scf::IfOp::create(b, loc, any, false);
          b.setInsertionPointToStart(&guard.getThenRegion().front());
        }
        loops(
            b, loc, {index(b, loc, batch), columnCount}, false,
            [&](ValueRange at) {
              for (Value iv : at)
                cast<BlockArgument>(iv).getOwner()->getParentOp()->setAttr(
                    "hart.record", b.getUnitAttr());
              Value column =
                  arith::MulIOp::create(b, loc, at[1], index(b, loc, columns));
              OpFoldResult span = b.getIndexAttr(columns);
              if (padded)
                span = arith::MinUIOp::create(
                           b, loc, index(b, loc, columns),
                           arith::SubIOp::create(b, loc, nSize, column))
                           .getResult();
              // Both sides as 2-D [rows][columns] views.
              auto matrix = [&](Value memory, ArrayRef<OpFoldResult> at,
                                ArrayRef<OpFoldResult> extents) -> Value {
                SmallVector<OpFoldResult> sizes;
                for (OpFoldResult extent : extents)
                  if (auto fixed = getConstantIntValue(extent))
                    sizes.push_back(b.getIndexAttr(*fixed));
                  else
                    sizes.push_back(extent);
                SmallVector<OpFoldResult> strides(at.size(), b.getIndexAttr(1));
                SmallVector<int64_t> shape;
                for (OpFoldResult extent :
                     ArrayRef<OpFoldResult>(sizes).take_back(2))
                  shape.push_back(getConstantIntValue(extent).value_or(
                      ShapedType::kDynamic));
                auto type = memref::SubViewOp::inferRankReducedResultType(
                    shape, cast<MemRefType>(memory.getType()), at, sizes,
                    strides);
                return memref::SubViewOp::create(b, loc, cast<MemRefType>(type),
                                                 memory, at, sizes, strides);
              };
              Value from = matrix(
                  scratch, {at[0], at[1], b.getIndexAttr(0), b.getIndexAttr(0)},
                  {b.getIndexAttr(1), b.getIndexAttr(1), copied, span});
              SmallVector<OpFoldResult> offsets, sizes;
              if (batched) {
                offsets.push_back(at[0]);
                sizes.push_back(b.getIndexAttr(1));
              }
              offsets.append({b.getIndexAttr(0), column});
              sizes.append({copied, span});
              Value to = matrix(output, offsets, sizes);
              Value count = getValueOrCreateConstantIndexOp(b, loc, span);
              Value lines =
                  arith::DivUIOp::create(b, loc, count, index(b, loc, 4));
              Value tail =
                  arith::RemUIOp::create(b, loc, count, index(b, loc, 4));
              // Copy only logical columns; byte DMA handles the final word
              // without a CPU copy or a read past the tensor's last element.
              for (bool partial : {false, true}) {
                OpBuilder::InsertionGuard guard(b);
                Value length = partial ? tail : lines;
                Value nonempty = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::ugt, length, zeroIndex);
                auto active = scf::IfOp::create(b, loc, nonempty, false);
                b.setInsertionPointToStart(&active.getThenRegion().front());
                Value offset = partial ? arith::MulIOp::create(b, loc, lines,
                                                               index(b, loc, 4))
                                             .getResult()
                                       : zeroIndex;
                Value floats = partial ? tail
                                       : arith::MulIOp::create(b, loc, lines,
                                                               index(b, loc, 4))
                                             .getResult();
                Value bytes = partial ? arith::MulIOp::create(b, loc, tail,
                                                              index(b, loc, 4))
                                            .getResult()
                                      : index(b, loc, 16);
                Value depth = partial ? index(b, loc, 1) : lines;
                loops(b, loc, {copied}, true, [&](ValueRange row) {
                  Value bank =
                      BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
                  for (bool write : {false, true}) {
                    Value memory = memref::SubViewOp::create(
                        b, loc, write ? to : from,
                        ArrayRef<OpFoldResult>{row[0], offset},
                        ArrayRef<OpFoldResult>{b.getIndexAttr(1), floats},
                        ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                               b.getIndexAttr(1)});
                    if (write)
                      bank = BankMvoutStridedOp::create(
                          b, loc, b.getI64Type(), memory, bank, zeroIndex,
                          depth, index(b, loc, 16), 0, 1, bytes, zeroIndex);
                    else
                      bank = BankMvinStridedOp::create(
                          b, loc, b.getI64Type(), memory, bank, zeroIndex,
                          depth, index(b, loc, 16), 0, 1, bytes, zeroIndex);
                  }
                  BankReleaseOp::create(b, loc, bank);
                });
              }
            });
      }
      Value written = bufferization::ToTensorOp::create(b, loc, outputType,
                                                        output, true, true);
      destination.getResult().replaceAllUsesWith(written);
      destination.erase();
      op->erase();
    }
    // Delete the Q/DQ and layout chains consumed by the Ball lowering.
    RewritePatternSet patterns(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    if (failed(lowerPolyQuantizedOutputs(getOperation())) ||
        failed(lowerPolyActivations(getOperation())) ||
        failed(lowerPolyIntegerSoftmax(getOperation())) ||
        failed(lowerPolyRowOperators(getOperation())))
      signalPassFailure();
  }
};
PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
