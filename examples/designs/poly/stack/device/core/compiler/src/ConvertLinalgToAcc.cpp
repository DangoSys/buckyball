//===- ConvertLinalgToAcc.cpp - Poly MXINT8 computations on banks -------===//
//
// Every product of a thread subgraph becomes tiles of Ball instructions:
// Mt x Nt output tiles, each the Ball's own M rows times N columns, within
// the limits of the Ball and the bank depth of the target (design.pb).
//
//   - MXINT8 products (both operands decoded MXINT8) run on
//     Mxint8matmulBall over packed rows of k = K (one bank row group per
//     row: K codes, K/32 exponents, zero padding to 16 bytes); a dynamic K
//     is split into blocks of 32 chained with `accumulate`.
//   - FP32 products run on MxmmBall (f32, rows one or a multiple of 16).
//
// Operands: a packed operand (a weight, or an activation the reduce core
// quantized) is read in place; it must be in the Ball layout, rows of
// 16 * S(K) bytes (the recipe packs weights so offline). An operand
// quantized online is quantized by Mxint8quantizeBall straight into the
// [batch][block][row][S] layout the product reads, several rows per
// instruction.
//
// Loops: batch, row tiles, column tiles, K blocks. A row tile of A is moved
// into its bank once and stays there for every column tile when K is one
// block; each weight tile is moved once per row tile and serves all its
// rows through the Ball's M. Row counts that are no multiple of Mt end with
// tiles of halving height (MxmmBall: down to 16 rows, then single rows): the
// command fields of a recorded stream loop are constant, so each tile height
// is its own loop.
//
// Results: a C tile is Mt x Nt row-major in its bank and mvout moves rows
// from bank row 0 with one stride, so it lands in place only when its rows
// are whole output rows (Nt = N) or a single row. Other tiles go to a
// tile-major scratch [batch][column tile][row][Nt], copied into the output
// by the core after completion, one 2-D copy per column tile.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Core/CoreOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mxint8matmulOps.h"
#include "mxint8quantizeOps.h"
#include "mxmmOps.h"
#include <cmath>

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

LogicalResult lowerPolyQuantizedOutputs(ModuleOp module);

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
             tensor::ExpandShapeOp>(op))
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
      loop->setAttr("core.stream", b.getUnitAttr());
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
  return bufferization::ToBufferOp::create(
      b, loc, MemRefType::get(type.getShape(), type.getElementType()), tensor);
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
    r.insert<AccDialect, arith::ArithDialect,
             bufferization::BufferizationDialect, memref::MemRefDialect,
             scf::SCFDialect, tensor::TensorDialect,
             ::buddy::mxint8matmul_ball::Mxint8matmulBallDialect,
             ::buddy::mxint8quantize_ball::Mxint8quantizeBallDialect,
             ::buddy::mxmm_ball::MxmmBallDialect>();
  }
  void runOnOperation() override {
    const auto &target = buckyball_target::getBuckyballTarget();
    const int64_t depth = target.bankDepth;
    if (target.bankWidthBits != 128 || target.bankNum < 3) {
      getOperation().emitError("poly needs 128-bit banks and three banks");
      return signalPassFailure();
    }
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
      if (profile && profile.getValue() == target.core &&
          isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
        products.push_back(op);
    });
    for (Operation *op : products) {
      OpBuilder b(op);
      Location loc = op->getLoc();
      Operand a = operand(op->getOperand(0)), w = operand(op->getOperand(1));
      bool mx = a.decode && w.decode;
      auto outputType = cast<RankedTensorType>(op->getResult(0).getType());
      auto aType = cast<RankedTensorType>(a.value.getType());
      bool batched = isa<linalg::BatchMatmulOp>(op);
      int64_t batch = batched ? outputType.getDimSize(0) : 1;
      int64_t n = outputType.getDimSize(batched ? 2 : 1);
      int64_t k = aType.getShape().back();
      auto fill = op->getOperand(2).getDefiningOp<linalg::FillOp>();
      bool zero = matchPattern(fill ? fill.getInputs()[0] : op->getOperand(2),
                               m_PosZeroFloat());
      if (batch < 0 || k == 0 || (k >= 0 && k % (mx ? 32 : 4)) || !zero ||
          (!mx && k < 0)) {
        op->emitError("poly product needs a static batch, aligned positive K "
                      "(static for FP32) and zero initialization");
        return signalPassFailure();
      }
      auto users = op->getResult(0).getUsers();
      auto destination =
          op->getResult(0).hasOneUse()
              ? dyn_cast<bufferization::MaterializeInDestinationOp>(
                    *users.begin())
              : bufferization::MaterializeInDestinationOp();
      if (!destination) {
        op->emitError("poly product must write its core subgraph destination");
        return signalPassFailure();
      }
      Value output = buffer(b, loc, destination.getDest());
      auto size = [&](int64_t axis) -> Value {
        if (!outputType.isDynamicDim(axis))
          return index(b, loc, outputType.getDimSize(axis));
        return tensor::DimOp::create(b, loc, destination.getDest(), axis);
      };
      Value mSize = size(batched ? 1 : 0);
      Value nSize = size(outputType.getRank() - 1);
      Value kSize;
      if (mx) {
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
      } else
        kSize = tensor::DimOp::create(b, loc, a.value, aType.getRank() - 1);
      auto productOp = cast<linalg::LinalgOp>(op);
      bool transposedRhs =
          cast<AffineDimExpr>(
              productOp.getIndexingMapsArray()[1].getResult(batched ? 1 : 0))
              .getPosition() == (batched ? 2 : 1);

      // The K block: all of a static K (one block), 32 for a dynamic K.
      int64_t chunk = k < 0 ? 32 : k;
      int64_t chunks = k < 0 ? ShapedType::kDynamic : 1;
      Value chunkCount = b.createOrFold<arith::DivUIOp>(
          loc,
          b.createOrFold<arith::AddIOp>(loc, kSize, index(b, loc, chunk - 1)),
          index(b, loc, chunk));
      // Bank rows per operand row and bytes (MX) or values (FP32) of it.
      int64_t rowDepth = mx ? packedRows(chunk) : chunk / 4;
      int64_t width = mx ? rowDepth * 16 : chunk;
      // Operand rows a bank holds; MxmmBall takes 1 or 16k rows, 16k columns.
      int64_t fit = std::min<int64_t>(depth / rowDepth, 1023);
      // Rows one quantization takes: its FP32 input and packed output.
      int64_t quantRows = std::min(depth / (chunk / 4), depth / rowDepth);
      if (fit < 1 || (mx && quantRows < 1)) {
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
          if (!slice.hasZeroOffset() || !slice.hasUnitStride())
            return Value();
          source = slice.getSource();
        }
        auto type = cast<RankedTensorType>(source.getType());
        if (k < 0 || type.getRank() != 2 || !rows ||
            int64_t(input.decode.getAxis()) != 1 || type.getDimSize(1) != width)
          return Value();
        return source;
      };
      Value aSource, wSource;
      if (mx && !a.quant && !(aSource = inPlace(a, false))) {
        op->emitError("poly packed activation is not in the Ball layout");
        return signalPassFailure();
      }
      if (mx && !w.quant && !(wSource = inPlace(w, true))) {
        op->emitError("poly weight is not in the Ball layout: rows of "
                      "16 * S(K) bytes, K codes then K / 32 exponents");
        return signalPassFailure();
      }

      // Column tiles: a divisor of the rows of a packed weight (no reads
      // past it), else as wide as the bank allows, balanced over N.
      int64_t columns;
      if (wSource) {
        columns =
            divisor(cast<RankedTensorType>(wSource.getType()).getDimSize(0),
                    std::min(fit, square));
      } else if (!mx) {
        columns = std::min<int64_t>(fit - fit % 16, 64);
        if (n >= 0)
          columns = std::min<int64_t>(columns, (n + 15) / 16 * 16);
      } else {
        columns = std::min(fit, square);
        columns -= columns % 4;
        if (n >= 0) {
          int64_t padded = (n + 3) / 4 * 4;
          int64_t count = (padded + columns - 1) / columns;
          columns = ((padded + count - 1) / count + 3) / 4 * 4;
        }
      }
      if (columns < (mx ? 4 : 16)) {
        op->emitError("poly product cannot fit a column tile in a bank");
        return signalPassFailure();
      }
      // Row tiles: what the A bank and the accumulator bank (Mt * Nt FP32)
      // hold.
      int64_t rows = std::min(fit, 4 * depth / columns);
      if (!mx)
        rows = rows >= 16 ? rows - rows % 16 : 1;
      if (columns < (mx ? 4 : 16) || rows < 1) {
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

      // An operand quantized online or in FP32: staged as
      // [batch][block][row][chunk] (in place when contiguous), then
      // quantized by the Ball into [batch][block][row][width].
      auto stage = [&](Operand input, bool rhs) -> Value {
        bool online = mx && input.quant;
        Value source = !mx ? input.value : input.quant.getInput();
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
        bool direct =
            k >= 0 && !(rhs && padded) &&
            (online ? int64_t(input.quant.getAxis()) == sourceType.getRank() - 1
                    : (!rhs || transposedRhs)) &&
            sourceType.getShape().back() == k &&
            sourceType.getRank() == (batched ? 3 : 2);
        if (direct) {
          // DMA reads the rows in place: no host copy.
          Value sourceBuffer = buffer(b, loc, source);
          sourceBuffer.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
          auto layout = StridedLayoutAttr::get(
              b.getContext(), 0,
              {rowCount < 0 ? ShapedType::kDynamic : rowCount * k, chunk, k,
               1});
          storage = memref::ReinterpretCastOp::create(
              b, loc,
              MemRefType::get({batch, 1, rowCount, chunk}, element, layout),
              sourceBuffer, b.getIndexAttr(0),
              ArrayRef<OpFoldResult>{
                  b.getIndexAttr(batch), b.getIndexAttr(1),
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
        } else {
          // The host gathers a transposed or padded operand once before
          // recording; padding rows and K are zeros.
          storage = memref::AllocOp::create(
              b, loc,
              MemRefType::get({batch, chunks, storedRows, chunk}, element),
              dynamic);
          linalg::FillOp::create(b, loc,
                                 ValueRange{arith::ConstantOp::create(
                                     b, loc, b.getZeroAttr(element))},
                                 storage);
          Value sourceK = tensor::DimOp::create(
              b, loc, source,
              online ? int64_t(input.quant.getAxis())
                     : (rhs && !transposedRhs ? sourceType.getRank() - 2
                                              : sourceType.getRank() - 1));
          loops(
              b, loc,
              {index(b, loc, batch), chunkCount, extent, index(b, loc, chunk)},
              false, [&](ValueRange at) {
                Value coordinate = arith::AddIOp::create(
                    b, loc,
                    arith::MulIOp::create(b, loc, at[1], index(b, loc, chunk)),
                    at[3]);
                SmallVector<Value> coords;
                if (batched)
                  coords.push_back(at[0]);
                coords.push_back(rhs && !transposedRhs ? coordinate : at[2]);
                coords.push_back(rhs && !transposedRhs ? at[2] : coordinate);
                for (Operation *view :
                     (mx ? input.views : SmallVector<Operation *>{})) {
                  if (auto transpose = dyn_cast<linalg::TransposeOp>(view)) {
                    SmallVector<Value> reordered(coords.size());
                    for (auto [i, axis] :
                         llvm::enumerate(transpose.getPermutation()))
                      reordered[axis] = coords[i];
                    coords = reordered;
                  } else {
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
                    for (int64_t axis = original.getRank() - 1; axis >= 0;
                         --axis) {
                      coords[axis] = arith::RemUIOp::create(
                          b, loc, flat,
                          tensor::DimOp::create(b, loc, view->getOperand(0),
                                                axis)
                              .getResult());
                      flat = arith::DivUIOp::create(
                          b, loc, flat,
                          tensor::DimOp::create(b, loc, view->getOperand(0),
                                                axis)
                              .getResult());
                    }
                  }
                }
                Value value;
                if (k < 0) {
                  Value valid = arith::CmpIOp::create(
                      b, loc, arith::CmpIPredicate::ult, coordinate, sourceK);
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
                  value = choose.getResult(0);
                } else
                  value = tensor::ExtractOp::create(b, loc, source, coords);
                memref::StoreOp::create(b, loc, value, storage, at);
              });
        }
        if (!online)
          return storage;
        // Quantized rows go straight where the product reads them: as many
        // rows per instruction as the input and output banks hold.
        Value quantized = memref::AllocOp::create(
            b, loc,
            MemRefType::get({batch, chunks, storedRows, width}, b.getI8Type()),
            dynamic);
        loops(
            b, loc, {index(b, loc, batch), chunkCount}, true,
            [&](ValueRange at) {
              rowTiles(
                  b, loc, storedExtent, quantRows, 1,
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
        return quantized;
      };
      // A packed operand in place as [1][1][row][width].
      auto view = [&](Value source) -> Value {
        Value memory = buffer(b, loc, source);
        memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
        auto type = cast<MemRefType>(memory.getType());
        int64_t height = type.getDimSize(0);
        Value extent = memref::DimOp::create(b, loc, memory, 0);
        auto layout = StridedLayoutAttr::get(
            b.getContext(), 0,
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
            memory, b.getIndexAttr(0),
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                   rowsSize, b.getIndexAttr(width)},
            ArrayRef<OpFoldResult>{plane, plane, b.getIndexAttr(width),
                                   b.getIndexAttr(1)});
      };
      Value lhs = aSource ? view(aSource) : stage(a, false);
      Value rhs = wSource ? view(wSource) : stage(w, true);

      // Where the tiles land: in place when a tile is whole output rows or a
      // single row of unpadded columns, else in the tile-major scratch.
      bool whole = !padded && columns == n;
      bool single = !padded;
      // MxmmBall takes one row or a multiple of 16.
      int64_t least = mx ? 1 : 16;
      bool scratchUsed = !whole && !(rows == 1 && single);
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
      loops(b, loc, {index(b, loc, batch)}, true, [&](ValueRange outer) {
        Value bi = outer[0];
        Value bl = aSource ? zeroIndex : bi;
        Value br = wSource ? zeroIndex : bi;
        rowTiles(b, loc, mSize, rows, least, [&](Value first, int64_t height) {
          bool resident = chunks == 1;
          Value ab;
          if (resident) {
            ab = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            ab = BankMvinOp::create(
                b, loc, b.getI64Type(),
                rowsOf(b, loc, lhs, bl, zeroIndex, first, height, width), ab,
                height * rowDepth, 1);
          }
          loops(b, loc, {columnCount}, true, [&](ValueRange at) {
            Value column =
                arith::MulIOp::create(b, loc, at[0], index(b, loc, columns));
            Value c = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            auto product = [&](Value part, bool accumulate, bool last) {
              Value lb = ab;
              if (!resident) {
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
              if (mx)
                ::buddy::mxint8matmul_ball::BankMatmulMxint8Op::create(
                    b, loc, b.getI64Type(), lb, wb, c, height, columns, chunk,
                    accumulate);
              else
                ::buddy::mxmm_ball::BankMatmulF32Op::create(
                    b, loc, b.getI64Type(), lb, wb, c, height, columns, chunk,
                    0, !accumulate, last);
              if (!resident)
                BankReleaseOp::create(b, loc, lb);
              BankReleaseOp::create(b, loc, wb);
            };
            if (resident) {
              product(zeroIndex, false, true);
            } else {
              product(zeroIndex, false, false);
              loops(
                  b, loc,
                  {arith::SubIOp::create(b, loc, chunkCount, index(b, loc, 1))},
                  true, [&](ValueRange part) {
                    product(arith::AddIOp::create(b, loc, part[0],
                                                  index(b, loc, 1)),
                            true, true);
                  });
            }
            Value out;
            if (whole || (height == 1 && single)) {
              SmallVector<OpFoldResult> offsets, sizes, strides;
              if (batched) {
                offsets.push_back(bi);
                sizes.push_back(b.getIndexAttr(1));
              }
              offsets.append({first, column});
              sizes.append({b.getIndexAttr(height), b.getIndexAttr(columns)});
              strides.assign(offsets.size(), b.getIndexAttr(1));
              out = memref::SubViewOp::create(b, loc, output, offsets, sizes,
                                              strides);
            } else {
              out = rowsOf(b, loc, scratch, bi, at[0], first, height, columns);
            }
            BankMvoutOp::create(b, loc, b.getI64Type(), out, c,
                                height * columns / 4, 1);
            BankReleaseOp::create(b, loc, c);
          });
          if (resident)
            BankReleaseOp::create(b, loc, ab);
        });
      });
      if (scratchUsed) {
        // The rows tiles left in the scratch: all but the single rows that
        // landed in place.
        Value copied = mSize;
        if (single && rows > 1) {
          Value rest =
              b.createOrFold<arith::RemUIOp>(loc, mSize, index(b, loc, rows));
          Value direct = b.createOrFold<arith::RemUIOp>(
              loc, rest, index(b, loc, mx ? 2 : 16));
          copied = b.createOrFold<arith::SubIOp>(loc, mSize, direct);
        }
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
              auto copy = memref::CopyOp::create(b, loc, from, to);
              copy->setAttr("core.after", b.getUnitAttr());
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
    if (failed(lowerPolyQuantizedOutputs(getOperation())))
      signalPassFailure();
  }
};
PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
