//===- ConvertLinalgToAcc.cpp - Goban INT8 products on matmul threads -----===//
//
// convert-linalg-to-acc maps every INT8 product of a thread subgraph to the
// Balls of the matmul thread: smatmul (A [rows, K] . B [K, 16] in INT32),
// toint8 (the online quantization of its FP32 operands) and int2fp (the
// INT32 accumulators to FP32 with the product of the two steps).
//
// smatmul reads all of K in one instruction: A and B are banks of G groups,
// group j the reduction block j of g columns, K = G * g zero-padded to a
// multiple of sixteen. B holds g lines of sixteen INT8 output channels per
// group; A, for rows = 1, g / 16 lines of sixteen INT8 values per group,
// otherwise M tiles of tileRows rows, line ((r / T) * (g / 16) + c / 16) * T
// + r % T. A, B and the accumulator take 2G + 1 banks.
//
//   - A weight is stored offline in the B layout, [S, g, G * 16] for the
//     S = ceil(N / 16) output slices (the recipe's frontend): the graph reads
//     it through expand_shape, transpose (0, 3, 2, 1), collapse_shape and
//     zero-offset slices to [N, K]. Every slice is a DMA of the parameter in
//     place.
//   - An online operand is quantized by toint8, one row (A) or one K line
//     (B) at a time, and the INT8 result written by mvout straight into the
//     bank layout of smatmul (its line stride). Its FP32 input is read in
//     place when it is contiguous [M, K] (A) or [K, N] (B) with K and N
//     complete; otherwise it is first staged, zero-padded, on the CPU.
//
// The product: per M tile of R rows (tileRows multiples, as many as A and the
// accumulator hold, M split evenly), A is loaded once and every output slice
// of sixteen columns streams its B through it: each weight slice moves once
// per M tile and serves all R rows. The accumulator tile, R rows of sixteen
// INT32 (four lines each), leaves by one DMA into a tile-major INT32 buffer;
// int2fp then reads it back a row at a time and writes the FP32 row of
// sixteen in place in the output. int2fp scales the sixteen columns only,
// and the input step of a token-mixing product differs per row, so the
// conversion is per row; mvout has one stride from bank row 0, so a
// multi-row accumulator tile cannot land in a row-major output in place
// either. Rows past M are zero rows of A, computed but never converted.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "int2fpOps.h"
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
#include "smatmulOps.h"
#include "toint8Ops.h"

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;
LogicalResult lowerGobanBiasOutputs(ModuleOp module);

namespace {
Value index(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIndexOp::create(b, loc, n);
}
void loops(OpBuilder &b, Location loc, ArrayRef<int64_t> sizes, bool stream,
           function_ref<void(ValueRange)> body) {
  OpBuilder::InsertionGuard guard(b);
  SmallVector<Value> ivs;
  for (int64_t size : sizes) {
    auto loop = scf::ForOp::create(b, loc, index(b, loc, 0),
                                   index(b, loc, size), index(b, loc, 1));
    if (stream)
      loop->setAttr("core.stream", b.getUnitAttr());
    ivs.push_back(loop.getInductionVar());
    b.setInsertionPointToStart(loop.getBody());
  }
  body(ivs);
}
Value buffer(OpBuilder &b, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  return bufferization::ToBufferOp::create(
      b, loc, MemRefType::get(type.getShape(), type.getElementType()), tensor);
}
// The view of `memory` at `offsets` (leading unit dimensions), `width`
// elements of its last dimension: a DMA reads or writes from its first
// element.
Value view(OpBuilder &b, Location loc, Value memory,
           ArrayRef<OpFoldResult> offsets, int64_t width) {
  SmallVector<OpFoldResult> sizes(offsets.size(), b.getIndexAttr(1)),
      strides(offsets.size(), b.getIndexAttr(1));
  sizes.back() = b.getIndexAttr(width);
  return memref::SubViewOp::create(b, loc, memory, offsets, sizes, strides);
}
Value broadcast(OpBuilder &b, Location loc, Value tensor, ValueRange at) {
  auto type = cast<RankedTensorType>(tensor.getType());
  SmallVector<Value> coords;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    coords.push_back(type.getDimSize(axis) == 1
                         ? index(b, loc, 0)
                         : at[at.size() - type.getRank() + axis]);
  return tensor::ExtractOp::create(b, loc, tensor, coords);
}
struct Operand {
  Int8ToFP32Op decode;
  FP32ToInt8Op quant;
  SmallVector<linalg::TransposeOp> views;
};
Operand operand(Value value) {
  Operand result;
  while (auto transpose = value.getDefiningOp<linalg::TransposeOp>()) {
    result.views.push_back(transpose);
    value = transpose.getInput();
  }
  result.decode = value.getDefiningOp<Int8ToFP32Op>();
  if (result.decode)
    result.quant = result.decode.getInput().getDefiningOp<FP32ToInt8Op>();
  return result;
}
SmallVector<Value> coordinates(Operand input, ValueRange at) {
  SmallVector<Value> result(at);
  for (auto view : input.views) {
    SmallVector<Value> reordered(result.size());
    for (auto [i, axis] : llvm::enumerate(view.getPermutation()))
      reordered[axis] = result[i];
    result = reordered;
  }
  return result;
}
// The weight parameter [S, g, G * 16] in the smatmul B layout the decoded
// [N, K] weight `value` reads, or null.
Value weightLayout(Value value) {
  while (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (!llvm::all_of(slice.getStaticOffsets(),
                      [](int64_t offset) { return offset == 0; }) ||
        !llvm::all_of(slice.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }))
      return {};
    value = slice.getSource();
  }
  auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>();
  auto transpose = collapse
                       ? collapse.getSrc().getDefiningOp<linalg::TransposeOp>()
                       : linalg::TransposeOp();
  auto expand =
      transpose ? transpose.getInput().getDefiningOp<tensor::ExpandShapeOp>()
                : tensor::ExpandShapeOp();
  if (!expand || transpose.getPermutation() != ArrayRef<int64_t>{0, 3, 2, 1} ||
      !isa<BlockArgument>(expand.getSrc()))
    return {};
  return expand.getSrc();
}

class ConvertLinalgToAccPass
    : public PassWrapper<ConvertLinalgToAccPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToAccPass)
  StringRef getArgument() const final { return "convert-linalg-to-acc"; }
  StringRef getDescription() const final {
    return "Map goban INT8 products, quantization and bias decoding to banks";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<AccDialect, arith::ArithDialect,
             bufferization::BufferizationDialect, linalg::LinalgDialect,
             memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
             ::buddy::smatmul_ball::SMatMulBallDialect,
             ::buddy::toint8_ball::ToInt8BallDialect,
             ::buddy::int2fp_ball::Int2FpBallDialect>();
  }
  void runOnOperation() override {
    const auto &target = buckyball_target::getBuckyballTarget();
    int64_t depth = target.bankDepth;
    int64_t tileRows =
        buckyball_target::getBuckyballBallParam("SMATMUL_OS", "tileRows");
    // A and B take G banks each, the accumulator one.
    int64_t maxGroups = (target.bankNum - 1) / 2;
    if (target.bankWidthBits != 128 || maxGroups < 1 || depth < 64) {
      getOperation().emitError("goban needs three 128-bit banks of 64 lines");
      return signalPassFailure();
    }
    SmallVector<Operation *> products;
    getOperation().walk([&](Operation *op) {
      if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op) &&
          operand(op->getOperand(0)).decode &&
          operand(op->getOperand(1)).decode)
        products.push_back(op);
    });
    for (Operation *op : products) {
      OpBuilder b(op);
      Location loc = op->getLoc();
      auto c = [&](int64_t n) { return index(b, loc, n); };
      auto attr = [&](int64_t n) -> OpFoldResult { return b.getIndexAttr(n); };
      auto add = [&](Value x, Value y) -> Value {
        return arith::AddIOp::create(b, loc, x, y);
      };
      auto mul = [&](Value x, int64_t n) -> Value {
        return arith::MulIOp::create(b, loc, x, c(n));
      };
      Type i64 = b.getI64Type();
      Operand a = operand(op->getOperand(0)), w = operand(op->getOperand(1));
      auto type = cast<RankedTensorType>(op->getResult(0).getType());
      bool batched = isa<linalg::BatchMatmulOp>(op);
      int64_t batch = batched ? type.getDimSize(0) : 1;
      int64_t m = type.getDimSize(batched ? 1 : 0), n = type.getShape().back();
      auto maps = cast<linalg::LinalgOp>(op).getIndexingMapsArray();
      bool transposed = cast<AffineDimExpr>(maps[1].getResult(batched ? 1 : 0))
                            .getPosition() == (batched ? 2 : 1);
      int64_t k =
          cast<RankedTensorType>(op->getOperand(0).getType()).getShape().back();
      auto fill = op->getOperand(2).getDefiningOp<linalg::FillOp>();
      if (!type.hasStaticShape() || k <= 0 || !a.quant ||
          !matchPattern(fill ? fill.getInputs()[0] : op->getOperand(2),
                        m_PosZeroFloat())) {
        op->emitError("goban needs static product shapes, an online INT8 "
                      "input and zero initialization");
        return signalPassFailure();
      }
      auto destination =
          op->getResult(0).hasOneUse()
              ? dyn_cast<bufferization::MaterializeInDestinationOp>(
                    *op->getResult(0).getUsers().begin())
              : bufferization::MaterializeInDestinationOp();
      if (!destination) {
        op->emitError("product must write its core subgraph destination");
        return signalPassFailure();
      }
      // Scaling along K cannot be factored out of an integer dot product.
      for (auto [input, rhs] :
           {std::pair<Operand, bool>{a, false}, {w, true}}) {
        int64_t reduction =
            rhs && !transposed ? (batched ? 1 : 0) : (batched ? 2 : 1);
        for (auto view : input.views)
          reduction = view.getPermutation()[reduction];
        auto scaleType =
            cast<RankedTensorType>(input.decode.getStep().getType());
        int64_t aligned =
            reduction -
            cast<RankedTensorType>(input.decode.getResult().getType())
                .getRank() +
            scaleType.getRank();
        if (aligned >= 0 && scaleType.getDimSize(aligned) != 1) {
          op->emitError("goban dot-product scale varies along K");
          return signalPassFailure();
        }
      }

      // The reduction split: from the weight layout, or the most groups
      // that divide K (padded to sixteen) for the largest M tile.
      int64_t slices = (n + 15) / 16, paddedK = (k + 15) / 16 * 16;
      int64_t groups = 0, lines = 0;
      Value weight;
      if (!w.quant) {
        // The [N, K] weight reaches the product as B transposed: by the
        // product's indexing maps or by one transpose.
        bool readTransposed =
            w.views.empty()
                ? transposed
                : !transposed && w.views.size() == 1 &&
                      w.views[0].getPermutation() == ArrayRef<int64_t>{1, 0};
        weight = readTransposed && !batched ? weightLayout(w.decode.getInput())
                                            : Value();
        auto layout = weight ? cast<RankedTensorType>(weight.getType())
                             : RankedTensorType();
        if (!layout || layout.getRank() != 3 ||
            layout.getDimSize(0) != slices || layout.getDimSize(2) % 16 ||
            layout.getDimSize(1) * layout.getDimSize(2) / 16 != paddedK) {
          op->emitError("goban weight must be stored in the smatmul B layout "
                        "[ceil(N / 16), g, G * 16]");
          return signalPassFailure();
        }
        lines = layout.getDimSize(1);
        groups = layout.getDimSize(2) / 16;
      } else {
        for (int64_t g = 1; g <= maxGroups; ++g)
          if (paddedK % (16 * g) == 0 && paddedK / g <= depth)
            groups = g;
        lines = paddedK / groups;
      }
      if (groups > maxGroups || lines % 16 || lines > depth) {
        op->emitError("goban K split exceeds the banks");
        return signalPassFailure();
      }
      int64_t slicesPerLine = lines / 16;
      // M tiles: R rows, a multiple of tileRows unless M is one row.
      int64_t layoutRows = m == 1 ? 1 : tileRows;
      int64_t rows = 1;
      if (m > 1) {
        int64_t most =
            std::min(depth / 4, depth / slicesPerLine) / tileRows * tileRows;
        if (most < tileRows) {
          op->emitError("goban A tile cannot hold tileRows rows");
          return signalPassFailure();
        }
        int64_t count = (m + most - 1) / most;
        rows = ((m + count - 1) / count + tileRows - 1) / tileRows * tileRows;
      }
      int64_t tiles = (m + rows - 1) / rows;
      int64_t tileLines = rows * slicesPerLine, width = groups * 16;

      auto alloc = [&](ArrayRef<int64_t> shape, Type element,
                       TypedAttr initial) -> Value {
        Value value =
            memref::AllocOp::create(b, loc, MemRefType::get(shape, element));
        if (initial)
          linalg::FillOp::create(
              b, loc, ValueRange{arith::ConstantOp::create(b, loc, initial)},
              value);
        return value;
      };
      Value zeros = alloc({16}, b.getI32Type(), b.getI32IntegerAttr(0));
      // The product of the two steps of every output element, sixteen
      // columns per output slice.
      Value scales =
          alloc({batch, m, slices, 16}, b.getF32Type(), b.getF32FloatAttr(1));
      auto coords = [&](Operand input, bool rhs, Value bt, Value r, Value kk) {
        SmallVector<Value> at;
        if (batched)
          at.push_back(bt);
        at.push_back(rhs && !transposed ? kk : r);
        at.push_back(rhs && !transposed ? r : kk);
        return coordinates(input, at);
      };
      loops(b, loc, {batch, m, n}, false, [&](ValueRange at) {
        Value left = broadcast(b, loc, a.decode.getStep(),
                               coords(a, false, at[0], at[1], c(0)));
        Value right = broadcast(b, loc, w.decode.getStep(),
                                coords(w, true, at[0], at[2], c(0)));
        memref::StoreOp::create(
            b, loc, arith::MulFOp::create(b, loc, left, right), scales,
            ValueRange{at[0], at[1],
                       arith::DivUIOp::create(b, loc, at[2], c(16)),
                       arith::RemUIOp::create(b, loc, at[2], c(16))});
      });

      // The FP32 input of an online operand as [batch, rows, columns], in
      // place or staged with zeros past its rows and columns.
      auto source = [&](Operand input, bool rhs, int64_t rowCount,
                        int64_t columns) -> Value {
        Value value = input.quant.getInput();
        int64_t sourceRows = rhs ? k : m, sourceColumns = rhs ? n : k;
        if (input.views.empty() && (!rhs || !transposed) &&
            sourceColumns == columns) {
          Value memory = buffer(b, loc, value);
          memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
          return memref::ReinterpretCastOp::create(
              b, loc,
              MemRefType::get({batch, sourceRows, columns}, b.getF32Type()),
              memory, attr(0),
              ArrayRef<OpFoldResult>{attr(batch), attr(sourceRows),
                                     attr(columns)},
              ArrayRef<OpFoldResult>{attr(sourceRows * columns), attr(columns),
                                     attr(1)});
        }
        Value staged = alloc({batch, rowCount, columns}, b.getF32Type(),
                             b.getF32FloatAttr(0));
        loops(b, loc, {batch, sourceRows, sourceColumns}, false,
              [&](ValueRange at) {
                Value element = tensor::ExtractOp::create(
                    b, loc, value,
                    rhs ? coords(input, true, at[0], at[2], at[1])
                        : coords(input, false, at[0], at[1], at[2]));
                memref::StoreOp::create(b, loc, element, staged, at);
              });
        return staged;
      };
      // Quantizes `values` FP32 values at `input` with the sixteen
      // multipliers at `factors` and writes their `values / 16` INT8 lines
      // at `output`, `stride` lines of 16 bytes apart.
      auto quantize = [&](Value input, Value factors, int64_t values,
                          Value output, int64_t stride) {
        Value in = BankAllocOp::create(b, loc, i64, 1, 1);
        Value factor = BankAllocOp::create(b, loc, i64, 1, 1);
        Value result = BankAllocOp::create(b, loc, i64, 1, 1);
        in = BankMvinOp::create(b, loc, i64, input, in, values / 4, 1);
        factor = BankMvinOp::create(b, loc, i64, factors, factor, 4, 1);
        result = ::buddy::toint8_ball::BankFp32ToInt8Op::create(
            b, loc, i64, in, factor, result, values / 4);
        BankMvoutOp::create(b, loc, i64, output, result, values / 16, stride);
        BankReleaseOp::create(b, loc, in);
        BankReleaseOp::create(b, loc, factor);
        BankReleaseOp::create(b, loc, result);
      };

      // A: [batch, tiles, tile lines, G * 16] INT8, rows past M zero.
      Value lhs = alloc({batch, tiles, tileLines, width}, b.getI8Type(),
                        tiles * rows > m ? b.getI8IntegerAttr(0) : TypedAttr());
      {
        Value input = source(a, false, m, paddedK);
        Value multipliers = alloc({batch, m, 16}, b.getF32Type(), TypedAttr());
        loops(b, loc, {batch, m, 16}, false, [&](ValueRange at) {
          memref::StoreOp::create(
              b, loc,
              broadcast(b, loc, a.quant.getMultiplier(),
                        coords(a, false, at[0], at[1], c(0))),
              multipliers, at);
        });
        // Row r = tile * R + block * T + q of reduction group j, in affine
        // nests: the whole tiles, the whole tileRows blocks of the last
        // tile, its last rows.
        auto quantizeRows = [&](Value bt, Value j, Value tile, Value block,
                                Value q) {
          Value r = add(add(mul(tile, rows), mul(block, layoutRows)), q);
          Value line = add(mul(block, slicesPerLine * layoutRows), q);
          quantize(view(b, loc, input, {bt, r, mul(j, lines)}, lines),
                   view(b, loc, multipliers, {bt, r, attr(0)}, 16), lines,
                   view(b, loc, lhs, {bt, tile, line, mul(j, 16)}, 16),
                   layoutRows * groups);
        };
        int64_t whole = m / rows, rest = m % rows;
        if (whole)
          loops(b, loc, {batch, groups, whole, rows / layoutRows, layoutRows},
                true, [&](ValueRange at) {
                  quantizeRows(at[0], at[1], at[2], at[3], at[4]);
                });
        if (rest / layoutRows)
          loops(b, loc, {batch, groups, rest / layoutRows, layoutRows}, true,
                [&](ValueRange at) {
                  quantizeRows(at[0], at[1], c(whole), at[2], at[3]);
                });
        if (rest % layoutRows)
          loops(b, loc, {batch, groups, rest % layoutRows}, true,
                [&](ValueRange at) {
                  quantizeRows(at[0], at[1], c(whole), c(rest / layoutRows),
                               at[2]);
                });
      }

      // B: the weight in place, or [batch, S, g, G * 16] quantized online
      // one K line at a time (line l of group j), K lines past K zero.
      Value rhs;
      if (weight) {
        rhs = buffer(b, loc, weight);
        rhs.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      } else {
        auto multiplierType =
            cast<RankedTensorType>(w.quant.getMultiplier().getType());
        if (multiplierType.getNumElements() != 1) {
          op->emitError("goban online B needs one multiplier");
          return signalPassFailure();
        }
        rhs = alloc({batch, slices, lines, width}, b.getI8Type(),
                    paddedK > k ? b.getI8IntegerAttr(0) : TypedAttr());
        Value input = source(w, true, k, slices * 16);
        Value multipliers = alloc({16}, b.getF32Type(), TypedAttr());
        loops(b, loc, {16}, false, [&](ValueRange at) {
          memref::StoreOp::create(
              b, loc,
              broadcast(b, loc, w.quant.getMultiplier(), {c(0), c(0), c(0)}),
              multipliers, at);
        });
        auto quantizeLine = [&](Value bt, Value j, Value l) {
          quantize(view(b, loc, input, {bt, add(mul(j, lines), l), attr(0)},
                        slices * 16),
                   multipliers, slices * 16,
                   view(b, loc, rhs, {bt, attr(0), l, mul(j, 16)}, 16),
                   lines * groups);
        };
        if (k / lines)
          loops(b, loc, {batch, k / lines, lines}, true,
                [&](ValueRange at) { quantizeLine(at[0], at[1], at[2]); });
        if (k % lines)
          loops(b, loc, {batch, k % lines}, true, [&](ValueRange at) {
            quantizeLine(at[0], c(k / lines), at[1]);
          });
      }

      // The products: per M tile, A once; per output slice, B through it
      // into an accumulator tile, out to [batch, S, tiles * R, 16] INT32.
      Value sums =
          alloc({batch, slices, tiles * rows, 16}, b.getI32Type(), TypedAttr());
      loops(b, loc, {batch, tiles}, true, [&](ValueRange at) {
        Value ab = BankAllocOp::create(b, loc, i64, 1, groups);
        ab = BankMvinOp::create(
            b, loc, i64,
            view(b, loc, lhs, {at[0], at[1], attr(0), attr(0)}, width), ab,
            tileLines, 1);
        loops(b, loc, {slices}, true, [&](ValueRange slice) {
          Value sum = BankAllocOp::create(b, loc, i64, 1, 1);
          sum = BankMvinOp::create(b, loc, i64, zeros, sum, 4, 1);
          Value wb = BankAllocOp::create(b, loc, i64, 1, groups);
          SmallVector<OpFoldResult> from;
          if (!weight)
            from.push_back(at[0]);
          from.append({slice[0], attr(0), attr(0)});
          wb = BankMvinOp::create(b, loc, i64, view(b, loc, rhs, from, width),
                                  wb, lines, 1);
          sum = ::buddy::smatmul_ball::BankMatmulI8Op::create(
              b, loc, i64, ab, wb, sum, rows, lines);
          BankReleaseOp::create(b, loc, wb);
          BankMvoutOp::create(b, loc, i64,
                              view(b, loc, sums,
                                   {at[0], slice[0], mul(at[1], rows), attr(0)},
                                   16),
                              sum, rows * 4, 1);
          BankReleaseOp::create(b, loc, sum);
        });
        BankReleaseOp::create(b, loc, ab);
      });

      // Each row of sixteen to FP32 in place in the output.
      Value output = buffer(b, loc, destination.getDest()), final = output;
      if (n != slices * 16) {
        SmallVector<int64_t> shape(type.getShape());
        shape.back() = slices * 16;
        output = memref::AllocOp::create(
            b, loc, MemRefType::get(shape, b.getF32Type()));
      }
      loops(b, loc, {batch, m, slices}, true, [&](ValueRange at) {
        Value values = BankAllocOp::create(b, loc, i64, 1, 1);
        values = BankMvinOp::create(
            b, loc, i64, view(b, loc, sums, {at[0], at[2], at[1], attr(0)}, 16),
            values, 4, 1);
        Value factors = BankAllocOp::create(b, loc, i64, 1, 1);
        factors = BankMvinOp::create(
            b, loc, i64,
            view(b, loc, scales, {at[0], at[1], at[2], attr(0)}, 16), factors,
            4, 1);
        Value result = BankAllocOp::create(b, loc, i64, 1, 1);
        result = ::buddy::int2fp_ball::BankInt32ToFp32Op::create(
            b, loc, i64, values, factors, result, 1, false);
        SmallVector<OpFoldResult> offsets;
        if (batched)
          offsets.push_back(at[0]);
        offsets.append({at[1], mul(at[2], 16)});
        BankMvoutOp::create(b, loc, i64, view(b, loc, output, offsets, 16),
                            result, 4, 1);
        BankReleaseOp::create(b, loc, values);
        BankReleaseOp::create(b, loc, factors);
        BankReleaseOp::create(b, loc, result);
      });
      if (output != final) {
        SmallVector<OpFoldResult> offsets(type.getRank(), attr(0)),
            strides(type.getRank(), attr(1)), sizes;
        for (int64_t size : type.getShape())
          sizes.push_back(attr(size));
        Value logical =
            memref::SubViewOp::create(b, loc, output, offsets, sizes, strides);
        auto copy = memref::CopyOp::create(b, loc, logical, final);
        copy->setAttr("core.after", b.getUnitAttr());
      }
      Value written =
          bufferization::ToTensorOp::create(b, loc, type, final, true, true);
      destination.getResult().replaceAllUsesWith(written);
      destination.erase();
      op->erase();
    }
    RewritePatternSet patterns(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    if (failed(lowerGobanBiasOutputs(getOperation())))
      signalPassFailure();
  }
};
PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
