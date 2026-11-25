//===- ConvertLinalgToAcc.cpp - Goban INT8 products on matmul threads -----===//
//
// convert-linalg-to-acc maps every INT8 product of a thread subgraph to the
// Balls of the matmul thread: smatmul (A [rows, K] . B [K, 16] in INT32),
// toint8 (the online quantization of its FP32 operands). The GP domain
// converts accumulator tiles to FP32 with the product of the two steps.
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
// INT32 (four lines each), stays in its bank. An operator of the GP domain
// (MatmulConversion.cpp) converts and scales all valid rows, scattering
// four-column groups into four banks. One grouped DMA writes the tile into the
// output. Row-dependent factors use the same grouped layout. Rows past M are
// zero rows of A, computed but never converted.
//
// The row_operators of convert-linalg-to-hart launch their operators on the GP
// domain (LowerRowOperators.cpp). The pass numbers the operators it launches
// and adds their entry `operator`, which runs the operator numbered by a7; the
// operators and the entry are built into the operator image by the GP domain's
// compiler.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "int2fpOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "smatmulOps.h"
#include "toint8Ops.h"
#include "transposeOps.h"

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;
LogicalResult lowerGobanBiasOutputs(ModuleOp module);
LogicalResult lowerRowOperators(ModuleOp module,
                                function_ref<int64_t(StringAttr)> number);
LogicalResult lowerGobanTransposeOutputs(ModuleOp module);
std::pair<Value, Value> loadOperators(OpBuilder &b, Location loc,
                                      int64_t program);
func::FuncOp createMatmulConversion(ModuleOp module, bool rowFactors);

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
      loop->setAttr("hart.stream", b.getUnitAttr());
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
    r.insert<AccDialect, arith::ArithDialect, func::FuncDialect,
             math::MathDialect, vector::VectorDialect,
             bufferization::BufferizationDialect, linalg::LinalgDialect,
             memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
             ::buddy::smatmul_ball::SMatMulBallDialect,
             ::buddy::toint8_ball::ToInt8BallDialect,
             ::buddy::int2fp_ball::Int2FpBallDialect,
             ::buddy::transpose_ball::TransposeBallDialect>();
  }
  void runOnOperation() override {
    const auto &target = target::getTarget();
    int64_t depth = target.bankDepth;
    int64_t tileRows = target::getBallParam("SMATMUL_OS", "tileRows");
    // A and B take G banks each, the accumulator one.
    int64_t maxGroups = std::min((target.bankNum - 1) / 2, target.bankNum - 9);
    if (target.bankWidthBits != 128 || maxGroups < 1 || depth < 64) {
      getOperation().emitError("goban needs ten 128-bit banks of 64 lines");
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
        op->emitError("product must write its hart subgraph destination");
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
        value.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(64));
        if (initial)
          linalg::FillOp::create(
              b, loc, ValueRange{arith::ConstantOp::create(b, loc, initial)},
              value);
        return value;
      };
      auto copyRows = [&](Value input, Value output, int64_t rowCount,
                          int64_t columns) {
        loops(b, loc, {batch, rowCount}, true, [&](ValueRange at) {
          SmallVector<OpFoldResult> from, to;
          if (cast<MemRefType>(input.getType()).getRank() == 3)
            from.push_back(at[0]);
          if (cast<MemRefType>(output.getType()).getRank() == 3)
            to.push_back(at[0]);
          from.append({at[1], attr(0)});
          to.append({at[1], attr(0)});
          Value bank = BankAllocOp::create(b, loc, i64, 1, 1);
          bank = BankMvinOp::create(b, loc, i64,
                                    view(b, loc, input, from, columns), bank,
                                    columns / 4, 1);
          bank = BankMvoutOp::create(b, loc, i64,
                                     view(b, loc, output, to, columns), bank,
                                     columns / 4, 1);
          BankReleaseOp::create(b, loc, bank);
        });
      };
      Value zeros = alloc({16}, b.getI32Type(), b.getI32IntegerAttr(0));
      // A scalar activation step gives the same conversion factors for every
      // row.
      int64_t scaleRows = cast<RankedTensorType>(a.decode.getStep().getType())
                                      .getNumElements() == 1
                              ? 1
                              : m;
      Value scales;
      auto coords = [&](Operand input, bool rhs, Value bt, Value r, Value kk) {
        SmallVector<Value> at;
        if (batched)
          at.push_back(bt);
        at.push_back(rhs && !transposed ? kk : r);
        at.push_back(rhs && !transposed ? r : kk);
        return coordinates(input, at);
      };
      auto leftConstant = a.decode.getStep().getDefiningOp<arith::ConstantOp>();
      auto rightConstant =
          w.decode.getStep().getDefiningOp<arith::ConstantOp>();
      if (batch == 1 && leftConstant && rightConstant && a.views.empty() &&
          w.views.empty() &&
          cast<RankedTensorType>(w.decode.getStep().getType())
                  .getNumElements() == n) {
        auto left = cast<DenseFPElementsAttr>(leftConstant.getValue());
        auto right = cast<DenseFPElementsAttr>(rightConstant.getValue());
        SmallVector<APFloat> activations(left.getValues<APFloat>());
        SmallVector<APFloat> weights(right.getValues<APFloat>());
        SmallVector<APFloat> values(batch * scaleRows * slices * 16,
                                    APFloat(1.0f));
        for (int64_t bt = 0; bt < batch; ++bt)
          for (int64_t row = 0; row < scaleRows; ++row)
            for (int64_t column = 0; column < n; ++column) {
              APFloat value = activations[row];
              value.multiply(weights[column], APFloat::rmNearestTiesToEven);
              values[(bt * scaleRows + row) * slices * 16 + column] = value;
            }
        auto type = RankedTensorType::get({batch, scaleRows, slices, 16},
                                          b.getF32Type());
        scales =
            buffer(b, loc,
                   arith::ConstantOp::create(
                       b, loc, type, DenseFPElementsAttr::get(type, values)));
        scales.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      } else {
        scales = alloc({batch, scaleRows, slices, 16}, b.getF32Type(),
                       n == slices * 16 ? TypedAttr() : b.getF32FloatAttr(1));
        loops(b, loc, {batch, scaleRows, n}, false, [&](ValueRange at) {
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
      }

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
        Value staged =
            alloc({batch, rowCount, columns}, b.getF32Type(), TypedAttr());
        if (columns > sourceColumns)
          loops(b, loc, {batch, sourceRows, columns - sourceColumns}, false,
                [&](ValueRange at) {
                  memref::StoreOp::create(
                      b, loc,
                      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0)),
                      staged,
                      ValueRange{at[0], at[1], add(at[2], c(sourceColumns))});
                });
        if (rowCount > sourceRows)
          loops(b, loc, {batch, rowCount - sourceRows, columns}, false,
                [&](ValueRange at) {
                  memref::StoreOp::create(
                      b, loc,
                      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0)),
                      staged,
                      ValueRange{at[0], add(at[1], c(sourceRows)), at[2]});
                });
        if (input.views.empty() && (!rhs || !transposed) &&
            sourceColumns % 4 == 0 && sourceColumns / 4 <= depth) {
          Value memory = buffer(b, loc, value);
          memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
          copyRows(memory, staged, sourceRows, sourceColumns);
          return staged;
        }
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
      // multipliers in `factor` and writes their `values / 16` INT8 lines
      // at `output`, `stride` lines of 16 bytes apart.
      auto quantize = [&](Value input, Value factor, Value inputBank,
                          Value outputBank, int64_t values, Value output,
                          int64_t stride) {
        Value in =
            BankMvinOp::create(b, loc, i64, input, inputBank, values / 4, 1);
        Value result = ::buddy::toint8_ball::BankFp32ToInt8Op::create(
            b, loc, i64, in, factor, outputBank, values / 4);
        BankMvoutOp::create(b, loc, i64, output, result, values / 16, stride);
      };

      // A: [batch, tiles, tile lines, G * 16] INT8, rows past M zero.
      Value lhs, activationInput, activationMultipliers;
      if (m != 1)
        lhs =
            alloc({batch, tiles, tileLines, width}, b.getI8Type(), TypedAttr());
      if ((tiles * rows - m) * paddedK >= 4096) {
        // Large padding tails use a zero bank and contiguous DMA spans.
        // Keep sub-page tails scalar to avoid a bank clear and stream setup.
        Value padding = BankAllocOp::create(b, loc, i64, 1, 1);
        padding = BankClearOp::create(b, loc, i64, padding, 1, 1);
        for (int64_t block = m / layoutRows; block < tiles * rows / layoutRows;
             ++block) {
          int64_t first = std::max<int64_t>(m - block * layoutRows, 0);
          int64_t tile = block / (rows / layoutRows);
          int64_t local = block % (rows / layoutRows);
          loops(b, loc, {batch, slicesPerLine}, true, [&](ValueRange at) {
            Value line = add(c(local * slicesPerLine * layoutRows + first),
                             mul(at[1], layoutRows));
            BankMvoutOp::create(
                b, loc, i64,
                view(b, loc, lhs, {at[0], c(tile), line, c(0)}, width), padding,
                (layoutRows - first) * groups, 1);
          });
        }
        BankReleaseOp::create(b, loc, padding);
      } else if (tiles * rows > m)
        loops(b, loc, {batch, tiles * rows - m, slicesPerLine, width}, false,
              [&](ValueRange at) {
                Value row = add(c(m), at[1]);
                Value tile = arith::DivUIOp::create(b, loc, row, c(rows));
                Value local = arith::RemUIOp::create(b, loc, row, c(rows));
                Value block =
                    arith::DivUIOp::create(b, loc, local, c(layoutRows));
                Value q = arith::RemUIOp::create(b, loc, local, c(layoutRows));
                Value line = add(mul(block, slicesPerLine * layoutRows),
                                 add(q, mul(at[2], layoutRows)));
                memref::StoreOp::create(
                    b, loc,
                    arith::ConstantOp::create(b, loc, b.getI8IntegerAttr(0)),
                    lhs, ValueRange{at[0], tile, line, at[3]});
              });
      {
        Value input = source(a, false, m, paddedK);
        int64_t multiplierRows =
            cast<RankedTensorType>(a.quant.getMultiplier().getType())
                        .getNumElements() == 1
                ? 1
                : m;
        Value multipliers =
            alloc({batch, multiplierRows, 16}, b.getF32Type(), TypedAttr());
        loops(b, loc, {batch, multiplierRows, 16}, false, [&](ValueRange at) {
          memref::StoreOp::create(
              b, loc,
              broadcast(b, loc, a.quant.getMultiplier(),
                        coords(a, false, at[0], at[1], c(0))),
              multipliers, at);
        });
        activationInput = input;
        activationMultipliers = multipliers;
        Value commonFactor;
        if (multiplierRows == 1 && m != 1) {
          commonFactor = BankAllocOp::create(b, loc, i64, 1, 1);
          commonFactor = BankMvinOp::create(
              b, loc, i64,
              view(b, loc, multipliers, {attr(0), attr(0), attr(0)}, 16),
              commonFactor, 4, 1);
        }
        Value inputBank, outputBank;
        if (m != 1) {
          inputBank = BankAllocOp::create(b, loc, i64, 1, 1);
          outputBank = BankAllocOp::create(b, loc, i64, 1, 1);
        }
        bool batchScalar =
            commonFactor && paddedK * 4 <= 4095 && layoutRows * 4 <= depth;
        int64_t dmaRows =
            1 + ((target::getTarget().dmaMaxRanges - 1) * 4096 - 64) /
                    (paddedK * 4);
        // Row r = tile * R + block * T + q of reduction group j, in affine
        // nests: the whole tiles, the whole tileRows blocks of the last
        // tile, its last rows.
        auto quantizeRows = [&](Value bt, Value j, Value tile, Value block,
                                Value q, int64_t count, bool gather,
                                bool packed) {
          if (packed) {
            // Quantize consecutive rows together, then scatter each 16-byte
            // column across those rows. This keeps each DMA maintenance range
            // dense instead of spanning the gaps between packed K columns.
            Value row = add(mul(tile, rows), mul(block, layoutRows));
            // The first bank row occupies the ten-bit op1_base field.
            int64_t chunk = 1 + 1023 / (lines / 4);
            for (int64_t first = 0; first < count; first += chunk) {
              int64_t span = std::min(chunk, count - first);
              loops(b, loc, {span}, true, [&](ValueRange at) {
                BankMvinStridedOp::create(
                    b, loc, i64,
                    view(b, loc, input,
                         {bt, add(row, add(c(first), at[0])), mul(j, lines)},
                         lines),
                    inputBank, mul(at[0], lines / 4), c(lines / 4), c(16), 0, 1,
                    c(16), c(0));
              });
              Value result = ::buddy::toint8_ball::BankFp32ToInt8Op::create(
                  b, loc, i64, inputBank, commonFactor, outputBank,
                  span * lines / 4);
              loops(b, loc, {slicesPerLine}, true, [&](ValueRange at) {
                Value line = add(mul(block, slicesPerLine * layoutRows),
                                 add(mul(at[0], layoutRows), c(first)));
                BankMvoutStridedOp::create(
                    b, loc, i64,
                    view(b, loc, lhs, {bt, tile, line, mul(j, 16)}, 16), result,
                    at[0], c(span), c(groups * 16), 0, slicesPerLine, c(16),
                    c(0));
              });
            }
            return;
          }
          if (gather) {
            // Gather one 16-column chunk of each row. Quantization leaves
            // one INT8 bank line per row, already in the matmul row order.
            Value row = add(mul(tile, rows), mul(block, layoutRows));
            Value line =
                add(mul(block, slicesPerLine * layoutRows), mul(q, layoutRows));
            Value in = inputBank;
            for (int64_t first = 0; first < count; first += dmaRows) {
              int64_t span = std::min(dmaRows, count - first);
              in = BankMvinStridedOp::create(
                  b, loc, i64,
                  view(b, loc, input,
                       {bt, add(row, c(first)), add(mul(j, lines), mul(q, 16))},
                       16),
                  in, c(first * 4), c(span * 4), c(paddedK * 4), 0, 4, c(16),
                  c(2));
            }
            Value result = ::buddy::toint8_ball::BankFp32ToInt8Op::create(
                b, loc, i64, in, commonFactor, outputBank, count * 4);
            BankMvoutOp::create(
                b, loc, i64,
                view(b, loc, lhs, {bt, tile, line, mul(j, 16)}, 16), result,
                count, groups);
            return;
          }
          Value r = add(add(mul(tile, rows), mul(block, layoutRows)), q);
          Value line = add(mul(block, slicesPerLine * layoutRows), q);
          Value factor = commonFactor;
          if (multiplierRows != 1) {
            factor = BankAllocOp::create(b, loc, i64, 1, 1);
            factor = BankMvinOp::create(
                b, loc, i64, view(b, loc, multipliers, {bt, r, attr(0)}, 16),
                factor, 4, 1);
          }
          quantize(view(b, loc, input, {bt, r, mul(j, lines)}, lines), factor,
                   inputBank, outputBank, lines,
                   view(b, loc, lhs, {bt, tile, line, mul(j, 16)}, 16),
                   layoutRows * groups);
          if (multiplierRows != 1)
            BankReleaseOp::create(b, loc, factor);
        };
        int64_t whole = m / rows, rest = m % rows;
        bool batchBlock =
            batchScalar &&
            slicesPerLine * ((layoutRows + dmaRows - 1) / dmaRows + 2) <
                layoutRows * 3;
        int64_t tail = rest % layoutRows;
        bool batchTail =
            batchScalar &&
            slicesPerLine * ((tail + dmaRows - 1) / dmaRows + 2) < tail * 3;
        bool packBlock = commonFactor && !batchScalar && groups * 16 <= 4095 &&
                         layoutRows * lines / 4 <= depth;
        bool packTail = commonFactor && !batchScalar && groups * 16 <= 4095 &&
                        tail >= 4 && tail * lines / 4 <= depth;
        int64_t blockIterations = packBlock    ? 1
                                  : batchBlock ? slicesPerLine
                                               : layoutRows;
        if (whole && m != 1)
          loops(b, loc,
                {batch, groups, whole, rows / layoutRows, blockIterations},
                true, [&](ValueRange at) {
                  quantizeRows(at[0], at[1], at[2], at[3], at[4], layoutRows,
                               batchBlock, packBlock);
                });
        if (rest / layoutRows)
          loops(b, loc, {batch, groups, rest / layoutRows, blockIterations},
                true, [&](ValueRange at) {
                  quantizeRows(at[0], at[1], c(whole), at[2], at[3], layoutRows,
                               batchBlock, packBlock);
                });
        if (rest % layoutRows)
          loops(b, loc,
                {batch, groups,
                 packTail    ? 1
                 : batchTail ? slicesPerLine
                             : tail},
                true, [&](ValueRange at) {
                  quantizeRows(at[0], at[1], c(whole), c(rest / layoutRows),
                               at[2], tail, batchTail, packTail);
                });
        if (commonFactor)
          BankReleaseOp::create(b, loc, commonFactor);
        if (m != 1) {
          BankReleaseOp::create(b, loc, inputBank);
          BankReleaseOp::create(b, loc, outputBank);
        }
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
        Value factor = BankAllocOp::create(b, loc, i64, 1, 1);
        factor = BankMvinOp::create(b, loc, i64, multipliers, factor, 4, 1);
        Value inputBank = BankAllocOp::create(b, loc, i64, 1, 1);
        Value outputBank = BankAllocOp::create(b, loc, i64, 1, 1);
        auto quantizeLine = [&](Value bt, Value j, Value l) {
          quantize(view(b, loc, input, {bt, add(mul(j, lines), l), attr(0)},
                        slices * 16),
                   factor, inputBank, outputBank, slices * 16,
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
        BankReleaseOp::create(b, loc, factor);
        BankReleaseOp::create(b, loc, inputBank);
        BankReleaseOp::create(b, loc, outputBank);
      }

      Value output = buffer(b, loc, destination.getDest()), final = output;
      if (n != slices * 16) {
        SmallVector<int64_t> shape(type.getShape());
        shape.back() = slices * 16;
        output = memref::AllocOp::create(
            b, loc, MemRefType::get(shape, b.getF32Type()));
        output.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(64));
      }
      int64_t program = op->getParentOfType<func::FuncOp>()
                            ->getAttrOfType<IntegerAttr>("hart.program_bank")
                            .getInt();
      int64_t factorGroups = scaleRows == 1 ? 1 : 4;
      int64_t readGroups = 1 + factorGroups;
      auto [codeSize, entry] = loadOperators(b, loc, program);
      int64_t operatorIndex = number(conversion(scaleRows != 1));
      // A descriptor depends only on the valid row count of its M tile.
      // Batches and column slices share it unchanged.
      Value descriptors = alloc({tiles, 12}, b.getI64Type(), TypedAttr());
      auto immediate = [&](uint64_t value) -> Value {
        return arith::ConstantIntOp::create(b, loc, value, 64);
      };
      loops(b, loc, {tiles}, false, [&](ValueRange at) {
        auto store = [&](int64_t field, Value value) {
          memref::StoreOp::create(b, loc, value, descriptors,
                                  ValueRange{at[0], c(field)});
        };
        for (int64_t field = 0; field < 12; ++field)
          store(field, immediate(0));
        store(0, entry);
        store(1, codeSize);
        store(2, immediate(0x40002000));
        store(4, immediate(UINT64_C(1) << 32));
        store(5, immediate(uint64_t(readGroups) << 32));
        Value remaining = arith::SubIOp::create(b, loc, c(m), mul(at[0], rows));
        Value valid = arith::MinUIOp::create(b, loc, remaining, c(rows));
        store(6, arith::IndexCastOp::create(b, loc, b.getI64Type(), valid));
        store(10, immediate(operatorIndex));
      });
      // Keep accumulator rows on device and convert the complete tile. The GP
      // scatters each four-column group into its own output bank.
      auto productTile = [&](Value bt, Value tile, int64_t validRows) {
        Value ab;
        if (m == 1) {
          // A single row already has the grouped matmul layout. Retain each
          // quantized group and join their bank mappings without a DDR copy.
          Value in = BankAllocOp::create(b, loc, i64, 1, 1);
          Value factor = BankAllocOp::create(b, loc, i64, 1, 1);
          factor = BankMvinOp::create(
              b, loc, i64,
              view(b, loc, activationMultipliers, {bt, attr(0), attr(0)}, 16),
              factor, 4, 1);
          for (int64_t group = 0; group < groups; ++group) {
            Value result = BankAllocOp::create(b, loc, i64, 1, 1);
            in = BankMvinOp::create(b, loc, i64,
                                    view(b, loc, activationInput,
                                         {bt, attr(0), attr(group * lines)},
                                         lines),
                                    in, lines / 4, 1);
            result = ::buddy::toint8_ball::BankFp32ToInt8Op::create(
                b, loc, i64, in, factor, result, lines / 4);
            ab = group == 0 ? result
                            : BankTransferOp::create(b, loc, i64, result, ab);
          }
          BankReleaseOp::create(b, loc, in);
          BankReleaseOp::create(b, loc, factor);
        } else {
          ab = BankAllocOp::create(b, loc, i64, 1, groups);
          ab = BankMvinOp::create(
              b, loc, i64,
              view(b, loc, lhs, {bt, tile, attr(0), attr(0)}, width), ab,
              tileLines, 1);
        }
        loops(b, loc, {slices}, true, [&](ValueRange slice) {
          Value sum = BankAllocOp::create(b, loc, i64, 1, 1);
          sum = BankMvinOp::create(b, loc, i64, zeros, sum, 4, 1);
          Value wb = BankAllocOp::create(b, loc, i64, 1, groups);
          SmallVector<OpFoldResult> from;
          if (!weight)
            from.push_back(bt);
          from.append({slice[0], attr(0), attr(0)});
          wb = BankMvinOp::create(b, loc, i64, view(b, loc, rhs, from, width),
                                  wb, lines, 1);
          sum = ::buddy::smatmul_ball::BankMatmulI8Op::create(
              b, loc, i64, ab, wb, sum, rows, lines);
          BankReleaseOp::create(b, loc, wb);
          Value factors = BankAllocOp::create(b, loc, i64, 1, factorGroups);
          factors = BankMvinOp::create(
              b, loc, i64,
              view(b, loc, scales,
                   {bt, scaleRows == 1 ? c(0) : mul(tile, rows), slice[0],
                    attr(0)},
                   16),
              factors, scaleRows == 1 ? 4 : validRows,
              scaleRows == 1 ? 1 : slices);
          Value inputs = BankTransferOp::create(b, loc, i64, factors, sum);
          Value descriptor = view(b, loc, descriptors, {tile, attr(0)}, 12);
          Value writes = BankAllocOp::create(b, loc, i64, 1, 4);
          // The loader latches every descriptor word before GP execution;
          // its storage becomes the first output group during the program.
          writes =
              BankMvinGroupOp::create(b, loc, i64, descriptor, writes, 0, 6, 1);
          writes =
              BankRunOperatorOp::create(b, loc, i64, inputs, writes, program);
          SmallVector<OpFoldResult> offsets;
          if (batched)
            offsets.push_back(bt);
          offsets.append({mul(tile, rows), mul(slice[0], 16)});
          BankMvoutOp::create(b, loc, i64, view(b, loc, output, offsets, 16),
                              writes, validRows, slices);
          BankReleaseOp::create(b, loc, inputs);
          BankReleaseOp::create(b, loc, writes);
        });
        BankReleaseOp::create(b, loc, ab);
      };
      if (m / rows)
        loops(b, loc, {batch, m / rows}, true,
              [&](ValueRange at) { productTile(at[0], at[1], rows); });
      if (m % rows)
        loops(b, loc, {batch}, true, [&](ValueRange at) {
          productTile(at[0], c(m / rows), m % rows);
        });
      MsetOp::create(b, loc, program, false, 0, 0);
      if (output != final) {
        if (n % 4 == 0 && n / 4 <= depth) {
          copyRows(output, final, m, n);
        } else {
          SmallVector<OpFoldResult> offsets(type.getRank(), attr(0)),
              strides(type.getRank(), attr(1)), sizes;
          for (int64_t size : type.getShape())
            sizes.push_back(attr(size));
          Value logical = memref::SubViewOp::create(b, loc, output, offsets,
                                                    sizes, strides);
          auto copy = memref::CopyOp::create(b, loc, logical, final);
          copy->setAttr("hart.after", b.getUnitAttr());
        }
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
    if (failed(lowerGobanBiasOutputs(getOperation())) ||
        failed(lowerRowOperators(
            getOperation(), [&](StringAttr name) { return number(name); })) ||
        failed(lowerGobanTransposeOutputs(getOperation())))
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
  StringAttr conversions[2];
  // The matmul conversion operator, with one factor row per row or one for all.
  StringAttr conversion(bool rowFactors) {
    if (!conversions[rowFactors])
      conversions[rowFactors] =
          createMatmulConversion(getOperation(), rowFactors).getSymNameAttr();
    return conversions[rowFactors];
  }
};
PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
