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
//     to whole tiles.
//
// A product computes Mt x Nt tiles of C, accumulating over the K blocks in
// its destination bank, within the limits of mxint8mm on the target's banks
// of D rows: Mt * S <= D, Nt * S <= D, Mt * Nt / 4 <= D, Nt a multiple of 4.
// Loop order: batch, B tile, A tile, K block. Every A tile meets a B tile
// once: the B tile is moved once per A tile and serves its Mt rows through
// the Ball's own M; with a single K block it stays in its bank for all the
// A tiles.
//
// C is row-major Mt x Nt in its bank and mvout writes its rows from bank row
// 0 with one stride, so a tile lands in place in the [M, N] destination only
// when it spans whole rows (Nt = N) and M needs no padding. Otherwise the
// tiles go to a tile-major buffer [Ntiles][Mpad][Nt] and the core copies
// each column of tiles into the destination after the thread completes.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/BuckyballTargetRegistry.h"
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
#include <cmath>
#include <map>
#include <tuple>

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

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
             ::buddy::mxint8quant_ball::Mxint8quantBallDialect>();
  }
  void runOnOperation() override {
    const auto &target = buckyball_target::getBuckyballTarget();
    int64_t depth = target.bankDepth;
    // A tile of A: the most rows, in whole passes of the array, that keep a
    // square tile of C within the bank (Mt * Nt / 4 <= D).
    int64_t arrayRows =
        buckyball_target::getBuckyballBallParam("MXINT8MM", "rows");
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
    // The quantized operands, by value, side and padded rows: the q, k and
    // v projections of a layer quantize their input once.
    std::map<std::tuple<void *, bool, int64_t>, Value> quantized;
    for (Operation *op : products) {
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
      auto fill = op->getOperand(2).getDefiningOp<linalg::FillOp>();
      bool zero = matchPattern(fill ? fill.getInputs()[0] : op->getOperand(2),
                               m_PosZeroFloat());
      if (!outputType.hasStaticShape() || k <= 0 || k % 32 || !zero ||
          !a.quant || (!w.quant && n % 4)) {
        op->emitError("piplus MXINT8 product needs static shapes, K multiple "
                      "of 32, zero initialization, an online quantized A and "
                      "a weight N multiple of 4");
        return signalPassFailure();
      }
      auto users = op->getResult(0).getUsers();
      auto destination =
          op->getResult(0).hasOneUse()
              ? dyn_cast<bufferization::MaterializeInDestinationOp>(
                    *users.begin())
              : bufferization::MaterializeInDestinationOp();
      if (!destination) {
        op->emitError(
            "piplus product must write its core subgraph destination");
        return signalPassFailure();
      }

      // B: a weight in the layout of the recipe, or quantized online. The K
      // block of an online pair is the recipe's: the largest multiple of 32
      // dividing K up to 128.
      Value weight;
      int64_t kc = 128;
      while (k % kc)
        kc -= 32;
      if (!w.quant) {
        auto slice =
            w.decode.getInput().getDefiningOp<tensor::ExtractSliceOp>();
        auto layout = slice ? cast<RankedTensorType>(slice.getSourceType())
                            : RankedTensorType();
        auto collapse = w.views.size() == 2
                            ? dyn_cast<tensor::CollapseShapeOp>(w.views[0])
                            : tensor::CollapseShapeOp();
        auto transpose = w.views.size() == 2
                             ? dyn_cast<linalg::TransposeOp>(w.views[1])
                             : linalg::TransposeOp();
        bool laid = slice && !batched && w.decode.getAxis() == 2 &&
                    layout.getRank() == 3 && layout.hasStaticShape() &&
                    layout.getDimSize(1) == n &&
                    k % layout.getDimSize(0) == 0 && collapse && transpose &&
                    transpose.getPermutation() == ArrayRef<int64_t>{0, 2, 1} &&
                    slice.hasZeroOffset() && slice.hasUnitStride();
        if (laid) {
          kc = k / layout.getDimSize(0);
          laid = kc % 32 == 0 && layout.getDimSize(2) == packedRows(kc) * 16 &&
                 slice.getType().getShape() ==
                     ArrayRef<int64_t>{layout.getDimSize(0), n, kc + kc / 32};
        }
        if (!laid) {
          op->emitError("piplus weight must be laid out [blocks][N][packed "
                        "block] and read through slice, mxint8_to_fp32 along "
                        "axis 2, transpose [0, 2, 1] and collapse to [K, N]");
          return signalPassFailure();
        }
        weight = buffer(b, loc, slice.getSource());
        weight.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      }
      int64_t blocks = k / kc, packed = packedRows(kc), width = packed * 16;

      // Tiles within mxint8mm: Mt * S, Nt * S and Mt * Nt / 4 bank rows. A
      // weight's N tile divides N; an online B is padded to whole tiles.
      int64_t limit = std::min<int64_t>(depth / packed, 1023);
      int64_t mt = evenTile(m, std::min(limit, tileRows));
      int64_t nLimit = std::min(limit, 4 * depth / mt) / 4 * 4;
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
        // DMA reads each block of a row in place when the operand is the
        // source itself (A) or its transpose of the last two axes (B).
        SmallVector<int64_t> expected;
        if (batched)
          expected.push_back(batch);
        expected.append({count, k});
        auto swap = input.views.size() == 1
                        ? dyn_cast<linalg::TransposeOp>(input.views[0])
                        : linalg::TransposeOp();
        SmallVector<int64_t> swapped =
            llvm::to_vector(llvm::seq<int64_t>(0, int64_t(expected.size())));
        std::swap(swapped[swapped.size() - 1], swapped[swapped.size() - 2]);
        bool direct =
            int64_t(input.quant.getAxis()) == sourceType.getRank() - 1 &&
            sourceType.getShape() == ArrayRef<int64_t>(expected) &&
            (rhs ? swap && swap.getPermutation() == ArrayRef<int64_t>(swapped)
                 : input.views.empty());
        Value values;
        if (direct) {
          Value sourceBuffer = buffer(b, loc, source);
          sourceBuffer.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
          auto layout =
              StridedLayoutAttr::get(b.getContext(), 0, {count * k, kc, k, 1});
          values = memref::ReinterpretCastOp::create(
              b, loc,
              MemRefType::get({batch, blocks, count, kc}, b.getF32Type(),
                              layout),
              sourceBuffer, b.getIndexAttr(0),
              ArrayRef<OpFoldResult>{b.getIndexAttr(batch),
                                     b.getIndexAttr(blocks),
                                     b.getIndexAttr(count), b.getIndexAttr(kc)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(count * k),
                                     b.getIndexAttr(kc), b.getIndexAttr(k),
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
        loops(b, loc, {batch, blocks, count}, true, [&](ValueRange at) {
          Value in = BankAllocOp::create(b, loc, i64, 1, 1);
          Value out = BankAllocOp::create(b, loc, i64, 1, 1);
          in = BankMvinOp::create(
              b, loc, i64, rows(b, loc, values, at.take_front(2), at[2], 1, kc),
              in, kc / 4, 1);
          out = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
              b, loc, i64, in, out, 1, kc);
          out = BankMvoutOp::create(
              b, loc, i64,
              rows(b, loc, result, at.take_front(2), at[2], 1, width), out,
              packed, 1);
          BankReleaseOp::create(b, loc, in);
          BankReleaseOp::create(b, loc, out);
        });
        return result;
      };
      Value lhs = quantize(a, m, mPadded, false);
      Value rhs = weight ? weight : quantize(w, n, nPadded, true);

      // C in place when a tile spans whole rows, else tile-major.
      Value destinationBuffer = buffer(b, loc, destination.getDest());
      bool inPlace = nt == n && mPadded == m;
      Value output = inPlace ? destinationBuffer
                             : memref::AllocOp::create(
                                   b, loc,
                                   MemRefType::get({batch, nTiles, mPadded, nt},
                                                   b.getF32Type()))
                                   .getResult();
      // The B tile at block `block` and column `column`.
      auto rhsTile = [&](Value at, Value block, Value column) {
        Value bank = BankAllocOp::create(b, loc, i64, 1, 1);
        Value tile = weight ? rows(b, loc, rhs, block, column, nt, width)
                            : rows(b, loc, rhs, ValueRange{at, block}, column,
                                   nt, width);
        return BankMvinOp::create(b, loc, i64, tile, bank, nt * packed, 1)
            .getResult();
      };
      loops(b, loc, {batch, nTiles}, true, [&](ValueRange bn) {
        Value column = arith::MulIOp::create(b, loc, bn[1], index(b, loc, nt));
        // A single K block: the B tile stays in its bank for all A tiles.
        Value resident =
            blocks == 1 ? rhsTile(bn[0], index(b, loc, 0), column) : Value();
        loops(b, loc, {mTiles}, true, [&](ValueRange mi) {
          Value row = arith::MulIOp::create(b, loc, mi[0], index(b, loc, mt));
          Value c = BankAllocOp::create(b, loc, i64, 1, 1);
          auto product = [&](Value block, bool accumulate) {
            Value ab = BankAllocOp::create(b, loc, i64, 1, 1);
            ab = BankMvinOp::create(
                b, loc, i64,
                rows(b, loc, lhs, ValueRange{bn[0], block}, row, mt, width), ab,
                mt * packed, 1);
            Value wb = resident ? resident : rhsTile(bn[0], block, column);
            ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
                b, loc, i64, ab, wb, c, mt, nt, kc, accumulate);
            BankReleaseOp::create(b, loc, ab);
            if (!resident)
              BankReleaseOp::create(b, loc, wb);
          };
          product(index(b, loc, 0), false);
          if (blocks > 1)
            loops(b, loc, {blocks - 1}, true, [&](ValueRange part) {
              product(arith::AddIOp::create(b, loc, part[0], index(b, loc, 1)),
                      true);
            });
          Value out;
          if (inPlace) {
            SmallVector<OpFoldResult> offsets, sizes;
            if (batched) {
              offsets.push_back(bn[0]);
              sizes.push_back(b.getIndexAttr(1));
            }
            offsets.append({row, b.getIndexAttr(0)});
            sizes.append({b.getIndexAttr(mt), b.getIndexAttr(n)});
            out = memref::SubViewOp::create(
                b, loc, output, offsets, sizes,
                SmallVector<OpFoldResult>(offsets.size(), b.getIndexAttr(1)));
          } else {
            out = rows(b, loc, output, bn, row, mt, nt);
          }
          BankMvoutOp::create(b, loc, i64, out, c, mt * nt / 4, 1);
          BankReleaseOp::create(b, loc, c);
        });
        if (resident)
          BankReleaseOp::create(b, loc, resident);
      });
      // Each column of tiles to the destination, by the core after the
      // thread completes.
      if (!inPlace) {
        auto copy = [&](Value at, Value tile, int64_t columns) {
          // [m, columns] on both sides: the unit axes are dropped.
          auto view = [&](Value memory, ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) -> Value {
            SmallVector<OpFoldResult> strides(offsets.size(),
                                              b.getIndexAttr(1));
            auto type = memref::SubViewOp::inferRankReducedResultType(
                {m, columns}, cast<MemRefType>(memory.getType()), offsets,
                sizes, strides);
            return memref::SubViewOp::create(b, loc, cast<MemRefType>(type),
                                             memory, offsets, sizes, strides);
          };
          Value first = arith::MulIOp::create(b, loc, tile, index(b, loc, nt));
          SmallVector<OpFoldResult> offsets, sizes;
          if (batched) {
            offsets.push_back(at);
            sizes.push_back(b.getIndexAttr(1));
          }
          offsets.append({b.getIndexAttr(0), first});
          sizes.append({b.getIndexAttr(m), b.getIndexAttr(columns)});
          Value target = view(destinationBuffer, offsets, sizes);
          Value source =
              view(output, {at, tile, b.getIndexAttr(0), b.getIndexAttr(0)},
                   {b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(m),
                    b.getIndexAttr(columns)});
          memref::CopyOp::create(b, loc, source, target)
              ->setAttr("core.after", b.getUnitAttr());
        };
        int64_t full = n / nt;
        loops(b, loc, {batch, full}, false,
              [&](ValueRange at) { copy(at[0], at[1], nt); });
        if (n % nt)
          loops(b, loc, {batch}, false, [&](ValueRange at) {
            copy(at[0], index(b, loc, full), n % nt);
          });
      }
      Value written = bufferization::ToTensorOp::create(
          b, loc, outputType, destinationBuffer, true, true);
      destination.getResult().replaceAllUsesWith(written);
      destination.erase();
      op->erase();
    }
    // Delete the Q/DQ and layout chains consumed by the Ball lowering.
    RewritePatternSet patterns(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
