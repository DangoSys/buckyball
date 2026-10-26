//===- MatMulPatterns.cpp - linalg matmul -> buckyball.smatmul_matmul ----===//
//
// Claims linalg.matmul (plain or transposed B) and linalg.batch_matmul, which
// splits into one matmul per batch. Non-quantized matmul is split into
// bank-sized panels. Quantized matmul retains its full N dimension so bank
// lowering can quantize one activation chunk and hand it to every output panel.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

#include <algorithm>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr size_t kBankLane = 16;

size_t ceilDiv(size_t a, size_t b) { return (a + b - 1) / b; }
size_t aMvinDepthLines(size_t m, size_t k) { return m * (k / kBankLane); }
size_t bMvinDepthLines(size_t k, size_t n) { return k * (n / kBankLane); }
size_t cMvoutDepthLines(size_t m, size_t n) { return m * (n / kBankLane); }

// Copies B [N, K] into a new [K, N] buffer.
Value transposeB(PatternRewriter &rewriter, Location loc, Value b) {
  auto type = cast<MemRefType>(b.getType());
  int64_t n = type.getShape()[0], k = type.getShape()[1];
  Value transposed = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get({k, n}, type.getElementType()));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto nLoop = rewriter.create<scf::ForOp>(
      loc, zero, rewriter.create<arith::ConstantIndexOp>(loc, n), one);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(nLoop.getBody());
  auto kLoop = rewriter.create<scf::ForOp>(
      loc, zero, rewriter.create<arith::ConstantIndexOp>(loc, k), one);
  rewriter.setInsertionPointToStart(kLoop.getBody());
  Value nIv = nLoop.getInductionVar(), kIv = kLoop.getInductionVar();
  Value value = rewriter.create<memref::LoadOp>(loc, b, ValueRange{nIv, kIv});
  rewriter.create<memref::StoreOp>(loc, value, transposed,
                                   ValueRange{kIv, nIv});
  return transposed;
}

// Drops a leading unit dimension of a rank-3 memref.
Value collapseUnitBatch(PatternRewriter &rewriter, Location loc, Value value) {
  auto type = cast<MemRefType>(value.getType());
  if (type.getRank() != 3 || type.getShape()[0] != 1)
    return value;
  return rewriter.create<memref::CollapseShapeOp>(
      loc, value, ArrayRef<ReassociationIndices>{{0, 1}, {2}});
}

class MatMulLowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  MatMulLowering(MLIRContext *context, int64_t bankDepth)
      : OpRewritePattern(context), bankDepth(bankDepth) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Attribute maps = op->getAttr("indexing_maps");
    bool isTransposeB = linalg::MatmulTransposeBOp::isDefaultIndexingMaps(maps);
    if (!linalg::MatmulOp::isDefaultIndexingMaps(maps) && !isTransposeB)
      return failure();
    for (Value operand : op->getOperands()) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.hasStaticShape() ||
          (type.getRank() != 2 &&
           !(type.getRank() == 3 && type.getShape()[0] == 1)))
        return failure();
    }
    Type aElement =
        cast<MemRefType>(op.getInputs()[0].getType()).getElementType();
    Type bElement =
        cast<MemRefType>(op.getInputs()[1].getType()).getElementType();
    Type cElement =
        cast<MemRefType>(op.getOutputs()[0].getType()).getElementType();
    if (!(aElement.isF32() && bElement.isInteger(8) && cElement.isF32()) &&
        !((aElement.isF32() || aElement.isBF16()) && bElement == aElement &&
          cElement == aElement) &&
        !(aElement.isInteger(8) && bElement.isInteger(8) &&
          cElement.isInteger(32)))
      return failure();

    Value aMemArray = collapseUnitBatch(rewriter, loc, op.getInputs()[0]);
    Value bMemArray = collapseUnitBatch(rewriter, loc, op.getInputs()[1]);
    Value cMemArray = collapseUnitBatch(rewriter, loc, op.getOutputs()[0]);
    if (isTransposeB)
      bMemArray = transposeB(rewriter, loc, bMemArray);

    auto aType = cast<MemRefType>(aMemArray.getType());
    auto bType = cast<MemRefType>(bMemArray.getType());
    auto cType = cast<MemRefType>(cMemArray.getType());

    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    auto cShape = cType.getShape();
    size_t M = aShape[aShape.size() - 2];
    size_t K = aShape[aShape.size() - 1];
    size_t N = bShape[bShape.size() - 1];

    if (bShape[bShape.size() - 2] != (int64_t)K ||
        cShape[cShape.size() - 2] != (int64_t)M ||
        cShape[cShape.size() - 1] != (int64_t)N)
      return op.emitError("matmul input/output shapes mismatch");

    const bool isQuantized = aType.getElementType().isF32() &&
                             bType.getElementType().isInteger(8) &&
                             cType.getElementType().isF32();
    const bool isWideFloat =
        (aType.getElementType().isF32() || aType.getElementType().isBF16()) &&
        bType.getElementType() == aType.getElementType() &&
        cType.getElementType() == aType.getElementType();
    const bool isInt8 = aType.getElementType().isInteger(8) &&
                        bType.getElementType().isInteger(8) &&
                        cType.getElementType().isInteger(32);
    if (!isQuantized && !isWideFloat && !isInt8)
      return failure();
    auto dwAddrAttr = op->getAttrOfType<IntegerAttr>("dw_addr");
    auto dwBytesAttr = op->getAttrOfType<IntegerAttr>("dw_bytes");
    auto perChannelAttr = op->getAttrOfType<BoolAttr>("per_channel");
    if (isQuantized && (!dwAddrAttr || !dwBytesAttr || !perChannelAttr ||
                        dwAddrAttr.getInt() < 16))
      return op.emitError("quantized matmul requires RAX Dw metadata");

    constexpr size_t mTileSize = 8;
    size_t M_pad = ceilDiv(M, mTileSize) * mTileSize;
    size_t K_pad = ceilDiv(K, kBankLane) * kBankLane;
    size_t N_pad = ceilDiv(N, kBankLane) * kBankLane;
    bool needPadding = (M_pad != M) || (K_pad != K) || (N_pad != N);

    Value aMemArrayPadded = aMemArray;
    Value bMemArrayPadded = bMemArray;
    Value cMemArrayPadded = cMemArray;

    if (needPadding) {
      auto aPadType = MemRefType::get({(int64_t)M_pad, (int64_t)K_pad},
                                      aType.getElementType());
      auto bPadType = MemRefType::get({(int64_t)K_pad, (int64_t)N_pad},
                                      bType.getElementType());
      auto cPadType = MemRefType::get({(int64_t)M_pad, (int64_t)N_pad},
                                      cType.getElementType());

      aMemArrayPadded = rewriter.create<memref::AllocOp>(loc, aPadType);
      bMemArrayPadded = rewriter.create<memref::AllocOp>(loc, bPadType);
      cMemArrayPadded = rewriter.create<memref::AllocOp>(loc, cPadType);

      Value aZero = rewriter.create<arith::ConstantOp>(
          loc, aType.getElementType(),
          rewriter.getZeroAttr(aType.getElementType()));
      Value bZero = rewriter.create<arith::ConstantOp>(
          loc, bType.getElementType(),
          rewriter.getZeroAttr(bType.getElementType()));
      Value cZero = rewriter.create<arith::ConstantOp>(
          loc, cType.getElementType(),
          rewriter.getZeroAttr(cType.getElementType()));
      rewriter.create<linalg::FillOp>(loc, aZero, aMemArrayPadded);
      rewriter.create<linalg::FillOp>(loc, bZero, bMemArrayPadded);
      rewriter.create<linalg::FillOp>(loc, cZero, cMemArrayPadded);

      Value aView = rewriter.create<memref::SubViewOp>(
          loc, aMemArrayPadded,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(M),
                                    rewriter.getIndexAttr(K)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, aMemArray, aView);

      Value bView = rewriter.create<memref::SubViewOp>(
          loc, bMemArrayPadded,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(K),
                                    rewriter.getIndexAttr(N)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, bMemArray, bView);
    }

    size_t M_tiling = needPadding ? M_pad : M;
    size_t K_tiling = needPadding ? K_pad : K;
    size_t N_tiling = needPadding ? N_pad : N;

    if (bankDepth <= 0 || (size_t)bankDepth % kBankLane != 0)
      return op.emitError("bankDepth must be a positive multiple of bank lane");
    if (M_tiling % mTileSize || K_tiling % kBankLane || N_tiling % kBankLane)
      return op.emitError(
          "M must be a multiple of 8 and K/N must be multiples of bank lane");
    if (isQuantized) {
      const int64_t dwRequired = perChannelAttr.getValue() ? N_tiling * 4 : 4;
      if (dwBytesAttr.getInt() < dwRequired)
        return op.emitError("Dw scale image does not cover padded N");
    }

    // Bank unit on M (depth) and N (lane). Quantized matmul carries one
    // logical K into SMatMul so FP2INT computes one activation scale.
    const size_t nTileSize = kBankLane;
    const size_t kTileSize =
        (isWideFloat || isQuantized) ? K_tiling : kBankLane;
    if (M_tiling % mTileSize != 0)
      return op.emitError("M does not split into bank-depth units");

    if (!isWideFloat && !isQuantized) {
      if (aMvinDepthLines(mTileSize, kTileSize) > (size_t)bankDepth ||
          bMvinDepthLines(kTileSize, nTileSize) > (size_t)bankDepth ||
          cMvoutDepthLines(mTileSize, nTileSize) > (size_t)bankDepth)
        return op.emitError(
            "bank-unit matmul exceeds 8-bit mvin or 32-bit mvout depth");
    }

    const size_t kTileNum = ceilDiv(K_tiling, kTileSize);

    OpBuilder::InsertionGuard guard(rewriter);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value mStepVal = rewriter.create<arith::ConstantIndexOp>(loc, mTileSize);
    Value kStepVal = rewriter.create<arith::ConstantIndexOp>(loc, kTileSize);
    Value mUpperVal = rewriter.create<arith::ConstantIndexOp>(loc, M_tiling);
    Value kUpperVal = rewriter.create<arith::ConstantIndexOp>(loc, K_tiling);

    auto mLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, mUpperVal, mStepVal);
    rewriter.setInsertionPointToStart(mLoop.getBody());
    Value mIv = mLoop.getInductionVar();

    if (isQuantized) {
      Value aTile = rewriter.create<memref::SubViewOp>(
          loc, aMemArrayPadded,
          SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                    rewriter.getIndexAttr(K_tiling)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      Value bTile = rewriter.create<memref::SubViewOp>(
          loc, bMemArrayPadded,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(K_tiling),
                                    rewriter.getIndexAttr(N_tiling)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      Value cTile = rewriter.create<memref::SubViewOp>(
          loc, cMemArrayPadded,
          SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                    rewriter.getIndexAttr(N_tiling)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, cType.getElementType(),
          rewriter.getZeroAttr(cType.getElementType()));
      rewriter.create<linalg::FillOp>(loc, zero, cTile);
      auto smatmul = rewriter.create<SMatMulMatmulOp>(loc, aTile, bTile, cTile);
      smatmul->setAttr("dwAddr", dwAddrAttr);
      smatmul->setAttr("dwBytes", dwBytesAttr);
      smatmul->setAttr("perChannel", perChannelAttr);
    } else {
      // N is static. Non-quantized matmul remains one physical N panel per op.
      for (size_t n0 = 0; n0 < N_tiling; n0 += nTileSize) {
        Value cTile = rewriter.create<memref::SubViewOp>(
            loc, cMemArrayPadded,
            SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(n0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                      rewriter.getIndexAttr(nTileSize)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});

        auto cElemType = cType.getElementType();
        Value zero = rewriter.create<arith::ConstantOp>(
            loc, cElemType, rewriter.getZeroAttr(cElemType));
        rewriter.create<linalg::FillOp>(loc, zero, cTile);

        if (kTileNum == 1) {
          Value aTile = rewriter.create<memref::SubViewOp>(
              loc, aMemArrayPadded,
              SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(0)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                        rewriter.getIndexAttr(kTileSize)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          Value bTile = rewriter.create<memref::SubViewOp>(
              loc, bMemArrayPadded,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(n0)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(kTileSize),
                                        rewriter.getIndexAttr(nTileSize)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          auto smatmul =
              rewriter.create<SMatMulMatmulOp>(loc, aTile, bTile, cTile);
        } else {
          auto kLoop =
              rewriter.create<scf::ForOp>(loc, zeroIdx, kUpperVal, kStepVal);
          rewriter.setInsertionPointToStart(kLoop.getBody());
          Value kIv = kLoop.getInductionVar();
          Value aTile = rewriter.create<memref::SubViewOp>(
              loc, aMemArrayPadded, SmallVector<OpFoldResult>{mIv, kIv},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                        rewriter.getIndexAttr(kTileSize)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          Value bTile = rewriter.create<memref::SubViewOp>(
              loc, bMemArrayPadded,
              SmallVector<OpFoldResult>{kIv, rewriter.getIndexAttr(n0)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(kTileSize),
                                        rewriter.getIndexAttr(nTileSize)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          if (isWideFloat)
            return op.emitError(
                "FP32 matmul must pass full K in one smatmul_matmul");
          // 8-bit legalize overwrites C; accumulate via host partial.
          auto partialTy = MemRefType::get(
              {(int64_t)mTileSize, (int64_t)nTileSize}, cElemType);
          Value partial = rewriter.create<memref::AllocOp>(loc, partialTy);
          rewriter.create<linalg::FillOp>(loc, zero, partial);
          rewriter.create<SMatMulMatmulOp>(loc, aTile, bTile, partial);
          rewriter.create<linalg::ElementwiseOp>(
              loc, ValueRange{cTile, partial}, ValueRange{cTile},
              linalg::ElementwiseKind::add);
          rewriter.create<memref::DeallocOp>(loc, partial);
          rewriter.setInsertionPointAfter(kLoop);
        }
      }
    }

    rewriter.setInsertionPointAfter(mLoop);

    if (needPadding) {
      Value cView = rewriter.create<memref::SubViewOp>(
          loc, cMemArrayPadded,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(M),
                                    rewriter.getIndexAttr(N)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, cView, cMemArray);
      rewriter.create<memref::DeallocOp>(loc, aMemArrayPadded);
      rewriter.create<memref::DeallocOp>(loc, bMemArrayPadded);
      rewriter.create<memref::DeallocOp>(loc, cMemArrayPadded);
    }

    if (isTransposeB)
      rewriter.create<memref::DeallocOp>(loc, bMemArray);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t bankDepth;
};

// One matmul per batch; each is claimed by MatMulLowering.
class BatchMatMulLowering : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Attribute maps = op->getAttr("indexing_maps");
    bool isTransposeB =
        linalg::BatchMatmulTransposeBOp::isDefaultIndexingMaps(maps);
    if (!linalg::BatchMatmulOp::isDefaultIndexingMaps(maps) && !isTransposeB)
      return failure();
    Location loc = op.getLoc();
    Value a = op.getInputs()[0], b = op.getInputs()[1], c = op.getOutputs()[0];
    auto aType = dyn_cast<MemRefType>(a.getType());
    auto bType = dyn_cast<MemRefType>(b.getType());
    auto cType = dyn_cast<MemRefType>(c.getType());
    if (!aType || !bType || !cType || !aType.hasStaticShape() ||
        !bType.hasStaticShape() || !cType.hasStaticShape())
      return failure();
    auto slice = [&](Value value, MemRefType type, int64_t batch) -> Value {
      ArrayRef<int64_t> shape = type.getShape();
      Value sub = rewriter.create<memref::SubViewOp>(
          loc, value, ArrayRef<int64_t>{batch, 0, 0},
          ArrayRef<int64_t>{1, shape[1], shape[2]}, ArrayRef<int64_t>{1, 1, 1});
      return rewriter.create<memref::CollapseShapeOp>(
          loc, sub, ArrayRef<ReassociationIndices>{{0, 1}, {2}});
    };
    for (int64_t batch = 0; batch < aType.getShape()[0]; ++batch) {
      Value bBatch = slice(b, bType, batch);
      if (isTransposeB)
        bBatch = transposeB(rewriter, loc, bBatch);
      rewriter.create<linalg::MatmulOp>(
          loc, ValueRange{slice(a, aType, batch), bBatch},
          ValueRange{slice(c, cType, batch)});
      if (isTransposeB)
        rewriter.create<memref::DeallocOp>(loc, bBatch);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::buddy {
void populateSMatMulBallLinalgPatterns(RewritePatternSet &patterns,
                                       int64_t bankDepth) {
  patterns.add<MatMulLowering>(patterns.getContext(), bankDepth);
  patterns.add<BatchMatMulLowering>(patterns.getContext());
}
} // namespace mlir::buddy
