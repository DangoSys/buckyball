//====- LowerBuckyballToBankSSAPass.cpp - Expand to bank-SSA ops
//-----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace buddy;

namespace {

static Value cstI64(OpBuilder &b, Location loc, uint64_t v) {
  return b.create<arith::ConstantOp>(loc, b.getI64Type(),
                                     b.getI64IntegerAttr(v));
}

static Value cstF32(OpBuilder &b, Location loc, float v) {
  return b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getF32FloatAttr(v));
}

static Value packF32BitsAsI64(OpBuilder &b, Location loc, Value f32Val) {
  Value i32Bits = b.create<arith::BitcastOp>(loc, b.getI32Type(), f32Val);
  return b.create<arith::ExtUIOp>(loc, b.getI64Type(), i32Bits);
}

static Value buildTileAbsMax(PatternRewriter &rewriter, Location loc, Value mem,
                             uint64_t rows, uint64_t cols) {
  auto maxTy = MemRefType::get({1}, rewriter.getF32Type());
  Value maxBuf = rewriter.create<memref::AllocOp>(loc, maxTy);

  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rowsIdx = rewriter.create<arith::ConstantIndexOp>(loc, rows);
  Value colsIdx = rewriter.create<arith::ConstantIndexOp>(loc, cols);
  Value zeroF32 = cstF32(rewriter, loc, 0.0f);

  rewriter.create<memref::StoreOp>(loc, zeroF32, maxBuf, ValueRange{zeroIdx});

  auto rowLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, rowsIdx, oneIdx);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  auto colLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, colsIdx, oneIdx);
  rewriter.setInsertionPointToStart(colLoop.getBody());

  Value elem = rewriter.create<memref::LoadOp>(
      loc, mem,
      ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
  Value neg = rewriter.create<arith::NegFOp>(loc, elem);
  Value abs = rewriter.create<arith::MaximumFOp>(loc, elem, neg);
  Value cur = rewriter.create<memref::LoadOp>(loc, maxBuf, ValueRange{zeroIdx});
  Value upd = rewriter.create<arith::MaximumFOp>(loc, cur, abs);
  rewriter.create<memref::StoreOp>(loc, upd, maxBuf, ValueRange{zeroIdx});

  rewriter.setInsertionPointAfter(rowLoop);
  Value result =
      rewriter.create<memref::LoadOp>(loc, maxBuf, ValueRange{zeroIdx});
  rewriter.create<memref::DeallocOp>(loc, maxBuf);
  return result;
}

static Value buildQuantScale(PatternRewriter &rewriter, Location loc,
                             Value maxAbs) {
  Value zeroF32 = cstF32(rewriter, loc, 0.0f);
  Value oneF32 = cstF32(rewriter, loc, 1.0f);
  Value qmaxF32 = cstF32(rewriter, loc, 127.0f);
  Value hasData = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                                 maxAbs, zeroF32);
  Value scaled = rewriter.create<arith::DivFOp>(loc, qmaxF32, maxAbs);
  return rewriter.create<arith::SelectOp>(loc, hasData, scaled, oneF32);
}

static LogicalResult getStaticRowStrideDiv16(MemRefType ty, uint64_t &out) {
  SmallVector<int64_t, 4> strides;
  int64_t off = 0;
  if (failed(ty.getStridesAndOffset(strides, off)) || strides.size() < 2)
    return failure();
  if (ShapedType::isDynamic(strides[0]) || strides[0] <= 0 ||
      strides[0] % 16 != 0)
    return failure();
  if (ShapedType::isDynamic(strides[1]) || strides[1] != 1)
    return failure();
  out = static_cast<uint64_t>(strides[0] / 16);
  return success();
}

class MatMulToBankSSAPattern : public OpRewritePattern<buckyball::MatMulOp> {
public:
  using OpRewritePattern<buckyball::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(buckyball::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value aMem = op.getAMemArray();
    Value bMem = op.getBMemArray();
    Value cMem = op.getCMemArray();

    auto aTy = dyn_cast<MemRefType>(aMem.getType());
    auto bTy = dyn_cast<MemRefType>(bMem.getType());
    auto cTy = dyn_cast<MemRefType>(cMem.getType());
    if (!aTy || !bTy || !cTy || !aTy.hasStaticShape() ||
        !bTy.hasStaticShape() || !cTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "bb_matmul requires static rank-2 memrefs");

    uint64_t m = aTy.getShape()[0];
    uint64_t k = aTy.getShape()[1];
    uint64_t kb = bTy.getShape()[0];
    uint64_t n = bTy.getShape()[1];
    if (k != kb)
      return rewriter.notifyMatchFailure(op, "inner dimensions must match");
    if (cTy.getShape()[0] != static_cast<int64_t>(m) ||
        cTy.getShape()[1] != static_cast<int64_t>(n))
      return rewriter.notifyMatchFailure(op, "output dimensions must match");
    if (m % 16 != 0 || k % 16 != 0 || n % 16 != 0)
      return rewriter.notifyMatchFailure(
          op, "buckyball.matmul requires M, K and N to be multiples of 16");

    uint64_t strideB = 0;
    uint64_t strideC = 0;
    if (failed(getStaticRowStrideDiv16(bTy, strideB)))
      return rewriter.notifyMatchFailure(
          op, "B requires static strided<[row,1]> and row % 16 == 0");
    if (failed(getStaticRowStrideDiv16(cTy, strideC)))
      return rewriter.notifyMatchFailure(
          op, "C requires static strided<[row,1]> and row % 16 == 0");

    constexpr uint64_t tile = 16;
    uint64_t depthA = tile * (k / tile);
    uint64_t depthB = k;
    uint64_t depthC = tile;

    OpBuilder::InsertionGuard guard(rewriter);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value mUpper = rewriter.create<arith::ConstantIndexOp>(loc, m);
    Value nUpper = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, tile);

    auto mLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, mUpper, step);
    rewriter.setInsertionPointToStart(mLoop.getBody());
    Value mIv = mLoop.getInductionVar();

    auto nLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, nUpper, step);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nIv = nLoop.getInductionVar();

    Value aTile = rewriter.create<memref::SubViewOp>(
        loc, aMem, SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(0)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(tile),
                                  rewriter.getIndexAttr(k)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                  rewriter.getIndexAttr(1)});
    Value bTile = rewriter.create<memref::SubViewOp>(
        loc, bMem, SmallVector<OpFoldResult>{rewriter.getIndexAttr(0), nIv},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(k),
                                  rewriter.getIndexAttr(tile)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                  rewriter.getIndexAttr(1)});
    Value cTile = rewriter.create<memref::SubViewOp>(
        loc, cMem, SmallVector<OpFoldResult>{mIv, nIv},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(tile),
                                  rewriter.getIndexAttr(tile)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                  rewriter.getIndexAttr(1)});

    Value maxA = buildTileAbsMax(rewriter, loc, aTile, tile, k);
    Value maxB = buildTileAbsMax(rewriter, loc, bTile, k, tile);
    Value scaleAF32 = buildQuantScale(rewriter, loc, maxA);
    Value scaleBF32 = buildQuantScale(rewriter, loc, maxB);
    Value scaleABits = packF32BitsAsI64(rewriter, loc, scaleAF32);
    Value scaleBBits = packF32BitsAsI64(rewriter, loc, scaleBF32);
    Value oneF32 = cstF32(rewriter, loc, 1.0f);
    Value scaleProd = rewriter.create<arith::MulFOp>(loc, scaleAF32, scaleBF32);
    Value dequantScaleF32 =
        rewriter.create<arith::DivFOp>(loc, oneF32, scaleProd);
    Value dequantScaleBits = packF32BitsAsI64(rewriter, loc, dequantScaleF32);

    auto aFp32 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    aFp32->setAttr("col", rewriter.getI64IntegerAttr(4));
    auto bFp32 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    bFp32->setAttr("col", rewriter.getI64IntegerAttr(4));
    auto aI8 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto bI8 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto aI8T =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());

    auto aLoad = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), aTile, aFp32.getBank(),
        cstI64(rewriter, loc, depthA), cstI64(rewriter, loc, 1));
    auto bLoad = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), bTile, bFp32.getBank(),
        cstI64(rewriter, loc, depthB), cstI64(rewriter, loc, strideB));

    auto aQuant = rewriter.create<buckyball::BankFp2IntOp>(
        loc, rewriter.getI64Type(), aLoad.getBankOut(), aI8.getBank(),
        cstI64(rewriter, loc, depthA), scaleABits);
    auto bQuant = rewriter.create<buckyball::BankFp2IntOp>(
        loc, rewriter.getI64Type(), bLoad.getBankOut(), bI8.getBank(),
        cstI64(rewriter, loc, depthB), scaleBBits);

    rewriter.create<buckyball::BankReleaseOp>(loc, aLoad.getBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, bLoad.getBankOut());

    auto aTrans = rewriter.create<buckyball::BankTransposeOp>(
        loc, rewriter.getI64Type(), aQuant.getOutBankOut(), aI8T.getBank(),
        cstI64(rewriter, loc, k), cstI64(rewriter, loc, 0));
    rewriter.create<buckyball::BankReleaseOp>(loc, aQuant.getOutBankOut());

    auto cI32 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    cI32->setAttr("col", rewriter.getI64IntegerAttr(4));
    auto cFp32 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    cFp32->setAttr("col", rewriter.getI64IntegerAttr(4));

    auto cMul = rewriter.create<buckyball::BankMulWarp16Op>(
        loc, rewriter.getI64Type(), aTrans.getOutBankOut(),
        bQuant.getOutBankOut(), cI32.getBank(), cstI64(rewriter, loc, k),
        cstI64(rewriter, loc, 0));
    rewriter.create<buckyball::BankReleaseOp>(loc, aTrans.getOutBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, bQuant.getOutBankOut());

    auto cDequant = rewriter.create<buckyball::BankInt2FpOp>(
        loc, rewriter.getI64Type(), cMul.getWrBankOut(), cFp32.getBank(),
        cstI64(rewriter, loc, depthC), dequantScaleBits);
    rewriter.create<buckyball::BankReleaseOp>(loc, cMul.getWrBankOut());

    auto cStore = rewriter.create<buckyball::BankMvoutOp>(
        loc, rewriter.getI64Type(), cTile, cDequant.getOutBankOut(),
        cstI64(rewriter, loc, depthC), cstI64(rewriter, loc, strideC));
    rewriter.create<buckyball::FenceOp>(loc);
    rewriter.create<buckyball::BankReleaseOp>(loc, cStore.getBankOut());

    rewriter.eraseOp(op);
    return success();
  }
};

// TransposeToBankSSAPattern removed - TransposeOp no longer has memref operands

// Im2colToBankSSAPattern removed - Im2colOp no longer has memref operands

// QuantToBankSSAPattern removed - QuantOp no longer has memref operands

// DequantToBankSSAPattern removed - DequantOp no longer has memref operands

class LowerBuckyballToBankSSAPass
    : public PassWrapper<LowerBuckyballToBankSSAPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToBankSSAPass)
  StringRef getArgument() const final { return "lower-buckyball-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower bb_matmul to explicit bank-SSA ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                    buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulToBankSSAPattern>(&getContext());
    // TransposeToBankSSAPattern, Im2colToBankSSAPattern, QuantToBankSSAPattern,
    // DequantToBankSSAPattern removed - these operations no longer have memref
    // operands
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerLowerBuckyballToBankSSAPass() {
  PassRegistration<LowerBuckyballToBankSSAPass>();
}
} // namespace buddy
} // namespace mlir
