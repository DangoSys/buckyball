//===- Im2colMatmulToBankSSAPatterns.cpp - f32 im2col_matmul -> Bank* -----===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kLane = 16;
constexpr int64_t kBankDepth = 1024;
constexpr int64_t kMmioBytes = 5 * 1024;
constexpr int64_t kFp2IntSrcGroups = 4;

static int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static LogicalResult rowStrideDivLane(MemRefType ty, int64_t &out) {
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(ty.getStridesAndOffset(strides, offset)) || strides.size() < 2)
    return failure();
  if (ShapedType::isDynamic(strides[0]) || strides[0] <= 0 ||
      strides[0] % kLane != 0)
    return failure();
  if (ShapedType::isDynamic(strides[1]) || strides[1] != 1)
    return failure();
  out = strides[0] / kLane;
  return success();
}

class Im2colMatmulToBankSSAPattern : public OpRewritePattern<Im2colMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto fTy = dyn_cast<MemRefType>(op.getFilter().getType());
    auto oTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inTy || !fTy || !oTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !oTy.hasStaticShape())
      return b.notifyMatchFailure(op, "requires static memrefs");
    if (!inTy.getElementType().isF32() || !fTy.getElementType().isInteger(8) ||
        !oTy.getElementType().isF32())
      return b.notifyMatchFailure(
          op,
          "requires FP32 activations, offline INT8 weights, and FP32 output");
    int64_t dwAddrBase = op.getDwAddr();
    int64_t dwBytes = op.getDwBytes();
    bool perChannel = op.getPerChannel();
    if (dwAddrBase < 16 || dwAddrBase % 4)
      return op.emitError(
          "requires a 4-byte-aligned Dw MMIO byte address from RAX metadata");

    int64_t inSize = op.getInSize();
    int64_t ksize = op.getKsize();
    int64_t n = op.getN();
    int64_t stride = op.getStride();
    int64_t padding = op.getPadding();
    int64_t startRow = op.getStartRow();
    int64_t startCol = op.getStartCol();
    if (stride < 1 || padding < 0 || startRow < 0 || startCol < 0)
      return op.emitError("stride/padding/start out of range");
    if (startRow > padding || startCol > padding)
      return op.emitError("startRow/Col must be <= padding");
    if (ksize < 1 || n < kLane || n > 4096 || n % kLane != 0)
      return op.emitError("ksize/n out of range");
    int64_t requiredDwBytes = perChannel ? n * 4 : 4;
    if (dwBytes < requiredDwBytes || dwAddrBase + requiredDwBytes > kMmioBytes)
      return op.emitError("Dw scale range exceeds Pebble MMIO space");

    int64_t kElems = ksize * ksize;
    int64_t padded = inSize + 2 * padding;
    if (padded < ksize + startRow || padded < ksize + startCol)
      return op.emitError("kernel+start larger than padded input");
    if ((padded - ksize - startRow) % stride != 0 ||
        (padded - ksize - startCol) % stride != 0)
      return op.emitError("inSize/pad/start/stride yield non-integer tile");
    int64_t tileH = (padded - ksize - startRow) / stride + 1;
    int64_t tileW = (padded - ksize - startCol) / stride + 1;
    if (tileH != tileW || tileH < 1)
      return op.emitError(
          "im2col_matmul requires square non-empty output tile");
    int64_t tile = tileH;
    int64_t wins = tile * tile;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t paddedWins = cdiv(wins, kLane) * kLane;
    int64_t paddedK = cdiv(kElems, kLane) * kLane;
    int64_t bRows = paddedK;
    int64_t aRows = (paddedWins / kLane) * paddedK;
    const int64_t outputGroups =
        buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW;
    if (outputGroups <= 0 || outputGroups > 4 || 4 % outputGroups)
      return op.emitError("SMatMulBall outBW must divide four result blocks");
    const int64_t outputRounds = 4 / outputGroups;
    if (aRows > kBankDepth)
      return op.emitError("im2col A layout exceeds bank depth");
    if (paddedWins * outputRounds > kBankDepth)
      return op.emitError("im2col C rows exceed bank depth");

    if (inTy.getShape()[0] != inRows || inTy.getShape()[1] != kLane ||
        fTy.getShape()[0] != bRows || fTy.getShape()[1] != n ||
        oTy.getShape()[0] != wins || oTy.getShape()[1] != n)
      return op.emitError("im2col_matmul packed shape mismatch");

    int64_t strideF = 0;
    if (failed(rowStrideDivLane(fTy, strideF)))
      return op.emitError("filter needs static strided<[row,1]> row%16==0");

    Value daAddr = createI64Const(b, loc, 0);

    Value inIB = allocBank(b, loc, 1, 1);
    Value inFB = allocBank(b, loc, 1, kFp2IntSrcGroups);
    Value loaded = mvinBank(b, loc, op.getInput(), inFB, inRows);
    Value quant =
        b.create<BankFp2IntOp>(loc, inIB.getType(), loaded, inIB,
                               createI64Const(b, loc, inRows), daAddr);
    releaseBank(b, loc, inFB);

    Value patches = allocBank(b, loc, 1, 1);
    Value patch = b.create<BankIm2colOp>(
        loc, patches.getType(), quant, patches, createI64Const(b, loc, inSize),
        createI64Const(b, loc, ksize), createI64Const(b, loc, stride),
        createI64Const(b, loc, padding), b.getI64IntegerAttr(startRow),
        b.getI64IntegerAttr(startCol));
    releaseBank(b, loc, quant);

    Value fIB = allocBank(b, loc, 1, 1);
    uint64_t cfg = matrixRs2((uint64_t)paddedWins, 16, (uint64_t)paddedK);
    Value packed = b.create<memref::AllocOp>(
        loc, MemRefType::get({outputRounds * paddedWins, outputGroups * 4},
                             b.getF32Type()));

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value panelStep = b.create<arith::ConstantIndexOp>(loc, kLane);
    Value panelEnd = b.create<arith::ConstantIndexOp>(loc, n);
    auto panelLoop = b.create<scf::ForOp>(loc, zero, panelEnd, panelStep);
    b.setInsertionPointToStart(panelLoop.getBody());
    Value n0 = panelLoop.getInductionVar();
    {
      Value bTile = b.create<memref::SubViewOp>(
          loc, op.getFilter(), SmallVector<OpFoldResult>{b.getIndexAttr(0), n0},
          SmallVector<OpFoldResult>{b.getIndexAttr(bRows),
                                    b.getIndexAttr(kLane)},
          SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
      Value fLoaded = mvinBank(b, loc, bTile, fIB, bRows, strideF);
      Value accB = allocBank(b, loc, 1, outputGroups);
      Value gemm =
          createBankSMatMul(b, loc, accB.getType(), patch, fLoaded, accB,
                            createI64Const(b, loc, (int64_t)cfg));
      Value dwAddr = createI64Const(b, loc, dwAddrBase);
      if (perChannel) {
        Value n0I64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), n0);
        dwAddr = b.create<arith::AddIOp>(
            loc, dwAddr,
            b.create<arith::MulIOp>(loc, n0I64, createI64Const(b, loc, 4)));
      }
      Value outB = allocBank(b, loc, 1, outputGroups);
      Value fp = perChannel
                     ? b.create<BankInt2FpChannelOp>(
                            loc, outB.getType(), gemm, outB,
                            createI64Const(b, loc, outputRounds * paddedWins),
                            daAddr, dwAddr)
                           .getResult()
                     : b.create<BankInt2FpTensorOp>(
                            loc, outB.getType(), gemm, outB,
                            createI64Const(b, loc, outputRounds * paddedWins),
                            daAddr, dwAddr)
                           .getResult();
      releaseBank(b, loc, accB);
      mvoutBank(b, loc, packed, fp, outputRounds * paddedWins);
      releaseBank(b, loc, outB);
      Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
      Value one = b.create<arith::ConstantIndexOp>(loc, 1);
      Value rounds = b.create<arith::ConstantIndexOp>(loc, outputRounds);
      Value packCols = b.create<arith::ConstantIndexOp>(loc, outputGroups * 4);
      Value sixteen = b.create<arith::ConstantIndexOp>(loc, kLane);
      Value winsV = b.create<arith::ConstantIndexOp>(loc, wins);
      Value n0V = n0;
      auto rowLoop = b.create<scf::ForOp>(loc, zero, winsV, one);
      b.setInsertionPointToStart(rowLoop.getBody());
      Value row = rowLoop.getInductionVar();
      auto colLoop = b.create<scf::ForOp>(loc, zero, sixteen, one);
      b.setInsertionPointToStart(colLoop.getBody());
      Value col = colLoop.getInductionVar();
      Value packedRow = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, row, rounds),
          b.create<arith::DivUIOp>(loc, col, packCols));
      Value packedCol = b.create<arith::RemUIOp>(loc, col, packCols);
      Value value = b.create<memref::LoadOp>(loc, packed,
                                             ValueRange{packedRow, packedCol});
      Value outputCol = b.create<arith::AddIOp>(loc, n0V, col);
      b.create<memref::StoreOp>(loc, value, op.getOutput(),
                                ValueRange{row, outputCol});
      b.setInsertionPointAfter(rowLoop);
    }
    b.setInsertionPointAfter(panelLoop);
    b.create<memref::DeallocOp>(loc, packed);

    releaseBank(b, loc, patch);
    releaseBank(b, loc, fIB);
    releaseBank(b, loc, inIB);
    b.create<FenceOp>(loc);

    b.eraseOp(op);
    return success();
  }
};

class Im2colFatMatmulToBankSSAPattern
    : public OpRewritePattern<Im2colFatMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colFatMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto fTy = dyn_cast<MemRefType>(op.getFilter().getType());
    auto oTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inTy || !fTy || !oTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !oTy.hasStaticShape())
      return b.notifyMatchFailure(op, "requires static memrefs");
    if (!inTy.getElementType().isF32() || !fTy.getElementType().isInteger(8) ||
        !oTy.getElementType().isF32())
      return b.notifyMatchFailure(
          op,
          "requires FP32 activations, offline INT8 weights, and FP32 output");
    int64_t dwAddrBase = op.getDwAddr();
    int64_t dwBytes = op.getDwBytes();
    bool perChannel = op.getPerChannel();
    if (dwAddrBase < 16 || dwAddrBase % 4)
      return op.emitError(
          "requires a 4-byte-aligned Dw MMIO byte address from RAX metadata");

    int64_t inSize = op.getInSize();
    int64_t ksize = op.getKsize();
    int64_t n = op.getN();
    int64_t nCin = op.getNCin();
    int64_t stride = op.getStride();
    int64_t padding = op.getPadding();
    int64_t startRow = op.getStartRow();
    int64_t startCol = op.getStartCol();
    if (stride < 1 || padding < 0 || startRow < 0 || startCol < 0)
      return op.emitError("stride/padding/start out of range");
    if (startRow > padding || startCol > padding)
      return op.emitError("startRow/Col must be <= padding");
    if (ksize < 1 || n < kLane || n > 4096 || n % kLane != 0)
      return op.emitError("ksize/n out of range");
    int64_t requiredDwBytes = perChannel ? n * 4 : 4;
    if (dwBytes < requiredDwBytes || dwAddrBase + requiredDwBytes > kMmioBytes)
      return op.emitError("Dw scale range exceeds Pebble MMIO space");
    if (nCin < 1 || nCin > 256)
      return op.emitError("nCin out of range");
    int64_t kElems = ksize * ksize;
    int64_t padded = inSize + 2 * padding;
    if (padded < ksize + startRow || padded < ksize + startCol)
      return op.emitError("kernel+start larger than padded input");
    if ((padded - ksize - startRow) % stride != 0 ||
        (padded - ksize - startCol) % stride != 0)
      return op.emitError("inSize/pad/start/stride yield non-integer tile");
    int64_t tileH = (padded - ksize - startRow) / stride + 1;
    int64_t tileW = (padded - ksize - startCol) / stride + 1;
    if (tileH != tileW || tileH < 1)
      return op.emitError("im2col_fat_matmul requires square output tile");
    int64_t wins = tileH * tileW;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t paddedWins = cdiv(wins, kLane) * kLane;
    int64_t paddedK = cdiv(kElems, kLane) * kLane;
    int64_t bRows = paddedK;
    int64_t aRowsSingle = (paddedWins / kLane) * paddedK;
    const int64_t outputGroups =
        buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW;
    if (outputGroups <= 0 || outputGroups > 4 || 4 % outputGroups)
      return op.emitError("SMatMulBall outBW must divide four result blocks");
    const int64_t outputRounds = 4 / outputGroups;
    if (aRowsSingle > kBankDepth)
      return op.emitError("im2col A layout exceeds bank depth");
    if (paddedWins * outputRounds > kBankDepth)
      return op.emitError("im2col C rows exceed bank depth");

    if (inTy.getShape()[0] != nCin * inRows || inTy.getShape()[1] != kLane ||
        fTy.getShape()[0] != nCin * bRows || fTy.getShape()[1] != n ||
        oTy.getShape()[0] != wins || oTy.getShape()[1] != n)
      return op.emitError("im2col_fat_matmul packed shape mismatch");

    int64_t strideF = 0;
    if (failed(rowStrideDivLane(fTy, strideF)))
      return op.emitError("filter needs static strided<[row,1]> row%16==0");

    Value daAddr = createI64Const(b, loc, 0);
    const int64_t cLines = outputRounds * paddedWins;
    Value cLinesV = createI64Const(b, loc, cLines);
    uint64_t cfg = matrixRs2((uint64_t)paddedWins, 16, (uint64_t)paddedK);

    Value packed = b.create<memref::AllocOp>(
        loc, MemRefType::get({cLines, outputGroups * 4}, b.getF32Type()));

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value panelEnd = b.create<arith::ConstantIndexOp>(loc, n);
    Value panelStep = b.create<arith::ConstantIndexOp>(loc, kLane);
    auto panelLoop = b.create<scf::ForOp>(loc, zero, panelEnd, panelStep);
    b.setInsertionPointToStart(panelLoop.getBody());
    Value n0 = panelLoop.getInductionVar();
    {
      // Keep only INT32 accumulator live across cin. Per-cin input/filter
      // banks are released before matadd so peak fits bank_num=8 @ outBW=2.
      Value accBank = allocBank(b, loc, 1, outputGroups);

      for (int64_t ci = 0; ci < nCin; ++ci) {
        Value channel = b.create<arith::ConstantIndexOp>(loc, ci);

        Value inputOffset = b.create<arith::MulIOp>(
            loc, channel, b.create<arith::ConstantIndexOp>(loc, inRows));
        Value plane = b.create<memref::SubViewOp>(
            loc, op.getInput(),
            ArrayRef<OpFoldResult>{inputOffset, b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(inRows),
                                   b.getIndexAttr(kLane)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});

        Value inIB = allocBank(b, loc, 1, 1);
        Value inFB = allocBank(b, loc, 1, kFp2IntSrcGroups);
        Value loaded = mvinBank(b, loc, plane, inFB, inRows);
        Value quant =
            b.create<BankFp2IntOp>(loc, inIB.getType(), loaded, inIB,
                                   createI64Const(b, loc, inRows), daAddr);
        releaseBank(b, loc, inFB);

        Value patches = allocBank(b, loc, 1, 1);
        Value patch = b.create<BankIm2colOp>(
            loc, patches.getType(), quant, patches,
            createI64Const(b, loc, inSize), createI64Const(b, loc, ksize),
            createI64Const(b, loc, stride), createI64Const(b, loc, padding),
            b.getI64IntegerAttr(startRow), b.getI64IntegerAttr(startCol));
        releaseBank(b, loc, quant);

        Value filterOffset = b.create<arith::MulIOp>(
            loc, channel, b.create<arith::ConstantIndexOp>(loc, bRows));
        Value bTile = b.create<memref::SubViewOp>(
            loc, op.getFilter(), SmallVector<OpFoldResult>{filterOffset, n0},
            SmallVector<OpFoldResult>{b.getIndexAttr(bRows),
                                      b.getIndexAttr(kLane)},
            SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        Value fIB = allocBank(b, loc, 1, 1);
        Value fLoaded = mvinBank(b, loc, bTile, fIB, bRows, strideF);

        if (ci == 0) {
          (void)createBankSMatMul(b, loc, accBank.getType(), patch, fLoaded,
                                  accBank,
                                  createI64Const(b, loc, (int64_t)cfg));
          releaseBank(b, loc, patch);
          releaseBank(b, loc, fIB);
        } else {
          Value gemmTmp = allocBank(b, loc, 1, outputGroups);
          Value gemm =
              createBankSMatMul(b, loc, gemmTmp.getType(), patch, fLoaded,
                                gemmTmp, createI64Const(b, loc, (int64_t)cfg));
          releaseBank(b, loc, patch);
          releaseBank(b, loc, fIB);

          Value spare = allocBank(b, loc, 1, outputGroups);
          (void)b.create<BankMatAddOp>(loc, spare.getType(), accBank, gemm,
                                       spare, cLinesV);
          releaseBank(b, loc, accBank);
          releaseBank(b, loc, gemmTmp);
          accBank = spare;
        }
      }

      Value dwAddr = createI64Const(b, loc, dwAddrBase);
      if (perChannel) {
        Value n0I64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), n0);
        dwAddr = b.create<arith::AddIOp>(
            loc, dwAddr,
            b.create<arith::MulIOp>(loc, n0I64, createI64Const(b, loc, 4)));
      }
      Value outB = allocBank(b, loc, 1, outputGroups);
      Value fp =
          perChannel
              ? b.create<BankInt2FpChannelOp>(loc, outB.getType(), accBank,
                                              outB, cLinesV, daAddr, dwAddr)
                    .getResult()
              : b.create<BankInt2FpTensorOp>(loc, outB.getType(), accBank, outB,
                                             cLinesV, daAddr, dwAddr)
                    .getResult();
      releaseBank(b, loc, accBank);
      mvoutBank(b, loc, packed, fp, cLines);
      releaseBank(b, loc, outB);
      b.create<FenceOp>(loc);

      Value rounds = b.create<arith::ConstantIndexOp>(loc, outputRounds);
      Value packCols = b.create<arith::ConstantIndexOp>(loc, outputGroups * 4);
      Value winsV = b.create<arith::ConstantIndexOp>(loc, wins);
      auto rowLoop = b.create<scf::ForOp>(loc, zero, winsV, one);
      b.setInsertionPointToStart(rowLoop.getBody());
      Value row = rowLoop.getInductionVar();
      auto colLoop = b.create<scf::ForOp>(loc, zero, panelStep, one);
      b.setInsertionPointToStart(colLoop.getBody());
      Value col = colLoop.getInductionVar();
      Value packedRow = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, row, rounds),
          b.create<arith::DivUIOp>(loc, col, packCols));
      Value packedCol = b.create<arith::RemUIOp>(loc, col, packCols);
      Value value = b.create<memref::LoadOp>(loc, packed,
                                             ValueRange{packedRow, packedCol});
      Value outputCol = b.create<arith::AddIOp>(loc, n0, col);
      b.create<memref::StoreOp>(loc, value, op.getOutput(),
                                ValueRange{row, outputCol});
      b.setInsertionPointAfter(rowLoop);
    }
    b.setInsertionPointAfter(panelLoop);
    b.create<memref::DeallocOp>(loc, packed);
    b.create<FenceOp>(loc);
    b.eraseOp(op);
    return success();
  }
};

class Im2colDepthwiseMatmulToBankSSAPattern
    : public OpRewritePattern<Im2colDepthwiseMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colDepthwiseMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto fTy = dyn_cast<MemRefType>(op.getFilter().getType());
    auto oTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inTy || !fTy || !oTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !oTy.hasStaticShape())
      return b.notifyMatchFailure(op, "requires static memrefs");
    if (!inTy.getElementType().isF32() || !fTy.getElementType().isInteger(8) ||
        !oTy.getElementType().isF32())
      return b.notifyMatchFailure(
          op,
          "requires FP32 activations, offline INT8 weights, and FP32 output");
    int64_t dwAddrBase = op.getDwAddr();
    int64_t dwBytes = op.getDwBytes();
    bool perChannel = op.getPerChannel();
    if (dwAddrBase < 16 || dwAddrBase % 4)
      return op.emitError(
          "requires a 4-byte-aligned Dw MMIO byte address from RAX metadata");

    int64_t inSize = op.getInSize();
    int64_t ksize = op.getKsize();
    int64_t n = op.getN();
    int64_t stride = op.getStride();
    int64_t padding = op.getPadding();
    int64_t startRow = op.getStartRow();
    int64_t startCol = op.getStartCol();
    if (stride < 1 || padding < 0 || startRow < 0 || startCol < 0)
      return op.emitError("stride/padding/start out of range");
    if (startRow > padding || startCol > padding)
      return op.emitError("startRow/Col must be <= padding");
    if (ksize < 1 || n < 1 || n > kLane)
      return op.emitError("ksize/n out of range");
    int64_t requiredDwBytes = perChannel ? n * 4 : 4;
    if (dwBytes < requiredDwBytes || dwAddrBase + requiredDwBytes > kMmioBytes)
      return op.emitError("Dw scale range exceeds Pebble MMIO space");

    int64_t kElems = ksize * ksize;
    int64_t padded = inSize + 2 * padding;
    if (padded < ksize + startRow || padded < ksize + startCol)
      return op.emitError("kernel+start larger than padded input");
    if ((padded - ksize - startRow) % stride != 0 ||
        (padded - ksize - startCol) % stride != 0)
      return op.emitError("inSize/pad/start/stride yield non-integer tile");
    int64_t tileH = (padded - ksize - startRow) / stride + 1;
    int64_t tileW = (padded - ksize - startCol) / stride + 1;
    if (tileH != tileW || tileH < 1)
      return op.emitError("requires square non-empty output tile");
    int64_t tile = tileH;
    int64_t wins = tile * tile;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t paddedWins = cdiv(wins, kLane) * kLane;
    int64_t paddedK = cdiv(kElems, kLane) * kLane;
    int64_t bRows = paddedK;
    const int64_t outputGroups =
        buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW;
    if (outputGroups <= 0 || outputGroups > 4 || 4 % outputGroups)
      return op.emitError("SMatMulBall outBW must divide four result blocks");
    const int64_t outputRounds = 4 / outputGroups;
    if (paddedWins * outputRounds > kBankDepth)
      return op.emitError("im2col_depthwise_matmul output exceeds bank depth");

    if (inTy.getShape()[0] != n * inRows || inTy.getShape()[1] != kLane ||
        fTy.getShape()[0] != n * bRows || fTy.getShape()[1] != kLane ||
        oTy.getShape()[0] != wins || oTy.getShape()[1] != kLane)
      return op.emitError("im2col_depthwise_matmul packed shape mismatch");

    uint64_t cfg = matrixRs2((uint64_t)paddedWins, 16, (uint64_t)paddedK);

    Value daAddr = createI64Const(b, loc, 0);

    Value f0 =
        b.create<arith::ConstantOp>(loc, b.getI8Type(), b.getI8IntegerAttr(0));
    Value fOne = b.create<memref::AllocOp>(
        loc, MemRefType::get({bRows, kLane}, b.getI8Type()));
    Value tmpOut = b.create<memref::AllocOp>(
        loc, MemRefType::get({outputRounds * paddedWins, outputGroups * 4},
                             b.getF32Type()));

    Value inIB = allocBank(b, loc, 1, 1);
    Value patches = allocBank(b, loc, 1, 1);
    Value fIB = allocBank(b, loc, 1, 1);

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value winsV = b.create<arith::ConstantIndexOp>(loc, wins);

    Value channelEnd = b.create<arith::ConstantIndexOp>(loc, n);
    auto channelLoop = b.create<scf::ForOp>(loc, zero, channelEnd, one);
    b.setInsertionPointToStart(channelLoop.getBody());
    Value channel = channelLoop.getInductionVar();
    {
      Value inputOffset = b.create<arith::MulIOp>(
          loc, channel, b.create<arith::ConstantIndexOp>(loc, inRows));
      Value plane = b.create<memref::SubViewOp>(
          loc, op.getInput(),
          ArrayRef<OpFoldResult>{inputOffset, b.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(inRows), b.getIndexAttr(kLane)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});

      b.create<linalg::FillOp>(loc, f0, fOne);
      for (int64_t k = 0; k < kElems; ++k) {
        Value rowV = b.create<arith::ConstantIndexOp>(loc, k);
        Value srcRow = b.create<arith::AddIOp>(
            loc,
            b.create<arith::MulIOp>(
                loc, channel, b.create<arith::ConstantIndexOp>(loc, bRows)),
            rowV);
        Value wt = b.create<memref::LoadOp>(loc, op.getFilter(),
                                            ValueRange{srcRow, channel});
        b.create<memref::StoreOp>(loc, wt, fOne, ValueRange{rowV, zero});
      }

      Value inFB = allocBank(b, loc, 1, kFp2IntSrcGroups);
      Value loaded = mvinBank(b, loc, plane, inFB, inRows);
      Value quant =
          b.create<BankFp2IntOp>(loc, inIB.getType(), loaded, inIB,
                                 createI64Const(b, loc, inRows), daAddr);
      releaseBank(b, loc, inFB);
      Value patch = b.create<BankIm2colOp>(
          loc, patches.getType(), quant, patches,
          createI64Const(b, loc, inSize), createI64Const(b, loc, ksize),
          createI64Const(b, loc, stride), createI64Const(b, loc, padding),
          b.getI64IntegerAttr(startRow), b.getI64IntegerAttr(startCol));

      Value fLoaded = mvinBank(b, loc, fOne, fIB, bRows);
      Value accB = allocBank(b, loc, 1, outputGroups);
      Value computed =
          createBankSMatMul(b, loc, accB.getType(), patch, fLoaded, accB,
                            createI64Const(b, loc, (int64_t)cfg));
      Value dwAddr = createI64Const(b, loc, dwAddrBase);
      if (perChannel) {
        Value channelI64 =
            b.create<arith::IndexCastOp>(loc, b.getI64Type(), channel);
        dwAddr = b.create<arith::AddIOp>(
            loc, dwAddr,
            b.create<arith::MulIOp>(loc, channelI64,
                                    createI64Const(b, loc, 4)));
      }
      Value outB = allocBank(b, loc, 1, outputGroups);
      Value fp = perChannel
                     ? b.create<BankInt2FpChannelOp>(
                            loc, outB.getType(), computed, outB,
                            createI64Const(b, loc, outputRounds * paddedWins),
                            daAddr, dwAddr)
                           .getResult()
                     : b.create<BankInt2FpTensorOp>(
                            loc, outB.getType(), computed, outB,
                            createI64Const(b, loc, outputRounds * paddedWins),
                            daAddr, dwAddr)
                           .getResult();
      releaseBank(b, loc, accB);
      mvoutBank(b, loc, tmpOut, fp, outputRounds * paddedWins);
      releaseBank(b, loc, outB);
      b.create<FenceOp>(loc);

      auto rL = b.create<scf::ForOp>(loc, zero, winsV, one);
      b.setInsertionPointToStart(rL.getBody());
      Value r = rL.getInductionVar();
      Value rounds = b.create<arith::ConstantIndexOp>(loc, outputRounds);
      Value packedRow = b.create<arith::MulIOp>(loc, r, rounds);
      Value v =
          b.create<memref::LoadOp>(loc, tmpOut, ValueRange{packedRow, zero});
      b.create<memref::StoreOp>(loc, v, op.getOutput(), ValueRange{r, channel});
      b.setInsertionPointAfter(rL);
    }
    b.setInsertionPointAfter(channelLoop);

    releaseBank(b, loc, inIB);
    releaseBank(b, loc, patches);
    releaseBank(b, loc, fIB);
    b.create<FenceOp>(loc);

    b.create<memref::DeallocOp>(loc, fOne);
    b.create<memref::DeallocOp>(loc, tmpOut);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::buddy {

void populatePebbleIm2colMatmulToBankSSAPatterns(RewritePatternSet &patterns) {
  patterns.add<Im2colMatmulToBankSSAPattern>(patterns.getContext());
  patterns.add<Im2colFatMatmulToBankSSAPattern>(patterns.getContext());
  patterns.add<Im2colDepthwiseMatmulToBankSSAPattern>(patterns.getContext());
}

} // namespace mlir::buddy
