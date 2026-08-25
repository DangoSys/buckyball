//===- MemTransposeToBankSSAPatterns.cpp - mem_transpose -> Bank* ---------===//
//
// Bank-unit transpose (same model as transpose ctests):
//   pad to bank-row width -> contiguous mvin(rows,stride=1)
//   -> transpose(rows) -> mvout(rows,stride=1)
// Dest bank holds transposed data as stacked bank rows of width
//   W = groups * (bankWidthBytes / elemBytes).
// That "16" for i8 is elems-per-bank-row from bank width, not a 16x16 tile.
//
// Packing: iter is always the mvin row count. For fat [R,C] with R<=maxW,
// pack long dim C as iter (up to bankDepth) so we do not emit iter=R tiny ops.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Utils/BankUtils.h"

#include <algorithm>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kBankWidthBytes = 16;
constexpr int64_t kBankDepth = 1024;
constexpr int64_t kMaxGroups = 4;

static int64_t alignUp(int64_t x, int64_t a) {
  if (a <= 0)
    return x;
  return (x + a - 1) / a * a;
}

static size_t elemsPerBankRow(Type elemType) {
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return 0;
  return kBankWidthBytes / (bitWidth / 8);
}

static Value zeroConst(OpBuilder &b, Location loc, Type elemTy) {
  if (auto it = dyn_cast<IntegerType>(elemTy))
    return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(it, 0));
  if (isa<Float32Type>(elemTy))
    return b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0f));
  return {};
}

static void emitBankTranspose(OpBuilder &b, Location loc, Value inContig,
                              Value outContig, int64_t rows, int64_t groups,
                              int64_t elemBits) {
  // CPU packed inContig before mvin; CPU reads outContig after mvout.
  // FPGA RoCC is async — fence both edges (bemu is sync so hid this).
  b.create<FenceOp>(loc);
  Value src = allocBank(b, loc, 1, groups);
  Value dst = allocBank(b, loc, 1, groups);
  Value loaded = mvinBank(b, loc, inContig, src, rows, /*stride=*/1);
  Value transposed = b.create<BankTransposeOp>(
      loc, dst.getType(), loaded, dst, createI64Const(b, loc, rows),
      createI64Const(b, loc, elemBits));
  mvoutBank(b, loc, outContig, transposed, rows, /*stride=*/1);
  b.create<FenceOp>(loc);
  releaseBank(b, loc, loaded);
  releaseBank(b, loc, transposed);
}

// dst[i,j] = src[j, c0+i]; dst:[h,rows] src:[rows,cols]
static void gatherCols(OpBuilder &b, Location loc, Value src, Value dst,
                       int64_t rows, int64_t c0, int64_t h) {
  Value c0v = b.create<arith::ConstantIndexOp>(loc, c0);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value hUb = b.create<arith::ConstantIndexOp>(loc, h);
  Value rUb = b.create<arith::ConstantIndexOp>(loc, rows);
  auto iLoop = b.create<scf::ForOp>(loc, zero, hUb, one);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(iLoop.getBody());
    Value i = iLoop.getInductionVar();
    Value col = b.create<arith::AddIOp>(loc, c0v, i);
    auto jLoop = b.create<scf::ForOp>(loc, zero, rUb, one);
    b.setInsertionPointToStart(jLoop.getBody());
    Value j = jLoop.getInductionVar();
    Value v = b.create<memref::LoadOp>(loc, src, ValueRange{j, col});
    b.create<memref::StoreOp>(loc, v, dst, ValueRange{i, j});
  }
}

// dst[c0+i, j] = src[j, i]; src:[rows,h] dst:[cols,rows]
static void scatterCols(OpBuilder &b, Location loc, Value src, Value dst,
                        int64_t rows, int64_t c0, int64_t h) {
  Value c0v = b.create<arith::ConstantIndexOp>(loc, c0);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value hUb = b.create<arith::ConstantIndexOp>(loc, h);
  Value rUb = b.create<arith::ConstantIndexOp>(loc, rows);
  auto iLoop = b.create<scf::ForOp>(loc, zero, hUb, one);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(iLoop.getBody());
    Value i = iLoop.getInductionVar();
    Value row = b.create<arith::AddIOp>(loc, c0v, i);
    auto jLoop = b.create<scf::ForOp>(loc, zero, rUb, one);
    b.setInsertionPointToStart(jLoop.getBody());
    Value j = jLoop.getInductionVar();
    Value v = b.create<memref::LoadOp>(loc, src, ValueRange{j, i});
    b.create<memref::StoreOp>(loc, v, dst, ValueRange{row, j});
  }
}

class MemTransposeToBankSSAPattern : public OpRewritePattern<MemTransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MemTransposeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return op.emitError("requires static input and output memrefs");

    int64_t rows = inputType.getShape()[0];
    int64_t cols = inputType.getShape()[1];
    if (rows <= 0 || cols <= 0)
      return op.emitError("transpose requires positive rows/cols");
    if (outputType.getShape()[0] != cols || outputType.getShape()[1] != rows)
      return op.emitError("output shape must transpose the input shape");

    Type elemTy = inputType.getElementType();
    int64_t elemBits = elemTy.getIntOrFloatBitWidth();
    if (elemBits != 8 && elemBits != 32)
      return op.emitError("transpose only supports 8-bit or 32-bit elements");

    size_t epr = elemsPerBankRow(elemTy);
    if (epr == 0)
      return op.emitError("unsupported transpose element type");

    Value zero = zeroConst(rewriter, loc, elemTy);
    if (!zero)
      return op.emitError("unsupported transpose element type for pad");

    int64_t eprI = (int64_t)epr;
    int64_t rowsPad = alignUp(rows, eprI);
    int64_t colsPad = alignUp(cols, eprI);
    int64_t maxW = eprI * kMaxGroups;

    auto inPadTy = MemRefType::get({rowsPad, colsPad}, elemTy);
    auto outPadTy = MemRefType::get({colsPad, rowsPad}, elemTy);
    Value inPad = rewriter.create<memref::AllocOp>(loc, inPadTy);
    Value outPad = rewriter.create<memref::AllocOp>(loc, outPadTy);
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{inPad});
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{outPad});

    Value inView = rewriter.create<memref::SubViewOp>(
        loc, inPad,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                  rewriter.getIndexAttr(0)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(rows),
                                  rewriter.getIndexAttr(cols)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                  rewriter.getIndexAttr(1)});
    rewriter.create<memref::CopyOp>(loc, input, inView);

    // Fat [R,C] with R fitting in bank width: pack C as iter (long side).
    bool fat = rowsPad <= maxW && colsPad > rowsPad;
    if (fat) {
      if (rowsPad % eprI != 0)
        return op.emitError("fat transpose rowsPad not bank-row aligned");
      int64_t groups = rowsPad / eprI;
      for (int64_t c0 = 0; c0 < colsPad;) {
        int64_t h = std::min(kBankDepth, colsPad - c0);
        auto inContigTy = MemRefType::get({h, rowsPad}, elemTy);
        auto outContigTy = MemRefType::get({rowsPad, h}, elemTy);
        Value inContig = rewriter.create<memref::AllocOp>(loc, inContigTy);
        Value outContig = rewriter.create<memref::AllocOp>(loc, outContigTy);
        gatherCols(rewriter, loc, inPad, inContig, rowsPad, c0, h);
        emitBankTranspose(rewriter, loc, inContig, outContig, h, groups,
                          elemBits);
        scatterCols(rewriter, loc, outContig, outPad, rowsPad, c0, h);
        rewriter.create<memref::DeallocOp>(loc, inContig);
        rewriter.create<memref::DeallocOp>(loc, outContig);
        c0 += h;
      }
    } else {
      for (int64_t r0 = 0; r0 < rowsPad;) {
        int64_t h = std::min(kBankDepth, rowsPad - r0);
        for (int64_t c0 = 0; c0 < colsPad;) {
          int64_t w = std::min(maxW, colsPad - c0);
          if (w % eprI != 0)
            return op.emitError("transpose tile width not bank-row aligned");
          int64_t groups = w / eprI;

          Value inSub = rewriter.create<memref::SubViewOp>(
              loc, inPad,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(r0),
                                        rewriter.getIndexAttr(c0)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(h),
                                        rewriter.getIndexAttr(w)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          auto inContigTy = MemRefType::get({h, w}, elemTy);
          Value inContig = rewriter.create<memref::AllocOp>(loc, inContigTy);
          rewriter.create<memref::CopyOp>(loc, inSub, inContig);

          auto outContigTy = MemRefType::get({w, h}, elemTy);
          Value outContig = rewriter.create<memref::AllocOp>(loc, outContigTy);
          emitBankTranspose(rewriter, loc, inContig, outContig, h, groups,
                            elemBits);

          Value outSub = rewriter.create<memref::SubViewOp>(
              loc, outPad,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(c0),
                                        rewriter.getIndexAttr(r0)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(w),
                                        rewriter.getIndexAttr(h)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          rewriter.create<memref::CopyOp>(loc, outContig, outSub);
          rewriter.create<memref::DeallocOp>(loc, inContig);
          rewriter.create<memref::DeallocOp>(loc, outContig);
          c0 += w;
        }
        r0 += h;
      }
    }

    Value outView = rewriter.create<memref::SubViewOp>(
        loc, outPad,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                  rewriter.getIndexAttr(0)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(cols),
                                  rewriter.getIndexAttr(rows)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                  rewriter.getIndexAttr(1)});
    rewriter.create<memref::CopyOp>(loc, outView, output);
    rewriter.create<memref::DeallocOp>(loc, inPad);
    rewriter.create<memref::DeallocOp>(loc, outPad);
    rewriter.create<FenceOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::buddy {

void populatePebbleMemTransposeToBankSSAPatterns(RewritePatternSet &patterns) {
  patterns.add<MemTransposeToBankSSAPattern>(patterns.getContext());
}

} // namespace mlir::buddy
