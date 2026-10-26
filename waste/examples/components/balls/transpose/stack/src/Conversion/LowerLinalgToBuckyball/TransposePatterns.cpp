//===- TransposePatterns.cpp - linalg.transpose -> buckyball.mem_transpose ===//
//
// Rank-2 transposes run on the Ball directly. Higher-rank permutations the
// Ball supports are decomposed into rank-2 transposes of contiguous slices;
// other permutations stay for the CPU.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class TransposeLowering : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Value output = op.getInit();
    Location loc = op.getLoc();
    auto inT = dyn_cast<MemRefType>(input.getType());
    auto outT = dyn_cast<MemRefType>(output.getType());
    if (!inT || !outT || !inT.hasStaticShape() || !outT.hasStaticShape() ||
        inT.getElementType() != outT.getElementType())
      return failure();

    ArrayRef<int64_t> perm = op.getPermutation();
    ArrayRef<int64_t> is = inT.getShape();
    ArrayRef<int64_t> os = outT.getShape();
    Type elem = inT.getElementType();
    SmallVector<Value> temps;

    // The Ball and collapse_shape need identity-layout buffers.
    auto asContig = [&](Value v) -> Value {
      auto t = cast<MemRefType>(v.getType());
      if (t.getLayout().isIdentity())
        return v;
      Value b = rewriter.create<memref::AllocOp>(
          loc, MemRefType::get(t.getShape(), t.getElementType()));
      rewriter.create<memref::CopyOp>(loc, v, b);
      temps.push_back(b);
      return b;
    };
    auto dstContig = [&]() -> Value {
      if (outT.getLayout().isIdentity())
        return output;
      Value b =
          rewriter.create<memref::AllocOp>(loc, MemRefType::get(os, elem));
      temps.push_back(b);
      return b;
    };
    auto collapse = [&](Value src, ArrayRef<int64_t> shape2,
                        ArrayRef<ReassociationIndices> reassoc) -> Value {
      return rewriter.create<memref::CollapseShapeOp>(
          loc, MemRefType::get(shape2, elem), src, reassoc);
    };
    auto finish = [&](Value dst) {
      if (dst != output)
        rewriter.create<memref::CopyOp>(loc, dst, output);
      for (Value t : llvm::reverse(temps))
        rewriter.create<memref::DeallocOp>(loc, t);
      rewriter.eraseOp(op);
      return success();
    };
    auto idx = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };
    auto ofr = [&](Value v) { return OpFoldResult(v); };
    // Copies one slice of a rank-4 source into a buffer and collapses it to
    // rank 2. Returns {collapsed, buffer}; the caller deallocates the buffer.
    auto slice2D =
        [&](Value src4, ArrayRef<OpFoldResult> off, ArrayRef<int64_t> sz4,
            ArrayRef<int64_t> shape2,
            ArrayRef<ReassociationIndices> reassoc) -> std::pair<Value, Value> {
      SmallVector<OpFoldResult> sz = {ofr(idx(sz4[0])), ofr(idx(sz4[1])),
                                      ofr(idx(sz4[2])), ofr(idx(sz4[3]))};
      SmallVector<OpFoldResult> str(4, ofr(idx(1)));
      Value sub = rewriter.create<memref::SubViewOp>(loc, src4, off, sz, str);
      Value buf =
          rewriter.create<memref::AllocOp>(loc, MemRefType::get(sz4, elem));
      rewriter.create<memref::CopyOp>(loc, sub, buf);
      return {collapse(buf, shape2, reassoc), buf};
    };
    // Transposes a rank-2 slice and stores it into a slice of the destination.
    auto transposeStore = [&](Value in2, Value dst4, ArrayRef<OpFoldResult> off,
                              ArrayRef<int64_t> outSz4,
                              ArrayRef<int64_t> shape2,
                              ArrayRef<ReassociationIndices> reassoc) {
      Value outBuf =
          rewriter.create<memref::AllocOp>(loc, MemRefType::get(outSz4, elem));
      rewriter.create<MemTransposeOp>(loc, in2,
                                      collapse(outBuf, shape2, reassoc));
      SmallVector<OpFoldResult> sz = {ofr(idx(outSz4[0])), ofr(idx(outSz4[1])),
                                      ofr(idx(outSz4[2])), ofr(idx(outSz4[3]))};
      SmallVector<OpFoldResult> str(4, ofr(idx(1)));
      rewriter.create<memref::CopyOp>(
          loc, outBuf,
          rewriter.create<memref::SubViewOp>(loc, dst4, off, sz, str));
      rewriter.create<memref::DeallocOp>(loc, outBuf);
    };
    auto loop = [&](int64_t upper) {
      auto forOp = rewriter.create<scf::ForOp>(loc, idx(0), idx(upper), idx(1));
      rewriter.setInsertionPointToStart(forOp.getBody());
      return forOp;
    };
    // One Ball transpose of the whole tensor viewed as [rows, cols].
    auto whole = [&](ArrayRef<int64_t> in2, ArrayRef<ReassociationIndices> inR,
                     ArrayRef<int64_t> out2,
                     ArrayRef<ReassociationIndices> outR) {
      Value src = asContig(input), dst = dstContig();
      rewriter.create<MemTransposeOp>(loc, collapse(src, in2, inR),
                                      collapse(dst, out2, outR));
      return finish(dst);
    };

    if (inT.getRank() == 2 && perm == ArrayRef<int64_t>({1, 0})) {
      Value dst = dstContig();
      rewriter.create<MemTransposeOp>(loc, asContig(input), dst);
      return finish(dst);
    }

    if (inT.getRank() == 3 && perm == ArrayRef<int64_t>({0, 2, 1}) &&
        is[0] == 1)
      return whole({is[1], is[2]}, {{0, 1}, {2}}, {os[1], os[2]},
                   {{0, 1}, {2}});

    if (inT.getRank() != 4)
      return failure();
    Value c0 = idx(0);

    // Moving a channel across singleton spatial dimensions keeps byte order.
    if (perm == ArrayRef<int64_t>({0, 3, 1, 2}) && is[0] == 1 && is[1] == 1 &&
        is[2] == 1) {
      Value src = asContig(input), dst = dstContig();
      rewriter.create<memref::CopyOp>(
          loc, collapse(src, {1, is[3]}, {{0, 1, 2}, {3}}),
          collapse(dst, {1, is[3]}, {{0}, {1, 2, 3}}));
      return finish(dst);
    }

    // 1x1 OIHW->OHWI is layout-identical after collapsing to [O,I].
    if (perm == ArrayRef<int64_t>({0, 2, 3, 1}) && is[2] == 1 && is[3] == 1) {
      Value src = asContig(input), dst = dstContig();
      rewriter.create<memref::CopyOp>(
          loc, collapse(src, {is[0], is[1]}, {{0}, {1, 2, 3}}),
          collapse(dst, {os[0], os[3]}, {{0}, {1, 2, 3}}));
      return finish(dst);
    }

    // N=1 NCHW->NHWC: [1,C,H,W] -> [1,H,W,C].
    if (perm == ArrayRef<int64_t>({0, 2, 3, 1}) && is[0] == 1)
      return whole({is[1], is[2] * is[3]}, {{0, 1}, {2, 3}},
                   {is[2] * is[3], is[1]}, {{0, 1, 2}, {3}});

    // Weight OIHW->OHWI: one [I, H*W] transpose per O.
    if (perm == ArrayRef<int64_t>({0, 2, 3, 1})) {
      int64_t i = is[1], h = is[2], w = is[3];
      Value src = asContig(input), dst = dstContig();
      auto oLoop = loop(is[0]);
      SmallVector<OpFoldResult> off = {ofr(oLoop.getInductionVar()), ofr(c0),
                                       ofr(c0), ofr(c0)};
      auto [in2, inBuf] =
          slice2D(src, off, {1, i, h, w}, {i, h * w}, {{0, 1}, {2, 3}});
      transposeStore(in2, dst, off, {1, h, w, i}, {h * w, i}, {{0, 1, 2}, {3}});
      rewriter.create<memref::DeallocOp>(loc, inBuf);
      rewriter.setInsertionPointAfter(oLoop);
      return finish(dst);
    }

    // N=1 NHWC->NCHW: [1,A,B,C] -> [1,C,A,B].
    if (perm == ArrayRef<int64_t>({0, 3, 1, 2}) && is[0] == 1)
      return whole({is[1] * is[2], is[3]}, {{0, 1, 2}, {3}},
                   {is[3], is[1] * is[2]}, {{0, 1}, {2, 3}});

    // [N,A,B,C] -> [N,C,B,A].
    if (perm == ArrayRef<int64_t>({0, 3, 2, 1})) {
      int64_t n = is[0], a = is[1], b = is[2], c = is[3];
      if (n == 1)
        return whole({a * b, c}, {{0, 1, 2}, {3}}, {c, a * b},
                     {{0, 1}, {2, 3}});
      Value src = asContig(input), dst = dstContig();
      auto nLoop = loop(n);
      Value ni = nLoop.getInductionVar();
      if (b <= a) {
        auto bLoop = loop(b);
        SmallVector<OpFoldResult> off = {ofr(ni), ofr(c0),
                                         ofr(bLoop.getInductionVar()), ofr(c0)};
        auto [in2, inBuf] =
            slice2D(src, off, {1, a, 1, c}, {a, c}, {{0, 1}, {2, 3}});
        transposeStore(in2, dst, off, {1, c, 1, a}, {c, a}, {{0, 1}, {2, 3}});
        rewriter.create<memref::DeallocOp>(loc, inBuf);
      } else {
        auto aLoop = loop(a);
        Value ai = aLoop.getInductionVar();
        SmallVector<OpFoldResult> inOff = {ofr(ni), ofr(ai), ofr(c0), ofr(c0)};
        SmallVector<OpFoldResult> outOff = {ofr(ni), ofr(c0), ofr(c0), ofr(ai)};
        auto [in2, inBuf] =
            slice2D(src, inOff, {1, 1, b, c}, {b, c}, {{0, 1, 2}, {3}});
        transposeStore(in2, dst, outOff, {1, c, b, 1}, {c, b},
                       {{0, 1}, {2, 3}});
        rewriter.create<memref::DeallocOp>(loc, inBuf);
      }
      rewriter.setInsertionPointAfter(nLoop);
      return finish(dst);
    }

    // [N,A,B,C] -> [N,B,A,C]: one [A,B] transpose per (n, c).
    if (perm == ArrayRef<int64_t>({0, 2, 1, 3})) {
      int64_t a = is[1], b = is[2];
      Value src = asContig(input), dst = dstContig();
      auto nLoop = loop(is[0]);
      auto cLoop = loop(is[3]);
      SmallVector<OpFoldResult> off = {ofr(nLoop.getInductionVar()), ofr(c0),
                                       ofr(c0), ofr(cLoop.getInductionVar())};
      auto [in2, inBuf] =
          slice2D(src, off, {1, a, b, 1}, {a, b}, {{0, 1}, {2, 3}});
      transposeStore(in2, dst, off, {1, b, a, 1}, {b, a}, {{0, 1}, {2, 3}});
      rewriter.create<memref::DeallocOp>(loc, inBuf);
      rewriter.setInsertionPointAfter(nLoop);
      return finish(dst);
    }

    // [N,H,A,B] -> [N,H,B,A]: one [A,B] transpose per (n, h).
    if (perm == ArrayRef<int64_t>({0, 1, 3, 2})) {
      int64_t a = is[2], b = is[3];
      Value src = asContig(input), dst = dstContig();
      auto nLoop = loop(is[0]);
      auto hLoop = loop(is[1]);
      SmallVector<OpFoldResult> off = {ofr(nLoop.getInductionVar()),
                                       ofr(hLoop.getInductionVar()), ofr(c0),
                                       ofr(c0)};
      auto [in2, inBuf] =
          slice2D(src, off, {1, 1, a, b}, {a, b}, {{0, 1, 2}, {3}});
      transposeStore(in2, dst, off, {1, 1, b, a}, {b, a}, {{0, 1, 2}, {3}});
      rewriter.create<memref::DeallocOp>(loc, inBuf);
      rewriter.setInsertionPointAfter(nLoop);
      return finish(dst);
    }

    // Depthwise OIHW->HWIO: [C,1,k,k] -> [k,k,C,1].
    if (perm == ArrayRef<int64_t>({2, 3, 0, 1}) && is[1] == 1)
      return whole({is[0], is[2] * is[3]}, {{0, 1}, {2, 3}},
                   {is[2] * is[3], is[0]}, {{0, 1}, {2, 3}});

    return failure();
  }
};
} // namespace

namespace mlir::buddy {
void populateTransposeBallLinalgPatterns(RewritePatternSet &patterns) {
  patterns.add<TransposeLowering>(patterns.getContext());
}
} // namespace mlir::buddy
