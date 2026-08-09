//===- MatrixMatmulToBankSSAPatterns.cpp - float matmul -------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

// Float matrix_matmul: keep IR compact with linalg.matmul (accumulates into C).
// Full bank-SSA quantize expansion blows up LLM-sized graphs and crashes
// convert-scf-to-cf (MLIR IRRewrite assert on huge modules).
class MatrixMatmulToBankSSAPattern : public OpRewritePattern<MatrixMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatrixMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto aTy = dyn_cast<MemRefType>(op.getAMemArray().getType());
    auto bTy = dyn_cast<MemRefType>(op.getBMemArray().getType());
    auto cTy = dyn_cast<MemRefType>(op.getCMemArray().getType());
    if (!aTy || !bTy || !cTy || !aTy.hasStaticShape() ||
        !bTy.hasStaticShape() || !cTy.hasStaticShape())
      return b.notifyMatchFailure(op, "requires static memrefs");
    Type elem = aTy.getElementType();
    if (!isa<FloatType>(elem) || bTy.getElementType() != elem ||
        cTy.getElementType() != elem)
      return b.notifyMatchFailure(op,
                                  "pebble bank-ssa compact path is float only");

    int64_t M = aTy.getShape()[0], K = aTy.getShape()[1];
    int64_t Kb = bTy.getShape()[0], N = bTy.getShape()[1];
    if (K != Kb || cTy.getShape()[0] != M || cTy.getShape()[1] != N)
      return op.emitError("matmul shape mismatch");

    b.create<linalg::MatmulOp>(loc,
                               ValueRange{op.getAMemArray(), op.getBMemArray()},
                               ValueRange{op.getCMemArray()});
    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populatePebbleMatrixMatmulToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MatrixMatmulToBankSSAPattern>(patterns.getContext());
}

void mlir::buddy::populatePebbleLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  populatePebbleMatrixMatmulToBankSSAPatterns(patterns);
  populatePebbleIm2colMatmulToBankSSAPatterns(patterns);
  populatePebbleMemTransposeToBankSSAPatterns(patterns);
}
