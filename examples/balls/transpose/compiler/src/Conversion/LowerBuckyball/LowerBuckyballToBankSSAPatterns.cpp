#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateTransposeBallLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns);
} // namespace mlir::buddy

namespace {
class MemTransposeToLinalgPattern : public OpRewritePattern<MemTransposeOp> {
public:
  using OpRewritePattern<MemTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemTransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto input = dyn_cast<MemRefType>(op.getInput().getType());
    auto output = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!input || !output || !input.hasStaticShape() ||
        !output.hasStaticShape() || input.getRank() != 2 ||
        output.getRank() != 2 || output.getShape()[0] != input.getShape()[1] ||
        output.getShape()[1] != input.getShape()[0])
      return op.emitError("requires matching static rank-2 transpose memrefs");
    MLIRContext *context = rewriter.getContext();
    AffineMap identity = AffineMap::getMultiDimIdentityMap(2, context);
    AffineMap transpose = AffineMap::get(
        2, 0, {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(0)},
        context);
    SmallVector<utils::IteratorType> iterators(2,
                                               utils::IteratorType::parallel);
    rewriter.create<linalg::GenericOp>(
        op.getLoc(), TypeRange{}, ValueRange{op.getInput()},
        ValueRange{op.getOutput()}, ArrayRef<AffineMap>{identity, transpose},
        iterators, [](OpBuilder &builder, Location location, ValueRange args) {
          builder.create<linalg::YieldOp>(location, args[0]);
        });
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::buddy::populateTransposeBallLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MemTransposeToLinalgPattern>(patterns.getContext());
}
