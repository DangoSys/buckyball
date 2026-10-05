#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
using namespace ::buddy::buckyball;
namespace {
class BankMXFP8Pattern : public OpRewritePattern<BankMXFP8Op> {
public:
  using OpRewritePattern<BankMXFP8Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankMXFP8Op op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<MXFP8Op>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                             op.getWrBank(), op.getConfig(), op.getFirst(),
                             op.getLast(), op.getOutputBase());
    rewriter.replaceOp(op, op.getWrBank());
    return success();
  }
};

class BankMXFP8WindowPattern : public OpRewritePattern<BankMXFP8WindowOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BankMXFP8WindowOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<MXFP8WindowOp>(
        op.getLoc(), op.getOp1Bank(), op.getOp2Bank(), op.getWrBank(),
        op.getRows(), op.getCols(), op.getCount(), op.getFullK(),
        op.getStartK(), op.getFirst(), op.getLast(), op.getOutputBase());
    rewriter.replaceOp(op, op.getWrBank());
    return success();
  }
};

class BankFP32Pattern : public OpRewritePattern<BankFP32Op> {
public:
  using OpRewritePattern<BankFP32Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankFP32Op op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<FP32Op>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                            op.getWrBank(), op.getConfig(), op.getFirst(),
                            op.getLast(), op.getOutputBase(),
                            op.getFusedAttr());
    rewriter.replaceOp(op, op.getWrBank());
    return success();
  }
};

} // namespace
namespace mlir::buddy {
void populateMxmmBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                PhysicalBankState &) {
  patterns.add<BankMXFP8Pattern, BankMXFP8WindowPattern, BankFP32Pattern>(
      patterns.getContext());
}

} // namespace mlir::buddy
