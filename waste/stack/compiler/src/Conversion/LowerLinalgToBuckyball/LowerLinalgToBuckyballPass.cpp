#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Conversion/LowerLinalgToBuckyball/LowerLinalgToBuckyball.h"

using namespace mlir;

namespace {
// A transpose that repeats an earlier one on the same input reuses its result.
class CancelDuplicateTransposePattern
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.getInput();
    for (Operation *user : in.getUsers()) {
      auto prev = dyn_cast<linalg::TransposeOp>(user);
      if (!prev || prev == op || prev.getPermutation() != op.getPermutation())
        continue;
      auto shape = cast<MemRefType>(op.getInit().getType()).getShape();
      if (cast<MemRefType>(in.getType()).getShape() != shape ||
          cast<MemRefType>(prev.getInit().getType()).getShape() != shape)
        continue;
      if (prev.getInit() != op.getInit())
        rewriter.create<memref::CopyOp>(op.getLoc(), prev.getInit(),
                                        op.getInit());
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// Two transposes whose permutations compose to identity become one copy.
class CancelTransposePairPattern
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp second,
                                PatternRewriter &rewriter) const override {
    auto first = second.getInput().getDefiningOp<linalg::TransposeOp>();
    if (!first || first.getInit() != second.getInput() ||
        !second.getInput().hasOneUse())
      return failure();
    ArrayRef<int64_t> p1 = first.getPermutation();
    ArrayRef<int64_t> p2 = second.getPermutation();
    if (p1.size() != p2.size())
      return failure();
    for (int64_t i = 0; i < (int64_t)p2.size(); ++i)
      if (p1[p2[i]] != i)
        return failure();
    if (cast<MemRefType>(first.getInput().getType()).getShape() !=
        cast<MemRefType>(second.getInit().getType()).getShape())
      return failure();
    if (first.getInput() != second.getInit())
      rewriter.create<memref::CopyOp>(second.getLoc(), first.getInput(),
                                      second.getInit());
    rewriter.eraseOp(second);
    if (first.getInit().use_empty())
      rewriter.eraseOp(first);
    return success();
  }
};

class LowerLinalgToBuckyballPass
    : public PassWrapper<LowerLinalgToBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToBuckyballPass)
  StringRef getArgument() const final { return "convert-linalg-to-buckyball"; }
  StringRef getDescription() const final {
    return "Collect the marked MegaKernel regions. Each Ball claims its own "
           "Linalg operations in convert-linalg-to-<ball>-ball.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::buckyball::BuckyballDialect, func::FuncDialect,
                    memref::MemRefDialect, arith::ArithDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    {
      RewritePatternSet cancel(context);
      cancel.add<CancelTransposePairPattern, CancelDuplicateTransposePattern>(
          context);
      if (failed(applyPatternsGreedily(getOperation(), std::move(cancel))))
        return signalPassFailure();
    }
    ConversionTarget target(*context);
    target.addLegalDialect<::buddy::buckyball::BuckyballDialect,
                           memref::MemRefDialect, arith::ArithDialect,
                           scf::SCFDialect, func::FuncDialect>();
    // Marked stages must lower; other Linalg operations stay unknown.
    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [](linalg::GenericOp op) -> std::optional<bool> {
          if (op->hasAttr("buckyball.mega_kernel"))
            return false;
          return std::nullopt;
        });

    RewritePatternSet patterns(context);
    mlir::buddy::populateMegaKernelLinalgLoweringPatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerLowerLinalgToBuckyballPass() {
  PassRegistration<LowerLinalgToBuckyballPass>();
}
