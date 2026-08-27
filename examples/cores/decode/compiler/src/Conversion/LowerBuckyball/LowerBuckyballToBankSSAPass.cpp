//===- LowerBuckyballToBankSSAPass.cpp - Decode bank-SSA lowering ---------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;

namespace {

class MemTransposeToLinalgPattern
    : public OpRewritePattern<::buddy::buckyball::MemTransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::buddy::buckyball::MemTransposeOp op,
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

namespace {

class LowerBuckyballToBankSSAPass
    : public PassWrapper<LowerBuckyballToBankSSAPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToBankSSAPass)

  StringRef getArgument() const final { return "lower-buckyball-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower Decode Buckyball ops to explicit bank-SSA ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MemTransposeToLinalgPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::buddy::registerLowerBuckyballToBankSSAPass() {
  PassRegistration<LowerBuckyballToBankSSAPass>();
}
