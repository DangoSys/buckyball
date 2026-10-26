//===- Passes.cpp - SMatMulBall passes ------------------------------------===//

#include "Conversion/BallPasses.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::buddy {
void populateSMatMulBallLinalgPatterns(RewritePatternSet &patterns,
                                       int64_t bankDepth);
void populateMatmulRegionToBankSSAPatterns(RewritePatternSet &patterns);
namespace buckyball {
void populateSMatMulBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                      RewritePatternSet &, bool,
                                                      int64_t);
void configureSMatMulBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LinalgToSMatMulPass
    : public mlir::buddy::LinalgToBallPass<LinalgToSMatMulPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToSMatMulPass)
  StringRef getArgument() const final {
    return "convert-linalg-to-smatmul-ball";
  }
  StringRef getDescription() const final {
    return "Lower the Linalg operations SMatMulBall executes.";
  }
  void populate(RewritePatternSet &patterns, int64_t, int64_t bankDepth,
                int64_t) const {
    mlir::buddy::populateSMatMulBallLinalgPatterns(patterns, bankDepth);
  }
};

class LowerSMatMulPass : public mlir::buddy::LowerBallPass<LowerSMatMulPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSMatMulPass)
  StringRef getArgument() const final { return "lower-smatmul-ball"; }
  StringRef getDescription() const final {
    return "Lower SMatMulBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateSMatMulBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureSMatMulBallLegalizeForExportTarget(target,
                                                                        stable);
  }
};

// Lowers MegaKernels made of MatMul stages as one resident bank region.
class LowerMatmulRegionPass
    : public PassWrapper<LowerMatmulRegionPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulRegionPass)
  StringRef getArgument() const final {
    return "lower-matmul-region-to-bank-ssa";
  }
  StringRef getDescription() const final {
    return "Lower MatMul MegaKernels to bank-SSA operations.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                linalg::LinalgDialect, ::buddy::buckyball::BuckyballDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::buddy::populateMatmulRegionToBankSSAPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerSMatMulPass>();
  PassRegistration<LinalgToSMatMulPass>();
  PassRegistration<LowerMatmulRegionPass>();
  return true;
}();
