//===- LowerBuckyballToBankSSAPass.cpp - Pebble bank-SSA lowering
//----------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;

namespace mlir::buddy {
void populatePebbleCoreBankSSALoweringPatterns(RewritePatternSet &patterns);
void populatePebbleResidentConvRegionToBankSSAPatterns(
    RewritePatternSet &patterns);
} // namespace mlir::buddy

namespace {

class LowerBuckyballToBankSSAPass
    : public PassWrapper<LowerBuckyballToBankSSAPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToBankSSAPass)

  StringRef getArgument() const final { return "lower-buckyball-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower Pebble Buckyball ops to explicit bank-SSA ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                linalg::LinalgDialect, ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet residentPatterns(&getContext());
    mlir::buddy::populatePebbleResidentConvRegionToBankSSAPatterns(
        residentPatterns);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(residentPatterns)))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    mlir::buddy::populatePebbleCoreBankSSALoweringPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    bool illegalMega = false;
    getOperation().walk([&](Operation *op) {
      if (isa<::buddy::buckyball::MegaKernelOp>(op)) {
        op->emitError(
            "MegaKernel region-wide bank SSA lowering is not implemented");
        illegalMega = true;
      } else if (isa<::buddy::buckyball::MegaMatmulOp,
                     ::buddy::buckyball::MegaConv2dOp,
                     ::buddy::buckyball::MegaConv2dDepthwiseOp>(op) &&
                 !op->getParentOfType<::buddy::buckyball::MegaKernelOp>()) {
        op->emitError(
            "MegaKernel stage is only legal inside buckyball.mega_kernel");
        illegalMega = true;
      }
    });
    if (illegalMega)
      signalPassFailure();
  }
};

} // namespace

void mlir::buddy::registerLowerBuckyballToBankSSAPass() {
  PassRegistration<LowerBuckyballToBankSSAPass>();
}
