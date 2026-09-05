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
#include "Trace/TraceDialect.h"

using namespace mlir;

namespace mlir::buddy {
void populatePebbleCoreBankSSALoweringPatterns(RewritePatternSet &patterns,
                                               bool traceMegaStages,
                                               int64_t traceMegaStageStart,
                                               int64_t traceMegaStageLimit);
void populatePebbleResidentConvRegionToBankSSAPatterns(
    RewritePatternSet &patterns, bool traceMegaStages,
    int64_t traceMegaStageStart, int64_t traceMegaStageLimit);
} // namespace mlir::buddy

namespace {

class LowerBuckyballToBankSSAPass
    : public PassWrapper<LowerBuckyballToBankSSAPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToBankSSAPass)
  LowerBuckyballToBankSSAPass() = default;
  LowerBuckyballToBankSSAPass(const LowerBuckyballToBankSSAPass &) {}

  StringRef getArgument() const final { return "lower-buckyball-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower Pebble Buckyball ops to explicit bank-SSA ops.";
  }

  Option<bool> traceMegaStages{
      *this, "trace-mega-stages",
      llvm::cl::desc("Materialize and trace leading MegaKernel stages"),
      llvm::cl::init(false)};
  Option<int64_t> traceMegaStageLimit{
      *this, "trace-mega-stage-limit",
      llvm::cl::desc("Trace only the first N MegaKernel stages (-1 means all)"),
      llvm::cl::init(-1)};
  Option<int64_t> traceMegaStageStart{
      *this, "trace-mega-stage-start",
      llvm::cl::desc("Do not dump MegaKernel stages before this index"),
      llvm::cl::init(0)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                    linalg::LinalgDialect, ::buddy::buckyball::BuckyballDialect,
                    ::buddy::trace::BuddyTraceDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet residentPatterns(&getContext());
    mlir::buddy::populatePebbleResidentConvRegionToBankSSAPatterns(
        residentPatterns, traceMegaStages, traceMegaStageStart,
        traceMegaStageLimit);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(residentPatterns)))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    mlir::buddy::populatePebbleCoreBankSSALoweringPatterns(
        patterns, traceMegaStages, traceMegaStageStart, traceMegaStageLimit);
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
