//===- LowerPebbleToBankSSAPass.cpp - Pebble bank-SSA lowering ----------===//
//
// Pebble lowers its MegaKernels as resident regions and keeps transposes on
// banks, then lowers the remaining Ball operations through BankSSALowering.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "Conversion/LowerLinalgToBuckyball/LowerLinalgToBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Trace/TraceDialect.h"

using namespace mlir;

namespace mlir::buddy {
void populateMatmulRegionToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleMegaConv2dToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleMemTransposeToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleResidentConvRegionToBankSSAPatterns(
    RewritePatternSet &patterns);
} // namespace mlir::buddy

LogicalResult lowerPebbleLargeConvolutions(func::FuncOp function);

namespace {

class FormPebbleMegaKernelsPass
    : public PassWrapper<FormPebbleMegaKernelsPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormPebbleMegaKernelsPass)
  StringRef getArgument() const final { return "form-pebble-mega-kernels"; }
  StringRef getDescription() const final {
    return "Fuse quantized operators within one Pebble core";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                memref::MemRefDialect, scf::SCFDialect, math::MathDialect,
                ::buddy::buckyball::BuckyballDialect,
                ::buddy::trace::BuddyTraceDialect>();
  }
  void runOnOperation() override {
    if (failed(lowerPebbleLargeConvolutions(getOperation()))) {
      signalPassFailure();
      return;
    }
    DenseMap<Block *, SmallVector<linalg::GenericOp>> blocks;
    getOperation().walk([&](linalg::GenericOp op) {
      for (NamedAttribute attr : op->getAttrs())
        if (attr.getName().strref().starts_with("buckyball.quantized_")) {
          blocks[op->getBlock()].push_back(op);
          break;
        }
    });
    for (auto &entry : blocks) {
      mlir::buddy::prepareQuantizedWeights(entry.second);
      mlir::buddy::formMegaKernelRegions(entry.second);
    }
  }
};

class LowerPebbleToBankSSAPass
    : public PassWrapper<LowerPebbleToBankSSAPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPebbleToBankSSAPass)
  LowerPebbleToBankSSAPass() = default;
  LowerPebbleToBankSSAPass(const LowerPebbleToBankSSAPass &) {}

  StringRef getArgument() const final { return "lower-pebble-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower Pebble Buckyball ops to explicit bank-SSA ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    linalg::LinalgDialect, ::buddy::buckyball::BuckyballDialect,
                    ::buddy::trace::BuddyTraceDialect>();
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
    mlir::buddy::populateMatmulRegionToBankSSAPatterns(patterns);
    mlir::buddy::populatePebbleMegaConv2dToBankSSAPatterns(patterns);
    mlir::buddy::populatePebbleMemTransposeToBankSSAPatterns(patterns);
    mlir::buddy::populateBankSSALoweringPatterns(patterns);
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

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  PassRegistration<LowerPebbleToBankSSAPass>();
  PassRegistration<FormPebbleMegaKernelsPass>();
  return true;
}();
