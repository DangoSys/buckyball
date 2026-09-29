//===- CoreTileLowering.cpp - Pebble Tile to Buckyball lowering ----------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerTileToBuckyball/LowerTileToBuckyball.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;
namespace tile = ::buddy::tile;

namespace mlir::buddy {
void populateTransposeBallTileLoweringPatterns(RewritePatternSet &patterns,
                                               int64_t bankWidthBytes,
                                               int64_t bankDepth,
                                               int64_t bankNum);
} // namespace mlir::buddy

namespace {

class CpuMatmulLowering : public OpRewritePattern<tile::TileMatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileMatMulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<linalg::MatmulOp>(
        op.getLoc(), ValueRange{op.getAMemArray(), op.getBMemArray()},
        ValueRange{op.getCMemArray()});
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerTileToBuckyballPass
    : public PassWrapper<LowerTileToBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToBuckyballPass)

  StringRef getArgument() const final { return "convert-tile-to-buckyball"; }
  StringRef getDescription() const final {
    return "Convert explicit Pebble Tile kernels to Buckyball";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tile::TileDialect, BuckyballDialect, func::FuncDialect,
                    memref::MemRefDialect, arith::ArithDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    const auto &targetConfig = buckyball_target::getBuckyballTarget();
    ConversionTarget target(getContext());
    target.addLegalDialect<BuckyballDialect, memref::MemRefDialect,
                           arith::ArithDialect, scf::SCFDialect,
                           func::FuncDialect, linalg::LinalgDialect>();
    target.addIllegalDialect<tile::TileDialect>();

    RewritePatternSet patterns(&getContext());
    mlir::buddy::populateTransposeBallTileLoweringPatterns(
        patterns, targetConfig.bankWidthBits / 8, targetConfig.bankDepth,
        targetConfig.bankNum);
    mlir::buddy::populateQuantizedKernelTileLoweringPatterns(patterns);
    patterns.add<CpuMatmulLowering>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::buddy {
void registerLowerTileToBuckyballPass() {
  PassRegistration<LowerTileToBuckyballPass>();
}
} // namespace mlir::buddy
