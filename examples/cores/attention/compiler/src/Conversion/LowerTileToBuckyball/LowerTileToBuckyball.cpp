#include "Target/BuckyballTargetRegistry.h"

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
#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;
namespace tile = ::buddy::tile;

namespace mlir::buddy {
#define BUCKYBALL_TILE_HOOK(BALL)                                              \
  void populate##BALL##TileLoweringPatterns(RewritePatternSet &, int64_t,      \
                                            int64_t, int64_t);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_TILE_HOOK
} // namespace mlir::buddy

namespace {
class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileTransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<MemRefType>(op.getAMemArray().getType());
    auto outputType = dyn_cast<MemRefType>(op.getBMemArray().getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() || inputType.getRank() != 2 ||
        outputType.getRank() != 2)
      return op.emitError("requires static rank-2 input and output memrefs");
    if (outputType.getShape()[0] != inputType.getShape()[1] ||
        outputType.getShape()[1] != inputType.getShape()[0] ||
        inputType.getElementType() != outputType.getElementType())
      return op.emitError("requires a matching transposed output memref");

    rewriter.create<MemTransposeOp>(op.getLoc(), op.getAMemArray(),
                                    op.getBMemArray());
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
    return "Convert Tile operations for the selected Buckyball target.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tile::TileDialect, ::buddy::buckyball::BuckyballDialect,
                func::FuncDialect, memref::MemRefDialect, arith::ArithDialect,
                scf::SCFDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    const auto &targetConfig = buckyball_target::getBuckyballTarget();
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<::buddy::buckyball::BuckyballDialect,
                           memref::MemRefDialect, arith::ArithDialect,
                           scf::SCFDialect, func::FuncDialect,
                           linalg::LinalgDialect>();
    target.addIllegalOp<tile::TileMatMulOp, tile::TileTransposeOp,
                        tile::TileQuantF32ToI8Op, tile::TileMegaKernelOp>();

    RewritePatternSet patterns(context);
    mlir::buddy::populateQuantizedKernelTileLoweringPatterns(patterns);
    patterns.add<TileTransposeLowering>(context);
    for (llvm::StringRef ball : targetConfig.balls) {
#define BUCKYBALL_TILE_HOOK(BALL)                                              \
  if (ball == #BALL)                                                           \
    mlir::buddy::populate##BALL##TileLoweringPatterns(                         \
        patterns, targetConfig.bankWidthBits / 8, targetConfig.bankDepth,      \
        targetConfig.bankNum);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_TILE_HOOK
    }
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerLowerTileToBuckyballPass() {
  PassRegistration<LowerTileToBuckyballPass>();
}
