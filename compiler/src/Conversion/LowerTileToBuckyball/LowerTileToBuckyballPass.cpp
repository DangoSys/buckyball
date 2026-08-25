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
#include "Tile/Transform.h"

using namespace mlir;
using namespace ::buddy::buckyball;
namespace tile = ::buddy::tile;

namespace mlir::buddy {
#define BUCKYBALL_TILE_HOOK(BALL)                                           \
  void populate##BALL##TileLoweringPatterns(RewritePatternSet &,            \
                                             int64_t, int64_t, int64_t);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_TILE_HOOK
#define BUCKYBALL_CORE_TILE_HOOK(CORE, NAME)                                \
  void populate##CORE##CoreTileLoweringPatterns(RewritePatternSet &,         \
                                                 int64_t, int64_t, int64_t);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_CORE_TILE_HOOK
} // namespace mlir::buddy

namespace {
class LowerTileToBuckyballPass
    : public PassWrapper<LowerTileToBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToBuckyballPass)

  StringRef getArgument() const final { return "convert-tile-to-buckyball"; }
  StringRef getDescription() const final {
    return "Convert Tile operations for the selected Buckyball target.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tile::TileDialect, ::buddy::buckyball::BuckyballDialect,
                    func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    const buckyball_target::BuckyballTargetConfig &targetConfig =
        buckyball_target::getBuckyballTarget();
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<::buddy::buckyball::BuckyballDialect,
                           memref::MemRefDialect, arith::ArithDialect,
                           scf::SCFDialect, func::FuncDialect,
                           linalg::LinalgDialect>();
    target.addIllegalOp<tile::TileMatMulOp>();
    target.addIllegalOp<tile::TileTransposeOp>();

    RewritePatternSet patterns(context);
    for (llvm::StringRef ball : targetConfig.balls) {
#define BUCKYBALL_TILE_HOOK(BALL)                                           \
  if (ball == #BALL) {                                                       \
    mlir::buddy::populate##BALL##TileLoweringPatterns(                      \
        patterns, targetConfig.bankWidthBits / 8, targetConfig.bankDepth,   \
        targetConfig.bankNum);                                               \
  }
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_TILE_HOOK
    }
#define BUCKYBALL_CORE_TILE_HOOK(CORE, NAME)                                \
    if (targetConfig.core == NAME)                                           \
      mlir::buddy::populate##CORE##CoreTileLoweringPatterns(                 \
          patterns, targetConfig.bankWidthBits / 8, targetConfig.bankDepth, \
          targetConfig.bankNum);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_CORE_TILE_HOOK
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerLowerTileToBuckyballPass() {
  PassRegistration<LowerTileToBuckyballPass>();
}
