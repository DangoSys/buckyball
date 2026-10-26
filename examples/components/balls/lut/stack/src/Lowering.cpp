//===- Lowering.cpp - LutBall in lower-buckyball --------------------------===//
//
// lookup_i8: source 0 holds `rows` rows of sixteen i8 codes, source 1 the
// table: one group of 256 entries shared by every channel, or four groups
// with 256 entries per channel (channel * 256 + code, 1024 bytes per group).
// The destination receives table[code] for every element.
//
//===----------------------------------------------------------------------===//
#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"
using namespace mlir;
using namespace ::buddy::buckyball;
namespace {
class LookupI8Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &b) const override {
    if (op.getKind() != "lookup_i8")
      return failure();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
        op.getSources().size() != 2 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), op.getDestination()))
      return op.emitOpError(
          "lookup_i8 requires distinct banks and rows within bank depth");
    mlir::buddy::emitInstruction(
        b, op.getLoc(), buckyball_target::getBuckyballFunct7("LUT"),
        mlir::buddy::constantI64(b, op.getLoc(),
                                 mlir::buddy::bankFields(op.getSources(),
                                                         op.getDestination(),
                                                         false, rows)),
        mlir::buddy::constantI64(b, op.getLoc(), 0));
    b.eraseOp(op);
    return success();
  }
};
void populate(RewritePatternSet &patterns) {
  patterns.add<LookupI8Lowering>(patterns.getContext());
}
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
