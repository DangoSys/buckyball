//===- Lowering.cpp - ToInt8Ball in lower-buckyball -----------------------===//
//
// fp32_to_int8: the source bank holds `rows` rows of four f32, a multiple of
// four; the destination receives rows / 4 rows of sixteen i8,
// clamp(round_half_even(x * multiplier), -128, 127). Argument 0 is the f32
// multiplier.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class Fp32ToInt8Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "fp32_to_int8")
      return failure();
    Location loc = op.getLoc();
    auto rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    uint64_t rs1 = mlir::buddy::bankFields(op.getSources(), op.getDestination(),
                                           false, rows);
    Value bits = arith::BitcastOp::create(rewriter, loc, rewriter.getI32Type(),
                                          op.getArguments()[0]);
    Value rs2 =
        arith::ExtUIOp::create(rewriter, loc, rewriter.getI64Type(), bits);
    mlir::buddy::emitInstruction(
        rewriter, loc, buckyball_target::getBuckyballFunct7("QUANT_F32_TO_I8"),
        mlir::buddy::constantI64(rewriter, loc, rs1), rs2);
    rewriter.eraseOp(op);
    return success();
  }
};

class Int32ToInt8Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "int32_to_int8")
      return failure();
    auto rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    auto base = cast<IntegerAttr>(op->getAttr("wr_base")).getInt();
    bool relu = cast<BoolAttr>(op->getAttr("relu")).getValue();
    int64_t depth = buckyball_target::getBuckyballTarget().bankDepth;
    if (rows <= 0 || rows > 127 || rows * 4 > depth || base < 0 || base > 127 ||
        base + rows > depth)
      return op.emitOpError(
          "INT32-to-INT8 rows or wr_base exceed the ISA range");
    // One row contains sixteen output bytes. A partial update retains the
    // destination version; reading an old destination alone does not do that.
    uint64_t rs1 = mlir::buddy::bankFields(op.getSources(), op.getDestination(),
                                           false, rows * 4) |
                   (uint64_t(op.getReadDestination()) << 63);
    uint64_t rs2 = uint64_t(relu) | (uint64_t(base) << 1) | (1ULL << 8) |
                   (uint64_t(rows) << 15) | (1ULL << 22);
    mlir::buddy::emitInstruction(
        rewriter, op.getLoc(),
        buckyball_target::getBuckyballFunct7("QUANT_I32_TO_I8"),
        mlir::buddy::constantI64(rewriter, op.getLoc(), rs1),
        mlir::buddy::constantI64(rewriter, op.getLoc(), rs2));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<Fp32ToInt8Lowering, Int32ToInt8Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
