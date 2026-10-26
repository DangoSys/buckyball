//===- Lowering.cpp - Int8AddBall in lower-buckyball ----------------------===//
//
// add_i8: the destination receives `rows` rows of sixteen i8,
// round_half_even(A * lhs_ratio + B * rhs_ratio) saturated to i8, and with
// `relu` clamped at zero. Arguments 0 and 1 are the f32 lhs_ratio and
// rhs_ratio, both finite and positive.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class AddI8Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "add_i8")
      return failure();
    Location loc = op.getLoc();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    bool relu = cast<BoolAttr>(op->getAttr("relu")).getValue();
    if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
        op.getSources().size() != 2 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), op.getDestination()))
      return op.emitOpError(
          "add_i8 requires distinct banks and rows within a bank");
    // rs2: lhs_ratio bits in [31:0], rhs_ratio bits in [63:32].
    auto ratioBits = [&](Value ratio) -> Value {
      Value bits =
          arith::BitcastOp::create(rewriter, loc, rewriter.getI32Type(), ratio);
      return arith::ExtUIOp::create(rewriter, loc, rewriter.getI64Type(), bits);
    };
    Value rhs =
        arith::ShLIOp::create(rewriter, loc, ratioBits(op.getArguments()[1]),
                              mlir::buddy::constantI64(rewriter, loc, 32));
    Value rs2 = arith::OrIOp::create(rewriter, loc,
                                     ratioBits(op.getArguments()[0]), rhs);
    mlir::buddy::emitInstruction(
        rewriter, loc,
        buckyball_target::getBuckyballFunct7(relu ? "INT8ADD_RELU" : "INT8ADD"),
        mlir::buddy::constantI64(rewriter, loc,
                                 mlir::buddy::bankFields(op.getSources(),
                                                         op.getDestination(),
                                                         false, rows)),
        rs2);
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<AddI8Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
