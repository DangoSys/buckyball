//===- Lowering.cpp - Int8MulBall in lower-buckyball ----------------------===//
//
// gate_mul_i8: source 0 holds the gate, one row of sixteen i8; source 1 holds
// `rows` rows of sixteen i8. The destination receives
// clamp(round_half_even(gate * x * ratio), -128, 127) per column, in every
// group of the banks; the three banks have the same number of groups.
// Argument 0 is the f32 ratio, finite and positive.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class GateMulI8Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &b) const override {
    if (op.getKind() != "gate_mul_i8")
      return failure();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
        op.getSources().size() != 2 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), op.getDestination()))
      return op.emitOpError(
          "gate_mul_i8 requires distinct banks and rows within a bank");
    // rs2: ratio bits in [31:0]; the gate row in [63:32] is the first row.
    Value bits = arith::BitcastOp::create(b, op.getLoc(), b.getI32Type(),
                                          op.getArguments()[0]);
    Value rs2 = arith::ExtUIOp::create(b, op.getLoc(), b.getI64Type(), bits);
    mlir::buddy::emitInstruction(
        b, op.getLoc(), buckyball_target::getBuckyballFunct7("INT8MUL"),
        mlir::buddy::constantI64(b, op.getLoc(),
                                 mlir::buddy::bankFields(op.getSources(),
                                                         op.getDestination(),
                                                         false, rows)),
        rs2);
    b.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<GateMulI8Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
