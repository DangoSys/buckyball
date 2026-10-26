//===- Lowering.cpp - Int8MulBall in lower-buckyball
//-----------------------===//
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
    int64_t gateRow = cast<IntegerAttr>(op->getAttr("gate_row")).getInt();
    int64_t depth = buckyball_target::getBuckyballTarget().bankDepth;
    if (rows <= 0 || rows > depth || gateRow < 0 || gateRow >= depth ||
        op.getSources().size() != 2 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), op.getDestination()))
      return op.emitOpError(
          "gate_mul_i8 requires distinct banks and valid row addresses");
    Value bits = arith::BitcastOp::create(b, op.getLoc(), b.getI32Type(),
                                          op.getArguments()[0]);
    Value ratio = arith::ExtUIOp::create(b, op.getLoc(), b.getI64Type(), bits);
    Value rs2 = arith::OrIOp::create(
        b, op.getLoc(), ratio,
        mlir::buddy::constantI64(b, op.getLoc(), uint64_t(gateRow) << 32));
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
