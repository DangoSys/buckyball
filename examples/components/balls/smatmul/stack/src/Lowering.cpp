//===- Lowering.cpp - SMatMulBall in lower-buckyball ----------------------===//
//
// matmul_i8 (A, B) -> C with read_destination: C = bias + A @ B for `rows`
// rows (one, or a multiple of tileRows up to a quarter of the bank depth) and
// sixteen columns. A and B have the same number of groups; group g holds
// reduction block g of `k` columns (a multiple of sixteen), so K = k * groups.
// In group g, B holds k rows of sixteen i8. A holds k / 16 rows of sixteen i8
// for rows = 1; otherwise M tiles of tileRows rows, each tile the k / 16
// slices of sixteen columns one after another: line
// ((r / tileRows) * (k / 16) + c / 16) * tileRows + r % tileRows.
// C starts with the bias, one row of sixteen i32 in its first four lines, and
// receives `rows` rows of sixteen i32 (four lines each).
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class MatmulI8Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "matmul_i8")
      return failure();
    if (!op.getReadDestination() || op.getSources().size() != 2)
      return op.emitOpError("INT8 matmul reads A, B and the bias in C");
    Location loc = op.getLoc();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    int64_t k = cast<IntegerAttr>(op->getAttr("k")).getInt();
    int64_t c = op.getDestination();
    const auto &target = buckyball_target::getBuckyballTarget();
    int64_t tileRows =
        buckyball_target::getBuckyballBallParam("SMATMUL_OS", "tileRows");
    if (target.bankWidthBits != 128 || rows <= 0 ||
        rows * 4 > target.bankDepth || (rows != 1 && rows % tileRows) ||
        k <= 0 || k > target.bankDepth || k % 16 ||
        rows * (k / 16) > target.bankDepth)
      return op.emitOpError(
          "INT8 matmul shape exceeds the bank or ISA geometry");
    if (op.getSources()[0] == op.getSources()[1] || op.getSources()[0] == c ||
        op.getSources()[1] == c)
      return op.emitOpError("INT8 matmul requires distinct A, B and C banks");
    mlir::buddy::emitInstruction(
        rewriter, loc, buckyball_target::getBuckyballFunct7("SMATMUL_BIAS"),
        mlir::buddy::constantI64(rewriter, loc,
                                 mlir::buddy::bankFields({c}, 0, false, 4)),
        mlir::buddy::constantI64(rewriter, loc, 0));
    // One accumulation over every group: first and last in one instruction.
    uint64_t rs2 =
        mlir::buddy::field(rows, 0, 11) | mlir::buddy::field(16, 12, 23) |
        mlir::buddy::field(1, 24, 24) | mlir::buddy::field(1, 25, 25);
    mlir::buddy::emitInstruction(
        rewriter, loc, buckyball_target::getBuckyballFunct7("SMATMUL_OS"),
        mlir::buddy::constantI64(
            rewriter, loc,
            mlir::buddy::bankFields(op.getSources(), c, false, k)),
        mlir::buddy::constantI64(rewriter, loc, rs2));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<MatmulI8Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
