//===- Lowering.cpp - SMatMulBall in lower-buckyball ----------------------===//
//
// matmul_i8 with read_destination: C = C + A @ B for `rows` rows (one or a
// multiple of eight) and sixteen columns, over `k` (a multiple of sixteen).
// C starts with one row of sixteen i32 (four bank rows), the bias of every
// row, and receives `rows` rows of sixteen i32 (four bank rows each).
// A is contiguous for rows=1, otherwise packed in tileRows x 16 blocks;
// B holds k rows of sixteen i8.
// matmul_i8_block extends the reduction across instructions: first loads
// the bias from C, and last writes the completed accumulation to C.
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
    bool block = op.getKind() == "matmul_i8_block";
    if (op.getKind() != "matmul_i8" && !block)
      return failure();
    if (!op.getReadDestination() || op.getSources().size() != 2)
      return op.emitOpError("INT8 matmul reads A, B and the old C");
    Location loc = op.getLoc();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    int64_t k = cast<IntegerAttr>(op->getAttr("k")).getInt();
    int64_t c = op.getDestination();
    bool first = !block || cast<BoolAttr>(op->getAttr("first")).getValue();
    bool last = !block || cast<BoolAttr>(op->getAttr("last")).getValue();
    const auto &target = buckyball_target::getBuckyballTarget();
    int64_t tileRows =
        buckyball_target::getBuckyballBallParam("SMATMUL_OS", "tileRows");
    int64_t tileCols =
        buckyball_target::getBuckyballBallParam("SMATMUL_OS", "tileCols");
    if (target.bankWidthBits != 128 || tileCols != 16 || tileRows <= 0 ||
        rows <= 0 || rows > 4095 || (rows != 1 && rows % tileRows) || k <= 0 ||
        k >= (1LL << 33) || k > target.bankDepth || k % 16 ||
        rows * (k / 16) > target.bankDepth || rows * 4 > target.bankDepth)
      return op.emitOpError(
          "INT8 matmul shape exceeds the bank or ISA geometry");
    if (op.getSources()[0] == op.getSources()[1] || op.getSources()[0] == c ||
        op.getSources()[1] == c)
      return op.emitOpError("INT8 matmul requires distinct A, B and C banks");
    // The bias of the accumulation: the four rows at the base of C.
    if (first)
      mlir::buddy::emitInstruction(
          rewriter, loc, buckyball_target::getBuckyballFunct7("SMATMUL_BIAS"),
          mlir::buddy::constantI64(rewriter, loc,
                                   mlir::buddy::bankFields({c}, 0, false, 4)),
          mlir::buddy::constantI64(rewriter, loc, 0));
    uint64_t rs2 =
        mlir::buddy::field(rows, 0, 11) | mlir::buddy::field(16, 12, 23) |
        mlir::buddy::field(first, 24, 24) | mlir::buddy::field(last, 25, 25);
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
