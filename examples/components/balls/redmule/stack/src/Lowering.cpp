//===- Lowering.cpp - RedmuleBall in lower-buckyball ----------------------===//
//
// matmul_f16: C = A @ B in FP16 with ordered FMAs, or C = C + A @ B with
// read_destination. `k` is a positive multiple of eight.
//   A (source 0): `rows` rows of `k` f16, k / 8 bank rows each.
//   B (source 1): `k` bank rows of eight f16.
//   C:            `rows` bank rows of eight f16.
//
// The Ball places every operand at bank row 0. With read_destination rs1
// marks the old C, which the Ball reads from the bank the destination held
// before this write, and the FMAs start from it instead of positive zero.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class MatmulF16Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "matmul_f16")
      return failure();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    int64_t k = cast<IntegerAttr>(op->getAttr("k")).getInt();
    int64_t c = op.getDestination();
    bool readDestination = op.getReadDestination();
    const auto &target = buckyball_target::getBuckyballTarget();
    if (op.getSources().size() != 2 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), c))
      return op.emitOpError("FP16 matmul requires distinct A, B and C banks");
    if (target.bankWidthBits != 128 || rows <= 0 || rows > 4095 || k <= 0 ||
        k % 8 || k >= (1LL << 33) || rows * (k / 8) > target.bankDepth ||
        k > target.bankDepth || rows > target.bankDepth)
      return op.emitOpError("FP16 matmul shape exceeds the bank or ISA "
                            "geometry");
    // rs2: rows in [11:0], the A, B and C row bases (all 0) in [41:12], and
    // in bit 42 whether the FMAs start from positive zero.
    uint64_t rs2 = mlir::buddy::field(rows, 0, 11) |
                   mlir::buddy::field(!readDestination, 42, 42);
    mlir::buddy::emitInstruction(
        rewriter, op.getLoc(),
        buckyball_target::getBuckyballFunct7("REDMULE_F16"),
        mlir::buddy::constantI64(
            rewriter, op.getLoc(),
            mlir::buddy::bankFields(op.getSources(), c, readDestination, k)),
        mlir::buddy::constantI64(rewriter, op.getLoc(), rs2));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<MatmulF16Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
