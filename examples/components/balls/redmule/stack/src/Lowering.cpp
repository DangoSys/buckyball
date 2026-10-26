//===- Lowering.cpp - RedmuleBall in lower-buckyball ----------------------===//
//
// matmul_f16: C = A @ B in FP16 with ordered FMAs, or C = C + A @ B with
// read_destination. A (source 0) holds `rows` rows of `k` f16 from bank row
// op1_base, k / 8 bank rows each; B (source 1) holds `k` rows of eight f16
// from bank row op2_base; C receives `rows` rows of eight f16 at bank row
// wr_base. `k` is a positive multiple of eight.
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
    int64_t op1Base = cast<IntegerAttr>(op->getAttr("op1_base")).getInt();
    int64_t op2Base = cast<IntegerAttr>(op->getAttr("op2_base")).getInt();
    int64_t wrBase = cast<IntegerAttr>(op->getAttr("wr_base")).getInt();
    int64_t c = op.getDestination();
    const auto &target = buckyball_target::getBuckyballTarget();
    if (op.getSources().size() != 2 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), c))
      return op.emitOpError("FP16 matmul requires distinct A, B and C banks");
    if (target.bankWidthBits != 128 || rows <= 0 || rows > 4095 || k <= 0 ||
        k % 8 || k >= (1LL << 33) || op1Base < 0 || op1Base > 1023 ||
        op2Base < 0 || op2Base > 1023 || wrBase < 0 || wrBase > 1023 ||
        op1Base + rows * (k / 8) > target.bankDepth ||
        op2Base + k > target.bankDepth || wrBase + rows > target.bankDepth)
      return op.emitOpError("FP16 matmul shape exceeds the bank or ISA "
                            "geometry");
    // The Ball reads the old C itself when not first; rs1 does not mark it.
    uint64_t rs2 = mlir::buddy::field(rows, 0, 11) |
                   mlir::buddy::field(op1Base, 12, 21) |
                   mlir::buddy::field(op2Base, 22, 31) |
                   mlir::buddy::field(wrBase, 32, 41) |
                   mlir::buddy::field(!op.getReadDestination(), 42, 42);
    mlir::buddy::emitInstruction(
        rewriter, op.getLoc(),
        buckyball_target::getBuckyballFunct7("REDMULE_F16"),
        mlir::buddy::constantI64(
            rewriter, op.getLoc(),
            mlir::buddy::bankFields(op.getSources(), c, false, k)),
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
