//===- Lowering.cpp - GemminiBall in lower-buckyball ----------------------===//
//
// matmul_i8_tile: one 16x16 tile C = A @ B over `k`, a positive multiple of
// sixteen. A and B fill their groups in order, each group a bank depth of
// rows before the next, so a K deeper than a bank spans several groups
// (allocated together or gathered with bank_transfer).
//   A: k / 16 blocks of sixteen rows of sixteen i8; row i of block r is
//      A[i, 16r .. 16r + 15].
//   B: k rows of sixteen i8, B[0 .. k - 1, 0 .. 15].
//   C: a bank of four groups, sixteen rows of sixteen i32.
// With read_destination C = in_shift(C + A @ B, shift), rounding to nearest
// even; without, C = A @ B.
//
// This is not smatmul's matmul_i8: C holds one output row per bank row across
// four groups instead of four bank rows of one group, there is no bias row,
// and read_destination accumulates the whole old C with a rounding shift.
//
// The Ball issues one instruction per 16-deep block, the operand groups and
// row bases derived from the block index. Without read_destination the first
// block runs on the output-stationary array (PRELOAD primes D = 0); every
// other block accumulates on the weight-stationary array, reading the old C.
// Only the last block applies the shift.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class MatmulI8TileLowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &b) const override {
    if (op.getKind() != "matmul_i8_tile")
      return failure();
    Location loc = op.getLoc();
    bool readDestination = op.getReadDestination();
    int64_t k = cast<IntegerAttr>(op->getAttr("k")).getInt();
    int64_t shift =
        readDestination ? cast<IntegerAttr>(op->getAttr("shift")).getInt() : 0;
    int64_t meshRows = buckyball_target::getBuckyballBallParam(
        "GEMMINI_COMPUTE_PRELOADED", "meshRows");
    int64_t meshColumns = buckyball_target::getBuckyballBallParam(
        "GEMMINI_COMPUTE_PRELOADED", "meshColumns");
    const auto &target = buckyball_target::getBuckyballTarget();
    ArrayRef<int64_t> sources = op.getSources();
    int64_t c = op.getDestination();
    if (target.bankWidthBits != 128 || meshRows != 16 || meshColumns != 16 ||
        target.bankDepth % 16 || target.bankDepth > 1024 ||
        sources.size() != 2 || sources[0] == sources[1] || k <= 0 || k % 16 ||
        k > 64 * target.bankDepth || llvm::is_contained(sources, c))
      return op.emitOpError("matmul_i8_tile requires distinct A, B and C "
                            "banks, k a multiple of 16 within 64 groups and a "
                            "16x16 mesh");
    int64_t a = sources[0], bBank = sources[1];
    int64_t steps = k / 16, groupBlocks = target.bankDepth / 16;
    uint64_t config = ~0ULL;
    for (int64_t step = 0; step < steps; ++step) {
      uint64_t group = step / groupBlocks, base = step % groupBlocks * 16;
      bool accumulate = readDestination || step > 0;
      // CONFIG: rs2[4] picks the weight-stationary array, rs2[40:9] the shift.
      uint64_t word = mlir::buddy::field(accumulate, 4, 4) |
                      mlir::buddy::field(step == steps - 1 ? shift : 0, 9, 40);
      if (word != config)
        mlir::buddy::emitInstruction(
            b, loc, buckyball_target::getBuckyballFunct7("GEMMINI_CONFIG"),
            mlir::buddy::constantI64(b, loc, 0),
            mlir::buddy::constantI64(b, loc, word));
      config = word;
      // rs2: the subcommand in [3:0], the first and second source row bases
      // in [15:6] and [25:16], their groups in [41:36] and [47:42]; C is at
      // row 0.
      uint64_t bases =
          mlir::buddy::field(base, 6, 15) | mlir::buddy::field(base, 16, 25) |
          mlir::buddy::field(group, 36, 41) | mlir::buddy::field(group, 42, 47);
      if (accumulate) {
        mlir::buddy::emitInstruction(
            b, loc,
            buckyball_target::getBuckyballFunct7("GEMMINI_COMPUTE_ACCUMULATED"),
            mlir::buddy::constantI64(
                b, loc, mlir::buddy::bankFields({a, bBank}, c, true, 16)),
            mlir::buddy::constantI64(b, loc, 3 | bases));
        continue;
      }
      // The output-stationary preload reads sixteen rows of B.
      mlir::buddy::emitInstruction(
          b, loc, buckyball_target::getBuckyballFunct7("GEMMINI_PRELOAD"),
          mlir::buddy::constantI64(
              b, loc, mlir::buddy::bankFields({bBank}, c, false, 16)),
          mlir::buddy::constantI64(b, loc,
                                   1 | mlir::buddy::field(base, 6, 15) |
                                       mlir::buddy::field(group, 36, 41)));
      mlir::buddy::emitInstruction(
          b, loc,
          buckyball_target::getBuckyballFunct7("GEMMINI_COMPUTE_PRELOADED"),
          mlir::buddy::constantI64(
              b, loc, mlir::buddy::bankFields({a, bBank}, c, false, 16)),
          mlir::buddy::constantI64(b, loc, 2 | bases));
    }
    b.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<MatmulI8TileLowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
