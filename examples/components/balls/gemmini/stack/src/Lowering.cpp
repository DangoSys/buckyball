//===- Lowering.cpp - GemminiBall in lower-buckyball ----------------------===//
//
// matmul_i8_tile: C = A @ B for one 16x16 tile. A holds sixteen rows of
// sixteen i8 (M x K) at row a_base, B sixteen rows of sixteen i8 (K x N) at
// row b_base, and C, a bank of four columns, receives sixteen rows of sixteen
// i32 at row c_base. a_base, b_base and c_base are the i64 arguments.
// With read_destination C = in_shift(C + A @ B, shift), rounding to nearest
// even; without, C = A @ B on the output-stationary array.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

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
    bool accumulate = op.getReadDestination();
    int64_t shift =
        accumulate ? cast<IntegerAttr>(op->getAttr("shift")).getInt() : 0;
    int64_t meshRows = buckyball_target::getBuckyballBallParam(
        "GEMMINI_COMPUTE_PRELOADED", "meshRows");
    int64_t meshColumns = buckyball_target::getBuckyballBallParam(
        "GEMMINI_COMPUTE_PRELOADED", "meshColumns");
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || meshRows != 16 || meshColumns != 16 ||
        target.bankDepth < 16 || op.getSources().size() != 2 ||
        op.getArguments().size() != 3 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), op.getDestination()))
      return op.emitOpError("matmul_i8_tile requires distinct A, B and C "
                            "banks, three row bases and a 16x16 mesh");
    int64_t a = op.getSources()[0], bBank = op.getSources()[1];
    int64_t c = op.getDestination();
    // rs2: the subcommand in [3:0], then the A, B and C row bases.
    auto rs2 = [&](uint64_t subcommand, Value aBase, Value bBase) {
      Value word = mlir::buddy::constantI64(b, loc, subcommand);
      for (auto [base, bit] : {std::pair{aBase, 6}, std::pair{bBase, 16},
                               std::pair{op.getArguments()[2], 26}})
        if (base)
          word = arith::OrIOp::create(
              b, loc, word,
              arith::ShLIOp::create(b, loc, base,
                                    mlir::buddy::constantI64(b, loc, bit)));
      return word;
    };
    Value aBase = op.getArguments()[0], bBase = op.getArguments()[1];
    // CONFIG: weight-stationary for the accumulation, as the ISA tests use it,
    // output-stationary for the plain product.
    mlir::buddy::emitInstruction(
        b, loc, buckyball_target::getBuckyballFunct7("GEMMINI_CONFIG"),
        mlir::buddy::constantI64(b, loc, 0),
        mlir::buddy::constantI64(b, loc,
                                 mlir::buddy::field(accumulate, 4, 4) |
                                     mlir::buddy::field(shift, 9, 40)));
    if (accumulate) {
      mlir::buddy::emitInstruction(
          b, loc,
          buckyball_target::getBuckyballFunct7("GEMMINI_COMPUTE_ACCUMULATED"),
          mlir::buddy::constantI64(
              b, loc, mlir::buddy::bankFields({a, bBank}, c, true, 16)),
          rs2(3, aBase, bBase));
      b.eraseOp(op);
      return success();
    }
    // The output-stationary preload only primes the array with D = 0; it
    // reads B, which is valid for sixteen rows.
    mlir::buddy::emitInstruction(
        b, loc, buckyball_target::getBuckyballFunct7("GEMMINI_PRELOAD"),
        mlir::buddy::constantI64(
            b, loc, mlir::buddy::bankFields({bBank}, c, false, 16)),
        rs2(1, bBase, Value()));
    mlir::buddy::emitInstruction(
        b, loc,
        buckyball_target::getBuckyballFunct7("GEMMINI_COMPUTE_PRELOADED"),
        mlir::buddy::constantI64(
            b, loc, mlir::buddy::bankFields({a, bBank}, c, false, 16)),
        rs2(2, aBase, bBase));
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
