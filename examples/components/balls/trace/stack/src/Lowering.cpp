//===- Lowering.cpp - TraceBall in lower-buckyball ------------------------===//
//
// The BDB counter touches no bank: its operations have no sources and
// destination 0, and rs1 is zero. `counter` (0 to 15) selects the counter.
// counter_start starts it with the i64 tag of argument 0; counter_read traces
// its current value; counter_stop traces the elapsed cycles and stops it.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class CounterLowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    uint64_t subcommand;
    if (op.getKind() == "counter_start")
      subcommand = 0;
    else if (op.getKind() == "counter_stop")
      subcommand = 1;
    else if (op.getKind() == "counter_read")
      subcommand = 2;
    else
      return failure();
    Location loc = op.getLoc();
    int64_t counter = cast<IntegerAttr>(op->getAttr("counter")).getInt();
    if (!op.getSources().empty() || op.getDestination() != 0 ||
        op.getReadDestination())
      return op.emitOpError("BDB counter touches no bank");
    if (counter < 0 || counter > 15 ||
        op.getArguments().size() != (subcommand == 0 ? 1 : 0))
      return op.emitOpError("BDB counter needs a counter in [0, 15] and a "
                            "tag only to start");
    Value rs2 = mlir::buddy::constantI64(rewriter, loc,
                                         mlir::buddy::field(subcommand, 0, 3) |
                                             mlir::buddy::field(counter, 4, 7));
    if (subcommand == 0) {
      Value tag =
          arith::ShLIOp::create(rewriter, loc, op.getArguments()[0],
                                mlir::buddy::constantI64(rewriter, loc, 8));
      rs2 = arith::OrIOp::create(rewriter, loc, rs2, tag);
    }
    mlir::buddy::emitInstruction(
        rewriter, loc, buckyball_target::getBuckyballFunct7("BDB_COUNTER"),
        mlir::buddy::constantI64(rewriter, loc, 0), rs2);
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<CounterLowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
