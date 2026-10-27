//===- traceOps.cpp - TraceBall operations --------------------------------===//
//
// The dialect of TraceBall, its virtual operations building their physical ones
// and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "traceOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace ::buddy::trace_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "traceDialect.cpp.inc"

#define GET_OP_CLASSES
#include "traceOps.cpp.inc"

void TraceBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "traceOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<TraceBallDialect>();
     }),
     true);
} // namespace

// One BDB_COUNTER instruction: the subcommand in rs2[3:0], the counter in
// rs2[7:4] and the tag of counter_start from rs2[8].
static LogicalResult counter(Operation *op, OpBuilder &builder,
                             uint64_t subcommand, int64_t counter, Value tag) {
  Location loc = op->getLoc();
  if (counter < 0 || counter > 15)
    return op->emitOpError("BDB counter needs a counter in [0, 15]");
  Value rs2 =
      constantI64(builder, loc, field(subcommand, 0, 3) | field(counter, 4, 7));
  if (tag)
    rs2 = arith::OrIOp::create(
        builder, loc, rs2,
        arith::ShLIOp::create(builder, loc, tag, constantI64(builder, loc, 8)));
  emitInstruction(builder, loc,
                  buckyball_target::getBuckyballFunct7("BDB_COUNTER"),
                  constantI64(builder, loc, 0), rs2);
  return success();
}

SmallVector<int64_t, 2> CounterStartOp::getSourceBanks() { return {}; }
int64_t CounterStartOp::getDestinationBank() { return 0; }
LogicalResult CounterStartOp::lowerToInstructions(OpBuilder &builder) {
  return counter(*this, builder, 0, getCounter(), getTag());
}

SmallVector<int64_t, 2> CounterStopOp::getSourceBanks() { return {}; }
int64_t CounterStopOp::getDestinationBank() { return 0; }
LogicalResult CounterStopOp::lowerToInstructions(OpBuilder &builder) {
  return counter(*this, builder, 1, getCounter(), Value());
}

SmallVector<int64_t, 2> CounterReadOp::getSourceBanks() { return {}; }
int64_t CounterReadOp::getDestinationBank() { return 0; }
LogicalResult CounterReadOp::lowerToInstructions(OpBuilder &builder) {
  return counter(*this, builder, 2, getCounter(), Value());
}
