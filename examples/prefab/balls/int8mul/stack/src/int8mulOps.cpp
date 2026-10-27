//===- int8mulOps.cpp - Int8MulBall operations ----------------------------===//
//
// The dialect of Int8MulBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "int8mulOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace ::buddy::int8mul_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "int8mulDialect.cpp.inc"

#define GET_OP_CLASSES
#include "int8mulOps.cpp.inc"

void Int8MulBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "int8mulOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Int8MulBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankGateMulI8Op::getSourceBanks() {
  return {getGate(), getInput()};
}
Value BankGateMulI8Op::getDestinationBank() { return getOutput(); }
bool BankGateMulI8Op::readsDestination() { return false; }
Operation *BankGateMulI8Op::buildPhysical(OpBuilder &builder,
                                          ArrayRef<int64_t> sources,
                                          int64_t destination) {
  return GateMulI8Op::create(builder, getLoc(), getRatio(), sources[0],
                             sources[1], destination, getRows());
}

SmallVector<int64_t, 2> GateMulI8Op::getSourceBanks() {
  return {int64_t(getGate()), int64_t(getInput())};
}
int64_t GateMulI8Op::getDestinationBank() { return getOutput(); }
LogicalResult GateMulI8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), gate = getGate(), input = getInput(),
          output = getOutput();
  if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
      gate == input || gate == output || input == output)
    return emitOpError(
        "gate_mul_i8 requires distinct banks and rows within a bank");
  // rs2: ratio bits in [31:0]; the gate row in [63:32] is the first row.
  Value bits =
      arith::BitcastOp::create(builder, loc, builder.getI32Type(), getRatio());
  Value rs2 = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), bits);
  emitInstruction(
      builder, loc, buckyball_target::getBuckyballFunct7("INT8MUL"),
      constantI64(builder, loc, bankFields({gate, input}, output, false, rows)),
      rs2);
  return success();
}
