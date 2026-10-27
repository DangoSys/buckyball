//===- int2fpOps.cpp - Int2FpBall operations ------------------------------===//
//
// The dialect of Int2FpBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "int2fpOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::int2fp_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "int2fpDialect.cpp.inc"

#define GET_OP_CLASSES
#include "int2fpOps.cpp.inc"

void Int2FpBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "int2fpOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Int2FpBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankInt32ToFp32Op::getSourceBanks() {
  return {getValues(), getScales()};
}
Value BankInt32ToFp32Op::getDestinationBank() { return getOutput(); }
bool BankInt32ToFp32Op::readsDestination() { return false; }
Operation *BankInt32ToFp32Op::buildPhysical(OpBuilder &builder,
                                            ArrayRef<int64_t> sources,
                                            int64_t destination) {
  return Int32ToFp32Op::create(builder, getLoc(), sources[0], sources[1],
                               destination, getRows(), getRelu());
}

SmallVector<int64_t, 2> Int32ToFp32Op::getSourceBanks() {
  return {int64_t(getValues()), int64_t(getScales())};
}
int64_t Int32ToFp32Op::getDestinationBank() { return getOutput(); }
LogicalResult Int32ToFp32Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows();
  emitInstruction(
      builder, loc, buckyball_target::getBuckyballFunct7("INT32_TO_FP32"),
      constantI64(builder, loc,
                  bankFields({int64_t(getValues()), int64_t(getScales())},
                             getOutput(), false, rows * 4)),
      constantI64(builder, loc, field(getRelu(), 0, 0)));
  return success();
}
