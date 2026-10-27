//===- mataddOps.cpp - MatAddBall operations ------------------------------===//
//
// The dialect of MatAddBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "mataddOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::matadd_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "mataddDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mataddOps.cpp.inc"

void MatAddBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mataddOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<MatAddBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankAddI32Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankAddI32Op::getDestinationBank() { return getC(); }
bool BankAddI32Op::readsDestination() { return false; }
Operation *BankAddI32Op::buildPhysical(OpBuilder &builder,
                                       ArrayRef<int64_t> sources,
                                       int64_t destination) {
  return AddI32Op::create(builder, getLoc(), sources[0], sources[1],
                          destination, getRows());
}

SmallVector<int64_t, 2> AddI32Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t AddI32Op::getDestinationBank() { return getC(); }
LogicalResult AddI32Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), a = getA(), b = getB(), c = getC();
  if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
      a == b || a == c || b == c)
    return emitOpError(
        "add_i32 requires distinct banks and rows within a bank");
  emitInstruction(builder, loc, buckyball_target::getBuckyballFunct7("MATADD"),
                  constantI64(builder, loc, bankFields({a, b}, c, false, rows)),
                  constantI64(builder, loc, 0));
  return success();
}
