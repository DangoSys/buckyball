//===- f32addOps.cpp - F32AddBall operations ------------------------------===//
//
// The dialect of F32AddBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "f32addOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::f32add_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "f32addDialect.cpp.inc"

#define GET_OP_CLASSES
#include "f32addOps.cpp.inc"

void F32AddBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "f32addOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<F32AddBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankAddF32Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankAddF32Op::getDestinationBank() { return getC(); }
bool BankAddF32Op::readsDestination() { return false; }
Operation *BankAddF32Op::buildPhysical(OpBuilder &builder,
                                       ArrayRef<int64_t> sources,
                                       int64_t destination) {
  return AddF32Op::create(builder, getLoc(), sources[0], sources[1],
                          destination, getRows());
}

SmallVector<int64_t, 2> AddF32Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t AddF32Op::getDestinationBank() { return getC(); }
LogicalResult AddF32Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), a = getA(), b = getB(), c = getC();
  if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
      a == b || a == c || b == c)
    return emitOpError(
        "add_f32 requires distinct A, B and C banks and rows within a bank");
  // rs2: group 0 in [4:0]; bit 5 clear reads B.
  emitInstruction(builder, loc, buckyball_target::getBuckyballFunct7("F32ADD"),
                  constantI64(builder, loc, bankFields({a, b}, c, false, rows)),
                  constantI64(builder, loc, 0));
  return success();
}
