//===- int8addOps.cpp - Int8AddBall operations ----------------------------===//
//
// The dialect of Int8AddBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "int8addOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace ::buddy::int8add_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "int8addDialect.cpp.inc"

#define GET_OP_CLASSES
#include "int8addOps.cpp.inc"

void Int8AddBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "int8addOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Int8AddBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankAddI8Op::getSourceBanks() { return {getA(), getB()}; }
Value BankAddI8Op::getDestinationBank() { return getC(); }
bool BankAddI8Op::readsDestination() { return false; }
Operation *BankAddI8Op::buildPhysical(OpBuilder &builder,
                                      ArrayRef<int64_t> sources,
                                      int64_t destination) {
  return AddI8Op::create(builder, getLoc(), getLhsRatio(), getRhsRatio(),
                         sources[0], sources[1], destination, getRows(),
                         getRelu());
}

SmallVector<int64_t, 2> AddI8Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t AddI8Op::getDestinationBank() { return getC(); }
LogicalResult AddI8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), a = getA(), b = getB(), c = getC();
  if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
      a == b || a == c || b == c)
    return emitOpError("add_i8 requires distinct banks and rows within a bank");
  // rs2: lhs_ratio bits in [31:0], rhs_ratio bits in [63:32].
  auto ratioBits = [&](Value ratio) -> Value {
    Value bits =
        arith::BitcastOp::create(builder, loc, builder.getI32Type(), ratio);
    return arith::ExtUIOp::create(builder, loc, builder.getI64Type(), bits);
  };
  Value rhs = arith::ShLIOp::create(builder, loc, ratioBits(getRhsRatio()),
                                    constantI64(builder, loc, 32));
  Value rs2 = arith::OrIOp::create(builder, loc, ratioBits(getLhsRatio()), rhs);
  emitInstruction(builder, loc,
                  buckyball_target::getBuckyballFunct7(
                      getRelu() ? "INT8ADD_RELU" : "INT8ADD"),
                  constantI64(builder, loc, bankFields({a, b}, c, false, rows)),
                  rs2);
  return success();
}
