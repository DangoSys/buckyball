//===- toint8Ops.cpp - ToInt8Ball operations ------------------------------===//
//
// The dialect of ToInt8Ball, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "toint8Ops.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace ::buddy::toint8_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "toint8Dialect.cpp.inc"

#define GET_OP_CLASSES
#include "toint8Ops.cpp.inc"

void ToInt8BallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toint8Ops.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<ToInt8BallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankFp32ToInt8Op::getSourceBanks() {
  return {getInput()};
}
Value BankFp32ToInt8Op::getDestinationBank() { return getOutput(); }
bool BankFp32ToInt8Op::readsDestination() { return false; }
Operation *BankFp32ToInt8Op::buildPhysical(OpBuilder &builder,
                                           ArrayRef<int64_t> sources,
                                           int64_t destination) {
  return Fp32ToInt8Op::create(builder, getLoc(), getMultiplier(), sources[0],
                              destination, getRows());
}

SmallVector<int64_t, 2> Fp32ToInt8Op::getSourceBanks() {
  return {int64_t(getInput())};
}
int64_t Fp32ToInt8Op::getDestinationBank() { return getOutput(); }
LogicalResult Fp32ToInt8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  uint64_t rs1 =
      bankFields({int64_t(getInput())}, getOutput(), false, getRows());
  Value bits = arith::BitcastOp::create(builder, loc, builder.getI32Type(),
                                        getMultiplier());
  Value rs2 = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), bits);
  emitInstruction(builder, loc,
                  buckyball_target::getBuckyballFunct7("QUANT_F32_TO_I8"),
                  constantI64(builder, loc, rs1), rs2);
  return success();
}

SmallVector<Value, 2> BankInt32ToInt8Op::getSourceBanks() {
  return {getValues(), getScales()};
}
Value BankInt32ToInt8Op::getDestinationBank() { return getOutput(); }
bool BankInt32ToInt8Op::readsDestination() { return false; }
Operation *BankInt32ToInt8Op::buildPhysical(OpBuilder &builder,
                                            ArrayRef<int64_t> sources,
                                            int64_t destination) {
  return Int32ToInt8Op::create(builder, getLoc(), sources[0], sources[1],
                               destination, getRows(), getRelu());
}

SmallVector<int64_t, 2> Int32ToInt8Op::getSourceBanks() {
  return {int64_t(getValues()), int64_t(getScales())};
}
int64_t Int32ToInt8Op::getDestinationBank() { return getOutput(); }
LogicalResult Int32ToInt8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows();
  int64_t depth = buckyball_target::getBuckyballTarget().bankDepth;
  if (rows <= 0 || rows * 4 > depth)
    return emitOpError("INT32-to-INT8 rows exceed the bank depth");
  // The output rows start at the base of the destination, packed; the
  // seven-bit height takes 128 rows as two columns of 64.
  int64_t width = rows > 127 ? 2 : 1;
  uint64_t rs1 = bankFields({int64_t(getValues()), int64_t(getScales())},
                            getOutput(), false, rows * 4);
  uint64_t rs2 = uint64_t(getRelu()) | (uint64_t(width) << 8) |
                 (uint64_t(rows / width) << 15) | (uint64_t(width) << 22);
  emitInstruction(
      builder, loc, buckyball_target::getBuckyballFunct7("QUANT_I32_TO_I8"),
      constantI64(builder, loc, rs1), constantI64(builder, loc, rs2));
  return success();
}
