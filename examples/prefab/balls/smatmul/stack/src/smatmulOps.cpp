//===- smatmulOps.cpp - SMatMulBall operations ----------------------------===//
//
// The dialect of SMatMulBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "smatmulOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::smatmul_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "smatmulDialect.cpp.inc"

#define GET_OP_CLASSES
#include "smatmulOps.cpp.inc"

void SMatMulBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "smatmulOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<SMatMulBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankMatmulI8Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankMatmulI8Op::getDestinationBank() { return getC(); }
bool BankMatmulI8Op::readsDestination() { return true; }
Operation *BankMatmulI8Op::buildPhysical(OpBuilder &builder,
                                         ArrayRef<int64_t> sources,
                                         int64_t destination) {
  return MatmulI8Op::create(builder, getLoc(), sources[0], sources[1],
                            destination, getRows(), getK());
}

SmallVector<int64_t, 2> MatmulI8Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t MatmulI8Op::getDestinationBank() { return getC(); }
LogicalResult MatmulI8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), k = getK(), a = getA(), b = getB(), c = getC();
  const auto &target = buckyball_target::getBuckyballTarget();
  int64_t tileRows =
      buckyball_target::getBuckyballBallParam("SMATMUL_OS", "tileRows");
  if (target.bankWidthBits != 128 || rows <= 0 || rows * 4 > target.bankDepth ||
      (rows != 1 && rows % tileRows) || k <= 0 || k > target.bankDepth ||
      k % 16 || rows * (k / 16) > target.bankDepth)
    return emitOpError("INT8 matmul shape exceeds the bank or ISA geometry");
  if (a == b || a == c || b == c)
    return emitOpError("INT8 matmul requires distinct A, B and C banks");
  emitInstruction(builder, loc,
                  buckyball_target::getBuckyballFunct7("SMATMUL_BIAS"),
                  constantI64(builder, loc, bankFields({c}, 0, false, 4)),
                  constantI64(builder, loc, 0));
  // One accumulation over every group: first and last in one instruction.
  uint64_t rs2 = field(rows, 0, 11) | field(16, 12, 23) | field(1, 24, 24) |
                 field(1, 25, 25);
  emitInstruction(builder, loc,
                  buckyball_target::getBuckyballFunct7("SMATMUL_OS"),
                  constantI64(builder, loc, bankFields({a, b}, c, false, k)),
                  constantI64(builder, loc, rs2));
  return success();
}
