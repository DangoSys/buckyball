//===- redmuleOps.cpp - RedmuleBall operations ----------------------------===//
//
// The dialect of RedmuleBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "redmuleOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::redmule_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "redmuleDialect.cpp.inc"

#define GET_OP_CLASSES
#include "redmuleOps.cpp.inc"

void RedmuleBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "redmuleOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<RedmuleBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankMatmulF16Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankMatmulF16Op::getDestinationBank() { return getC(); }
bool BankMatmulF16Op::readsDestination() { return getReadDestination(); }
Operation *BankMatmulF16Op::buildPhysical(OpBuilder &builder,
                                          ArrayRef<int64_t> sources,
                                          int64_t destination) {
  return MatmulF16Op::create(builder, getLoc(), sources[0], sources[1],
                             destination, getRows(), getK(),
                             getReadDestination());
}

SmallVector<int64_t, 2> MatmulF16Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t MatmulF16Op::getDestinationBank() { return getC(); }
LogicalResult MatmulF16Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), k = getK(), a = getA(), b = getB(), c = getC();
  bool readDestination = getReadDestination();
  const auto &target = buckyball_target::getBuckyballTarget();
  if (a == b || a == c || b == c)
    return emitOpError("FP16 matmul requires distinct A, B and C banks");
  if (target.bankWidthBits != 128 || rows <= 0 || rows > 4095 || k <= 0 ||
      k % 8 || k >= (1LL << 33) || rows * (k / 8) > target.bankDepth ||
      k > target.bankDepth || rows > target.bankDepth)
    return emitOpError("FP16 matmul shape exceeds the bank or ISA geometry");
  // rs2: rows in [11:0], the A, B and C row bases (all 0) in [41:12], and in
  // bit 42 whether the FMAs start from positive zero.
  uint64_t rs2 = field(rows, 0, 11) | field(!readDestination, 42, 42);
  emitInstruction(
      builder, loc, buckyball_target::getBuckyballFunct7("REDMULE_F16"),
      constantI64(builder, loc, bankFields({a, b}, c, readDestination, k)),
      constantI64(builder, loc, rs2));
  return success();
}
