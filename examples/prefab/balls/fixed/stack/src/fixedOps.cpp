#include "fixedOps.h"
#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"
using namespace mlir;
using namespace ::buddy::fixed_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
#include "fixedDialect.cpp.inc"
#define GET_OP_CLASSES
#include "fixedOps.cpp.inc"
void FixedBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "fixedOps.cpp.inc"
      >();
}
namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<FixedBallDialect>();
     }),
     true);
} // namespace
SmallVector<Value, 2> BankFixedOp::getSourceBanks() { return {getA(), getB()}; }
Value BankFixedOp::getDestinationBank() { return getC(); }
bool BankFixedOp::readsDestination() { return false; }
Operation *BankFixedOp::buildPhysical(OpBuilder &b, ArrayRef<int64_t> sources,
                                      int64_t destination) {
  return FixedOp::create(b, getLoc(), sources[0], sources[1], destination,
                         getRows(), getMode(), getFraction(), getBroadcast(),
                         getStream());
}
SmallVector<int64_t, 2> FixedOp::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t FixedOp::getDestinationBank() { return getC(); }
LogicalResult FixedOp::lowerToInstructions(OpBuilder &b) {
  int64_t rows = getRows(), mode = getMode(), fraction = getFraction();
  if (getA() < 0 || getB() < 0 || getC() < 0 ||
      getA() > target::getTarget().privateBankMax ||
      getB() > target::getTarget().privateBankMax ||
      getC() > target::getTarget().privateBankMax || rows <= 0 || rows % 4 ||
      rows > target::getTarget().bankDepth ||
      (mode <= 2 && target::getTarget().bankDepth < 8) || mode < 0 ||
      mode > 7 ||
      (getStream() && (mode > 2 || rows * 2 > target::getTarget().bankDepth)) ||
      (mode == 7 && rows != 4) ||
      (mode >= 6 && target::getTarget().bankDepth < 8) || fraction < 1 ||
      fraction > 24 || (getBroadcast() && mode != 3) || getA() == getB() ||
      getA() == getC() || (getB() == getC() && mode != 6))
    return emitOpError(
        "fixed requires distinct banks (sum may write B), rows divisible by "
        "four within a bank, "
        "mode 0..7 (mean rows=4), fraction 1..24 and multiply-only broadcast");
  uint64_t rs1 =
      bankFields({int64_t(getA()), int64_t(getB())}, getC(), false, rows);
  uint64_t rs2 = mode | (uint64_t(getBroadcast()) << 3) |
                 (uint64_t(fraction) << 4) | (uint64_t(getStream()) << 9);
  emitInstruction(b, getLoc(), target::getFunct7("FIXED"),
                  constantI64(b, getLoc(), rs1), constantI64(b, getLoc(), rs2));
  return success();
}
