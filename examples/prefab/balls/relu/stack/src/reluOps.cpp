//===- reluOps.cpp - ReluBall operations ----------------------------------===//
//
// The dialect of ReluBall, its virtual operations building their physical ones
// and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "reluOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::relu_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "reluDialect.cpp.inc"

#define GET_OP_CLASSES
#include "reluOps.cpp.inc"

void ReluBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "reluOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect(
         [](DialectRegistry &registry) { registry.insert<ReluBallDialect>(); }),
     true);
} // namespace

SmallVector<Value, 2> BankReluI32Op::getSourceBanks() { return {getBank()}; }
Value BankReluI32Op::getDestinationBank() { return getBank(); }
bool BankReluI32Op::readsDestination() { return false; }
Operation *BankReluI32Op::buildPhysical(OpBuilder &builder,
                                        ArrayRef<int64_t> sources,
                                        int64_t destination) {
  return ReluI32Op::create(builder, getLoc(), destination, getRows());
}

SmallVector<int64_t, 2> ReluI32Op::getSourceBanks() {
  return {int64_t(getBank())};
}
int64_t ReluI32Op::getDestinationBank() { return getBank(); }
LogicalResult ReluI32Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), bank = getBank();
  int64_t depth = buckyball_target::getBuckyballTarget().bankDepth;
  if (rows <= 0 || rows > depth || depth % 16)
    return emitOpError("relu_i32 rows exceed the bank");
  // Group 0 of the bank in the bank1 field; bit 63 marks the in-place
  // update. rs2 is the segment stride.
  uint64_t rs1 =
      bankFields({bank}, bank, false, (rows + 15) / 16 * 16) | (1ULL << 63);
  emitInstruction(builder, loc, buckyball_target::getBuckyballFunct7("RELU"),
                  constantI64(builder, loc, rs1),
                  constantI64(builder, loc, depth));
  return success();
}
