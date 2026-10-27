//===- mxquantOps.cpp - MxquantBall operations ----------------------------===//
//
// The dialect of MxquantBall, its virtual operation building its physical one
// and the instruction of the physical operation.
//
//===----------------------------------------------------------------------===//

#include "mxquantOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::mxquant_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;

#include "mxquantDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mxquantOps.cpp.inc"

void MxquantBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxquantOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<MxquantBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankQuantMxfp8Op::getSourceBanks() {
  return {getInput()};
}
Value BankQuantMxfp8Op::getDestinationBank() { return getOutput(); }
bool BankQuantMxfp8Op::readsDestination() { return false; }
Operation *BankQuantMxfp8Op::buildPhysical(OpBuilder &builder,
                                           ArrayRef<int64_t> sources,
                                           int64_t destination) {
  return QuantMxfp8Op::create(builder, getLoc(), sources[0], destination,
                              getCount());
}

SmallVector<int64_t, 2> QuantMxfp8Op::getSourceBanks() {
  return {int64_t(getInput())};
}
int64_t QuantMxfp8Op::getDestinationBank() { return getOutput(); }
LogicalResult QuantMxfp8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t count = getCount(), input = getInput(), output = getOutput();
  const auto &target = buckyball_target::getBuckyballTarget();
  if (target.bankWidthBits != 128 || count <= 0 || count % 32 ||
      count > target.bankDepth * 4)
    return emitOpError("quant_mxfp8 count exceeds the bank");
  if (input == output)
    return emitOpError("quant_mxfp8 requires distinct input and output banks");
  emitInstruction(
      builder, loc, buckyball_target::getBuckyballFunct7("MXQUANT"),
      constantI64(builder, loc, bankFields({input}, output, false, count)),
      constantI64(builder, loc, 0));
  return success();
}
