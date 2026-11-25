//===- softmaxOps.cpp - SoftmaxBall operations ----------------------------===//
//
// The dialect of SoftmaxBall, its virtual operation building its physical one
// and the instruction of the physical operation.
//
//===----------------------------------------------------------------------===//

#include "softmaxOps.h"

#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"

using namespace mlir;
using namespace ::buddy::softmax_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "softmaxDialect.cpp.inc"

#define GET_OP_CLASSES
#include "softmaxOps.cpp.inc"

void SoftmaxBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "softmaxOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<SoftmaxBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankSoftmaxOp::getSourceBanks() { return {getInput()}; }
Value BankSoftmaxOp::getDestinationBank() { return getOutput(); }
bool BankSoftmaxOp::readsDestination() { return false; }
Operation *BankSoftmaxOp::buildPhysical(OpBuilder &builder,
                                        ArrayRef<int64_t> sources,
                                        int64_t destination) {
  return SoftmaxOp::create(builder, getLoc(), sources[0], destination,
                           getRows(), getW(), getScaleLog2());
}

SmallVector<int64_t, 2> SoftmaxOp::getSourceBanks() {
  return {int64_t(getInput())};
}
int64_t SoftmaxOp::getDestinationBank() { return getOutput(); }
LogicalResult SoftmaxOp::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), w = getW(), input = getInput(),
          output = getOutput(), scale = getScaleLog2();
  const auto &target = target::getTarget();
  if (target.bankWidthBits != 128 || rows <= 0 || w < 16 || w > 2048 ||
      rows * ((w + 3) / 4) > target.bankDepth)
    return emitOpError("softmax exceeds the bank");
  if (input == output)
    return emitOpError("softmax requires distinct input and output banks");
  if (scale < -126 || scale > 127)
    return emitOpError("softmax scale exponent exceeds the instruction");
  // Input and output start at bank row 0: the bank-row fields in_base (16-29)
  // and out_base (30-43) are 0.
  emitInstruction(
      builder, loc, target::getFunct7("SOFTMAX"),
      constantI64(builder, loc, bankFields({input}, output, false, rows)),
      constantI64(builder, loc, field(w, 0, 15) | field(scale & 0xff, 44, 51)));
  return success();
}
