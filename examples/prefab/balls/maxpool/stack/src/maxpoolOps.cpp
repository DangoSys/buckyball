//===- maxpoolOps.cpp - MaxPoolBall operations ----------------------------===//
//
// The dialect of MaxPoolBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "maxpoolOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::maxpool_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "maxpoolDialect.cpp.inc"

#define GET_OP_CLASSES
#include "maxpoolOps.cpp.inc"

void MaxPoolBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "maxpoolOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<MaxPoolBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankMaxpoolI8Op::getSourceBanks() { return {getInput()}; }
Value BankMaxpoolI8Op::getDestinationBank() { return getOutput(); }
bool BankMaxpoolI8Op::readsDestination() { return false; }
Operation *BankMaxpoolI8Op::buildPhysical(OpBuilder &builder,
                                          ArrayRef<int64_t> sources,
                                          int64_t destination) {
  return MaxpoolI8Op::create(builder, getLoc(), sources[0], destination,
                             getInputSide(), getKernel(), getStride(),
                             getPadding());
}

SmallVector<int64_t, 2> MaxpoolI8Op::getSourceBanks() {
  return {int64_t(getInput())};
}
int64_t MaxpoolI8Op::getDestinationBank() { return getOutput(); }
LogicalResult MaxpoolI8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t input = getInput(), output = getOutput();
  if (input == output)
    return emitOpError("maxpool_i8 requires distinct input and output banks");
  int64_t inputSide = getInputSide(), kernel = getKernel(),
          stride = getStride(), padding = getPadding();
  int64_t span = inputSide + 2 * padding - kernel;
  if (inputSide <= 0 || inputSide > 15 || kernel <= 0 || kernel > 15 ||
      stride <= 0 || stride > 15 || padding < 0 || padding > 15 || span < 0 ||
      span % stride || span / stride + 1 > 15 ||
      inputSide * inputSide > buckyball_target::getBuckyballTarget().bankDepth)
    return emitOpError("maxpool_i8 geometry exceeds the ISA field ranges");
  int64_t outputSide = span / stride + 1;
  // Both maps start at the base of their groups; output rows are packed.
  uint64_t rs2 = field(inputSide, 0, 3) | field(outputSide, 4, 7) |
                 field(kernel, 8, 11) | field(stride, 12, 15) |
                 field(padding, 16, 19) | field(outputSide, 32, 37);
  emitInstruction(
      builder, loc, buckyball_target::getBuckyballFunct7("MAXPOOL"),
      constantI64(builder, loc,
                  bankFields({input}, output, false, outputSide * outputSide) |
                      (1ULL << 63)),
      constantI64(builder, loc, rs2));
  return success();
}
