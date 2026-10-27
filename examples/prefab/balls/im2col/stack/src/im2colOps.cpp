//===- im2colOps.cpp - Im2colBall operations ------------------------------===//
//
// The dialect of Im2colBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "im2colOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::im2col_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "im2colDialect.cpp.inc"

#define GET_OP_CLASSES
#include "im2colOps.cpp.inc"

void Im2colBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "im2colOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Im2colBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankIm2colI8Op::getSourceBanks() { return {getInput()}; }
Value BankIm2colI8Op::getDestinationBank() { return getOutput(); }
bool BankIm2colI8Op::readsDestination() { return false; }
Operation *BankIm2colI8Op::buildPhysical(OpBuilder &builder,
                                         ArrayRef<int64_t> sources,
                                         int64_t destination) {
  return Im2colI8Op::create(builder, getLoc(), sources[0], destination,
                            getInputSide(), getKernel(), getStride(),
                            getPadding(), getTileRows());
}

SmallVector<int64_t, 2> Im2colI8Op::getSourceBanks() {
  return {int64_t(getInput())};
}
int64_t Im2colI8Op::getDestinationBank() { return getOutput(); }
LogicalResult Im2colI8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t input = getInput(), output = getOutput();
  if (input == output)
    return emitOpError("im2col_i8 requires distinct input and output banks");
  int64_t side = getInputSide(), kernel = getKernel(), stride = getStride(),
          padding = getPadding(), tileRows = getTileRows();
  const auto &target = buckyball_target::getBuckyballTarget();
  int64_t span = side + 2 * padding - kernel;
  if (target.bankWidthBits != 128 || side <= 0 ||
      side > buckyball_target::getBuckyballBallParam("IM2COL", "maxIter") ||
      side * side > target.bankDepth || kernel <= 0 ||
      kernel > buckyball_target::getBuckyballBallParam("IM2COL", "maxKSize") ||
      stride <= 0 || stride > 15 || padding < 0 ||
      padding >
          buckyball_target::getBuckyballBallParam("IM2COL", "maxPadding") ||
      span < 0 || span % stride || tileRows <= 0 || tileRows > 16 ||
      (tileRows & (tileRows - 1)))
    return emitOpError("im2col_i8 geometry exceeds the ISA field ranges");
  int64_t windows = (span / stride + 1) * (span / stride + 1);
  if (windows > 1023)
    return emitOpError("im2col_i8 has more windows than one instruction");
  uint64_t rs2 = field(kernel, 0, 3) | field(stride, 4, 7) |
                 field(padding, 8, 11) | field(windows, 36, 45) |
                 field(tileRows, 46, 50);
  emitInstruction(
      builder, loc, buckyball_target::getBuckyballFunct7("IM2COL"),
      constantI64(builder, loc, bankFields({input}, output, false, side)),
      constantI64(builder, loc, rs2));
  return success();
}
