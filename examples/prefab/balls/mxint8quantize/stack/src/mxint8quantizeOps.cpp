//===- mxint8quantizeOps.cpp - Mxint8quantizeBall operations --------------===//
//
// The dialect of Mxint8quantizeBall, its virtual operation building its
// physical one and the instruction of the physical operation.
//
//===----------------------------------------------------------------------===//

#include "mxint8quantizeOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::mxint8quantize_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "mxint8quantizeDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mxint8quantizeOps.cpp.inc"

void Mxint8quantizeBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxint8quantizeOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Mxint8quantizeBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankQuantMxint8Op::getSourceBanks() {
  return {getInput()};
}
Value BankQuantMxint8Op::getDestinationBank() { return getOutput(); }
bool BankQuantMxint8Op::readsDestination() { return false; }
Operation *BankQuantMxint8Op::buildPhysical(OpBuilder &builder,
                                            ArrayRef<int64_t> sources,
                                            int64_t destination) {
  return QuantMxint8Op::create(builder, getLoc(), sources[0], destination,
                               getRows(), getK());
}

SmallVector<int64_t, 2> QuantMxint8Op::getSourceBanks() {
  return {int64_t(getInput())};
}
int64_t QuantMxint8Op::getDestinationBank() { return getOutput(); }
LogicalResult QuantMxint8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), k = getK(), input = getInput(),
          output = getOutput();
  const auto &target = buckyball_target::getBuckyballTarget();
  int64_t packed = k / 16 + (k + 511) / 512;
  if (target.bankWidthBits != 128 || rows <= 0 || k <= 0 || k % 32 ||
      k >= 65536 || rows * k / 4 > target.bankDepth ||
      rows * packed > target.bankDepth)
    return emitOpError("quant_mxint8 exceeds the bank");
  if (input == output)
    return emitOpError("quant_mxint8 requires distinct input and output banks");
  // The rows start at bank row 0 of input and output.
  emitInstruction(
      builder, loc, buckyball_target::getBuckyballFunct7("MXINT8QUANT"),
      constantI64(builder, loc, bankFields({input}, output, false, rows)),
      constantI64(builder, loc, field(k, 0, 15)));
  return success();
}
