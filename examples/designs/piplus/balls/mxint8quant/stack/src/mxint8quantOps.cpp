//===- mxint8quantOps.cpp - Mxint8quantBall operations --------------------===//
//
// The dialect of Mxint8quantBall, its virtual operation building its physical
// one and the instruction of the physical operation.
//
//===----------------------------------------------------------------------===//

#include "mxint8quantOps.h"

#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"

using namespace mlir;
using namespace ::buddy::mxint8quant_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "mxint8quantDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mxint8quantOps.cpp.inc"

void Mxint8quantBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxint8quantOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Mxint8quantBallDialect>();
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
                               getRows(), getK(), getColumns(), getInBase(),
                               getOutBase(), getInPitch(), getUpdate());
}

SmallVector<int64_t, 2> QuantMxint8Op::getSourceBanks() {
  return {int64_t(getInput())};
}
int64_t QuantMxint8Op::getDestinationBank() { return getOutput(); }
LogicalResult QuantMxint8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), k = getK(), input = getInput(),
          output = getOutput(), inBase = getInBase(), outBase = getOutBase(),
          inPitch = getInPitch();
  const auto &target = target::getTarget();
  int64_t packed = k / 16 + (k + 511) / 512;
  // The bank rows of an input row and the input rows.
  int64_t width = getColumns() ? rows / 4 : k / 4,
          count = getColumns() ? k : rows;
  if (target.bankWidthBits != 128 || rows <= 0 || k <= 0 || k % 32 ||
      k >= 65536 || inPitch < width ||
      inBase + (count - 1) * inPitch + width > target.bankDepth ||
      outBase + rows * packed > target.bankDepth || (getColumns() && rows % 4))
    return emitOpError("quant_mxint8 exceeds the bank");
  if (input == output)
    return emitOpError("quant_mxint8 requires distinct input and output banks");
  // Input at inBase (16-29), output at outBase (30-43), inPitch (45-58).
  emitInstruction(builder, loc, target::getFunct7("MXINT8QUANT"),
                  constantI64(builder, loc,
                              bankFields({input}, output, false, rows) |
                                  field(getUpdate(), 63, 63)),
                  constantI64(builder, loc,
                              field(k, 0, 15) | field(inBase, 16, 29) |
                                  field(outBase, 30, 43) |
                                  field(getColumns(), 44, 44) |
                                  field(inPitch, 45, 58)));
  return success();
}
