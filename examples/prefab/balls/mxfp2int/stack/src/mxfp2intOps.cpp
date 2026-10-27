//===- mxfp2intOps.cpp - Mxfp2IntBall operations --------------------------===//
//
// The dialect of Mxfp2IntBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "mxfp2intOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::mxfp2int_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "mxfp2intDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mxfp2intOps.cpp.inc"

void Mxfp2IntBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxfp2intOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Mxfp2IntBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankMxfp4I8Op::getSourceBanks() {
  return {getData(), getScales()};
}
Value BankMxfp4I8Op::getDestinationBank() { return getOutput(); }
bool BankMxfp4I8Op::readsDestination() { return false; }
Operation *BankMxfp4I8Op::buildPhysical(OpBuilder &builder,
                                        ArrayRef<int64_t> sources,
                                        int64_t destination) {
  return Mxfp4I8Op::create(builder, getLoc(), sources[0], sources[1],
                           destination, getRows());
}

SmallVector<int64_t, 2> Mxfp4I8Op::getSourceBanks() {
  return {int64_t(getData()), int64_t(getScales())};
}
int64_t Mxfp4I8Op::getDestinationBank() { return getOutput(); }
LogicalResult Mxfp4I8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), data = getData(), scales = getScales(),
          output = getOutput();
  const auto &target = buckyball_target::getBuckyballTarget();
  if (target.bankWidthBits != 128 || rows <= 0 || rows * 2 > target.bankDepth)
    return emitOpError("mxfp4_i8 rows exceed half the bank");
  if (data == scales || data == output || scales == output)
    return emitOpError(
        "mxfp4_i8 requires distinct data, scale and output banks");
  emitInstruction(builder, loc,
                  buckyball_target::getBuckyballFunct7("MXFP2INT"),
                  constantI64(builder, loc,
                              bankFields({data, scales}, output, false, rows)),
                  constantI64(builder, loc, 0));
  return success();
}
