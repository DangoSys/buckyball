//===- lutOps.cpp - LutBall operations ------------------------------------===//
//
// The dialect of LutBall, its virtual operations building their physical ones
// and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "lutOps.h"

#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"

using namespace mlir;
using namespace ::buddy::lut_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "lutDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lutOps.cpp.inc"

void LutBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lutOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect(
         [](DialectRegistry &registry) { registry.insert<LutBallDialect>(); }),
     true);
} // namespace

SmallVector<Value, 2> BankLookupI8Op::getSourceBanks() {
  return {getCodes(), getTable()};
}
Value BankLookupI8Op::getDestinationBank() { return getOutput(); }
bool BankLookupI8Op::readsDestination() { return false; }
Operation *BankLookupI8Op::buildPhysical(OpBuilder &builder,
                                         ArrayRef<int64_t> sources,
                                         int64_t destination) {
  return LookupI8Op::create(builder, getLoc(), sources[0], sources[1],
                            destination, getRows());
}

SmallVector<int64_t, 2> LookupI8Op::getSourceBanks() {
  return {int64_t(getCodes()), int64_t(getTable())};
}
int64_t LookupI8Op::getDestinationBank() { return getOutput(); }
LogicalResult LookupI8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), codes = getCodes(), table = getTable(),
          output = getOutput();
  if (rows <= 0 || rows > target::getTarget().bankDepth || codes == table ||
      codes == output || table == output)
    return emitOpError(
        "lookup_i8 requires distinct banks and rows within bank depth");
  emitInstruction(builder, loc, target::getFunct7("LUT"),
                  constantI64(builder, loc,
                              bankFields({codes, table}, output, false, rows)),
                  constantI64(builder, loc, 0));
  return success();
}

SmallVector<Value, 2> BankLookupI32Op::getSourceBanks() {
  return {getCodes(), getTable()};
}
Value BankLookupI32Op::getDestinationBank() { return getOutput(); }
bool BankLookupI32Op::readsDestination() { return false; }
Operation *BankLookupI32Op::buildPhysical(OpBuilder &builder,
                                          ArrayRef<int64_t> sources,
                                          int64_t destination) {
  return LookupI32Op::create(builder, getLoc(), sources[0], sources[1],
                             destination, getRows());
}

SmallVector<int64_t, 2> LookupI32Op::getSourceBanks() {
  return {int64_t(getCodes()), int64_t(getTable())};
}
int64_t LookupI32Op::getDestinationBank() { return getOutput(); }
LogicalResult LookupI32Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), codes = getCodes(), table = getTable(),
          output = getOutput();
  if (target::getTarget().bankDepth < 256 || rows <= 0 ||
      rows > target::getTarget().bankDepth || codes == table ||
      codes == output || table == output)
    return emitOpError(
        "lookup_i32 requires distinct banks and rows within bank depth");
  emitInstruction(builder, loc, target::getFunct7("LUT"),
                  constantI64(builder, loc,
                              bankFields({codes, table}, output, false, rows)),
                  constantI64(builder, loc, 1));
  return success();
}

SmallVector<Value, 2> BankSoftmaxI8Op::getSourceBanks() {
  return {getCodes(), getTable()};
}
Value BankSoftmaxI8Op::getDestinationBank() { return getOutput(); }
bool BankSoftmaxI8Op::readsDestination() { return false; }
Operation *BankSoftmaxI8Op::buildPhysical(OpBuilder &builder,
                                          ArrayRef<int64_t> sources,
                                          int64_t destination) {
  return SoftmaxI8Op::create(builder, getLoc(), sources[0], sources[1],
                             destination, getRows(), getMultiplier());
}

SmallVector<int64_t, 2> SoftmaxI8Op::getSourceBanks() {
  return {int64_t(getCodes()), int64_t(getTable())};
}
int64_t SoftmaxI8Op::getDestinationBank() { return getOutput(); }
LogicalResult SoftmaxI8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), codes = getCodes(), table = getTable(),
          output = getOutput();
  if (getMultiplier() <= 0 || getMultiplier() >= (1LL << 30) ||
      target::getTarget().bankDepth < 256 || rows <= 0 ||
      rows > target::getTarget().bankDepth || codes == table ||
      codes == output || table == output)
    return emitOpError(
        "softmax_i8 requires distinct banks and rows within bank depth");
  emitInstruction(builder, loc, target::getFunct7("LUT"),
                  constantI64(builder, loc,
                              bankFields({codes, table}, output, false, rows)),
                  constantI64(builder, loc, 2 | (getMultiplier() << 32)));
  return success();
}
