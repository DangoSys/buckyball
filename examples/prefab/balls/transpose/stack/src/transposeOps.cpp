//===- transposeOps.cpp - TransposeBall operations ------------------------===//
//
// The dialect of TransposeBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "transposeOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::transpose_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "transposeDialect.cpp.inc"

#define GET_OP_CLASSES
#include "transposeOps.cpp.inc"

void TransposeBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "transposeOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<TransposeBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankTransposeI8Op::getSourceBanks() {
  return {getInput()};
}
Value BankTransposeI8Op::getDestinationBank() { return getOutput(); }
bool BankTransposeI8Op::readsDestination() { return false; }
Operation *BankTransposeI8Op::buildPhysical(OpBuilder &builder,
                                            ArrayRef<int64_t> sources,
                                            int64_t destination) {
  return TransposeI8Op::create(builder, getLoc(), sources[0], destination,
                               getRows(), getCols());
}

SmallVector<int64_t, 2> TransposeI8Op::getSourceBanks() {
  return {int64_t(getInput())};
}
int64_t TransposeI8Op::getDestinationBank() { return getOutput(); }
LogicalResult TransposeI8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t input = getInput(), output = getOutput(), rows = getRows(),
          cols = getCols();
  if (input == output)
    return emitOpError("transpose_i8 requires distinct input and output banks");
  const auto &target = buckyball_target::getBuckyballTarget();
  int64_t lanes = target.bankWidthBits / 8;
  if (lanes <= 0 || target.bankWidthBits % 8 || rows <= 0 ||
      rows >= (1LL << 33) || rows > target.bankDepth || cols <= 0 ||
      cols % lanes || cols / lanes > 31)
    return emitOpError("transpose_i8 shape exceeds the bank or ISA geometry");
  emitInstruction(
      builder, loc, buckyball_target::getBuckyballFunct7("TRANSPOSE"),
      constantI64(builder, loc, bankFields({input}, output, false, rows)),
      constantI64(builder, loc, 8));
  return success();
}
