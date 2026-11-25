//===- mxint8actOps.cpp - Mxint8actBall operations ------------------------===//
//
// The dialect of Mxint8actBall, its virtual operation building its physical one
// and the instruction of the physical operation.
//
//===----------------------------------------------------------------------===//

#include "mxint8actOps.h"

#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"

using namespace mlir;
using namespace ::buddy::mxint8act_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;

#include "mxint8actDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mxint8actOps.cpp.inc"

void Mxint8actBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxint8actOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Mxint8actBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankActMxint8Op::getSourceBanks() {
  return {getInput(), getTable()};
}
Value BankActMxint8Op::getDestinationBank() { return getOutput(); }
bool BankActMxint8Op::readsDestination() { return false; }
Operation *BankActMxint8Op::buildPhysical(OpBuilder &builder,
                                          ArrayRef<int64_t> sources,
                                          int64_t destination) {
  return ActMxint8Op::create(builder, getLoc(), sources[0], sources[1],
                             destination, getRows(), getK(), getInBase(),
                             getOutBase(), getTableBase(), getGated());
}
SmallVector<int64_t, 2> ActMxint8Op::getSourceBanks() {
  return {int64_t(getInput()), int64_t(getTable())};
}
int64_t ActMxint8Op::getDestinationBank() { return getOutput(); }
LogicalResult ActMxint8Op::lowerToInstructions(OpBuilder &builder) {
  const auto &target = target::getTarget();
  int64_t k = getK(), rows = getRows(), input = getInput(), table = getTable(),
          output = getOutput();
  int64_t inputRows = ((k + k / 32 + 15) / 16) * (getGated() ? 2 : 1);
  int64_t outputRows = (k + k / 32 + 15) / 16;
  if (target.bankWidthBits != 128 || k <= 0 || k > 65535 || k % 32 ||
      rows <= 0 || getInBase() < 0 || getOutBase() < 0 || getTableBase() < 0 ||
      getInBase() + rows * inputRows > target.bankDepth ||
      getOutBase() + rows * outputRows > target.bankDepth ||
      getTableBase() + 1024 > target.bankDepth || input == table ||
      input == output || table == output)
    return emitOpError(
        "MXINT8 activation requires distinct banks and complete in-bank spans");
  int64_t payload = int64_t(getGated()) | (k << 1) | (getInBase() << 17) |
                    (getOutBase() << 31) | (getTableBase() << 45);
  emitInstruction(builder, getLoc(), target::getFunct7("MXINT8ACT"),
                  constantI64(builder, getLoc(),
                              bankFields({input, table}, output, false, rows)),
                  constantI64(builder, getLoc(), payload));
  return success();
}
