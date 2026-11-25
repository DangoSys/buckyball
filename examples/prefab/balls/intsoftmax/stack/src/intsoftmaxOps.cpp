#include "intsoftmaxOps.h"
#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace ::buddy::intsoftmax_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
#include "intsoftmaxDialect.cpp.inc"
#define GET_OP_CLASSES
#include "intsoftmaxOps.cpp.inc"

void IntSoftmaxBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "intsoftmaxOps.cpp.inc"
      >();
}
namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<IntSoftmaxBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankSoftmaxOp::getSourceBanks() {
  return {getInput(), getTable()};
}
Value BankSoftmaxOp::getDestinationBank() { return getOutput(); }
bool BankSoftmaxOp::readsDestination() { return false; }
Operation *BankSoftmaxOp::buildPhysical(OpBuilder &builder,
                                        ArrayRef<int64_t> sources,
                                        int64_t destination) {
  return SoftmaxOp::create(builder, getLoc(), sources[0], sources[1],
                           destination, getRows(), getColumns());
}
SmallVector<int64_t, 2> SoftmaxOp::getSourceBanks() {
  return {int64_t(getInput()), int64_t(getTable())};
}
int64_t SoftmaxOp::getDestinationBank() { return getOutput(); }
LogicalResult SoftmaxOp::lowerToInstructions(OpBuilder &builder) {
  const auto &target = target::getTarget();
  int64_t input = getInput(), table = getTable(), output = getOutput();
  if (target.bankWidthBits != 128 || target.bankDepth != 1024 ||
      input == table || input == output || table == output)
    return emitOpError("requires three distinct 16 KiB banks");
  APInt rows, columns;
  bool staticRows = matchPattern(getRows(), m_ConstantInt(&rows));
  bool staticColumns = matchPattern(getColumns(), m_ConstantInt(&columns));
  if (staticRows && (rows.isNegative() || rows.isZero() || rows.ugt(1024)))
    return emitOpError("logical rows must fit one bank");
  if (staticColumns &&
      (columns.isNegative() || columns.isZero() || columns.ugt(128)))
    return emitOpError("columns must be between one and 128");
  if (staticRows && staticColumns &&
      rows.getZExtValue() * ((columns.getZExtValue() + 3) / 4) > 1024)
    return emitOpError("softmax span exceeds bank capacity");
  auto loc = getLoc();
  Value count = arith::ShLIOp::create(builder, loc, getRows(),
                                      constantI64(builder, loc, 30));
  Value rs1 = arith::OrIOp::create(
      builder, loc, count,
      constantI64(builder, loc, bankFields({input, table}, output, false, 0)));
  emitInstruction(builder, loc, target::getFunct7("INTSOFTMAX"), rs1,
                  getColumns());
  return success();
}
