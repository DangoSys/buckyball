#include "mxint8unpackOps.h"
#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace ::buddy::mxint8unpack_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
#include "mxint8unpackDialect.cpp.inc"
#define GET_OP_CLASSES
#include "mxint8unpackOps.cpp.inc"

void Mxint8UnpackBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxint8unpackOps.cpp.inc"
      >();
}
namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Mxint8UnpackBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankUnpackOp::getSourceBanks() { return {getInput()}; }
Value BankUnpackOp::getDestinationBank() { return getOutput(); }
bool BankUnpackOp::readsDestination() { return false; }
Operation *BankUnpackOp::buildPhysical(OpBuilder &builder,
                                       ArrayRef<int64_t> sources,
                                       int64_t destination) {
  return UnpackOp::create(builder, getLoc(), sources[0], destination,
                          getBlocks());
}
SmallVector<int64_t, 2> UnpackOp::getSourceBanks() {
  return {int64_t(getInput())};
}
int64_t UnpackOp::getDestinationBank() { return getOutput(); }
LogicalResult UnpackOp::lowerToInstructions(OpBuilder &builder) {
  const auto &target = target::getTarget();
  int64_t input = getInput(), output = getOutput();
  if (target.bankWidthBits != 128 || target.bankDepth != 1024 ||
      input == output)
    return emitOpError("requires two distinct 16 KiB banks");
  APInt blocks;
  if (matchPattern(getBlocks(), m_ConstantInt(&blocks)) &&
      (blocks.isNegative() || blocks.isZero() || blocks.ugt(128)))
    return emitOpError("blocks must be between one and 128");
  auto loc = getLoc();
  Value count = arith::ShLIOp::create(builder, loc, getBlocks(),
                                      constantI64(builder, loc, 30));
  Value rs1 = arith::OrIOp::create(
      builder, loc, count,
      constantI64(builder, loc, bankFields({input}, output, false, 0)));
  emitInstruction(builder, loc, target::getFunct7("MXINT8UNPACK"), rs1,
                  constantI64(builder, loc, 0));
  return success();
}
