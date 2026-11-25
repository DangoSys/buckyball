#include "mxint8gemvOps.h"
#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"
using namespace mlir;
using namespace ::buddy::mxint8gemv_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;
#include "mxint8gemvDialect.cpp.inc"
#define GET_OP_CLASSES
#include "mxint8gemvOps.cpp.inc"
void Mxint8gemvBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxint8gemvOps.cpp.inc"
      >();
}
namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Mxint8gemvBallDialect>();
     }),
     true);
} // namespace
SmallVector<Value, 2> BankGemvMxint8Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankGemvMxint8Op::getDestinationBank() { return getC(); }
bool BankGemvMxint8Op::readsDestination() { return getAccumulate(); }
Operation *BankGemvMxint8Op::buildPhysical(OpBuilder &builder,
                                           ArrayRef<int64_t> sources,
                                           int64_t destination) {
  return GemvMxint8Op::create(builder, getLoc(), sources[0], sources[1],
                              destination, getK(), getACodes(), getAExponents(),
                              getBBase(), getCBase(), getAccumulate(),
                              getFinalize());
}
SmallVector<int64_t, 2> GemvMxint8Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t GemvMxint8Op::getDestinationBank() { return getC(); }
LogicalResult GemvMxint8Op::lowerToInstructions(OpBuilder &builder) {
  int64_t a = getA(), b = getB(), c = getC(), k = getK();
  int64_t ac = getACodes(), ae = getAExponents(), bb = getBBase(),
          cb = getCBase();
  const auto &target = target::getTarget();
  int64_t packed = k / 16 + (k + 511) / 512;
  if (target.bankWidthBits != 128 || k <= 0 || k % 32 || ac < 0 || ae < 0 ||
      bb < 0 || cb < 0 || ac + k / 16 > target.bankDepth ||
      ae + (k / 32 + 15) / 16 > target.bankDepth ||
      bb + 8 * packed > target.bankDepth || cb + 17 > target.bankDepth)
    return emitOpError(
        "GEMV window exceeds the bank or K is not aligned to 32");
  if (a == b || a == c || b == c)
    return emitOpError("GEMV requires distinct A, B and C banks");
  emitInstruction(
      builder, getLoc(), target::getFunct7("MXINT8GEMV"),
      constantI64(builder, getLoc(), bankFields({a, b}, c, getAccumulate(), k)),
      constantI64(builder, getLoc(),
                  field(ac, 0, 13) | field(ae, 14, 27) | field(bb, 28, 41) |
                      field(cb, 42, 55) | field(getAccumulate(), 56, 56) |
                      field(getFinalize(), 57, 57)));
  return success();
}
