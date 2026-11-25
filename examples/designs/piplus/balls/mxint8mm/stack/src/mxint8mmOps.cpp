//===- mxint8mmOps.cpp - Mxint8mmBall operations --------------------------===//
//
// The dialect of Mxint8mmBall, its virtual operation building its physical
// one and the instruction of the physical operation.
//
//===----------------------------------------------------------------------===//

#include "mxint8mmOps.h"

#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"

using namespace mlir;
using namespace ::buddy::mxint8mm_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "mxint8mmDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mxint8mmOps.cpp.inc"

void Mxint8mmBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxint8mmOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Mxint8mmBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankMatmulMxint8Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankMatmulMxint8Op::getDestinationBank() { return getC(); }
bool BankMatmulMxint8Op::readsDestination() {
  return getAccumulate() || getBias();
}
Operation *BankMatmulMxint8Op::buildPhysical(OpBuilder &builder,
                                             ArrayRef<int64_t> sources,
                                             int64_t destination) {
  return MatmulMxint8Op::create(builder, getLoc(), sources[0], sources[1],
                                destination, getM(), getN(), getK(),
                                getAccumulate(), getBias(), getABase(),
                                getCPitch(), getCBase(), getUpdate());
}

SmallVector<int64_t, 2> MatmulMxint8Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t MatmulMxint8Op::getDestinationBank() { return getC(); }
LogicalResult MatmulMxint8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t a = getA(), b = getB(), c = getC(), m = getM(), n = getN(),
          k = getK();
  bool accumulate = getAccumulate(), bias = getBias();
  int64_t aBase = getABase(), cPitch = getCPitch(), cBase = getCBase();
  const auto &target = target::getTarget();
  int64_t packed = k / 16 + (k + 511) / 512;
  if (target.bankWidthBits != 128 || m <= 0 || m >= 1024 || n <= 0 ||
      n >= 1024 || n % 4 || k <= 0 || k % 32 ||
      aBase + m * packed > target.bankDepth || n * packed > target.bankDepth ||
      cPitch < n / 4 || cBase + (m - 1) * cPitch + n / 4 > target.bankDepth ||
      (bias && cBase < n / 4))
    return emitOpError("matmul_mxint8 exceeds the bank");
  if (a == b || a == c || b == c)
    return emitOpError("matmul_mxint8 requires distinct banks");
  // A at aBase (20-33), B at bank row 0, C's rows cPitch (34-47) apart from
  // cBase (48-61).
  emitInstruction(
      builder, loc, target::getFunct7("MXINT8MM"),
      constantI64(builder, loc,
                  bankFields({a, b}, c, accumulate || bias, k) |
                      field(getUpdate(), 63, 63)),
      constantI64(builder, loc,
                  field(m, 0, 9) | field(n, 10, 19) | field(aBase, 20, 33) |
                      field(cPitch, 34, 47) | field(cBase, 48, 61) |
                      field(accumulate, 62, 62) | field(bias, 63, 63)));
  return success();
}
