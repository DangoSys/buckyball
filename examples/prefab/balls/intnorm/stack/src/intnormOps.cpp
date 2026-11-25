#include "intnormOps.h"
#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"

using namespace mlir;
using namespace ::buddy::intnorm_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "intnormDialect.cpp.inc"
#define GET_OP_CLASSES
#include "intnormOps.cpp.inc"

void IntnormBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "intnormOps.cpp.inc"
      >();
}
namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<IntnormBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankNormMxint8Op::getSourceBanks() {
  return {getInput(), getCoefficients()};
}
Value BankNormMxint8Op::getDestinationBank() { return getOutput(); }
bool BankNormMxint8Op::readsDestination() { return false; }
Operation *BankNormMxint8Op::buildPhysical(OpBuilder &builder,
                                           ArrayRef<int64_t> sources,
                                           int64_t destination) {
  return NormMxint8Op::create(builder, getLoc(), sources[0], sources[1],
                              destination, getRows(), getK(), getInBase(),
                              getCoefficientBase(), getOutBase());
}
SmallVector<int64_t, 2> NormMxint8Op::getSourceBanks() {
  return {int64_t(getInput()), int64_t(getCoefficients())};
}
int64_t NormMxint8Op::getDestinationBank() { return getOutput(); }
LogicalResult NormMxint8Op::lowerToInstructions(OpBuilder &builder) {
  const auto &target = target::getTarget();
  int64_t rows = getRows(), k = getK(), a = getInput(), b = getCoefficients(),
          c = getOutput(), in = getInBase(), coeff = getCoefficientBase(),
          out = getOutBase();
  if (target.bankWidthBits != 128 || rows <= 0 || rows >= 65536 || k <= 0 ||
      k >= 65536 || k % 512 || in < 0 || in >= 16384 || coeff < 0 ||
      coeff >= 16384 || out < 0 || out >= 16384 ||
      in + rows * k / 4 > target.bankDepth ||
      coeff + 1 + k / 2 > target.bankDepth ||
      out + rows * (k + k / 32) / 16 + k / 4 > target.bankDepth)
    return emitOpError("integer normalization exceeds the bank or ISA fields");
  if (a == b || a == c || b == c)
    return emitOpError("integer normalization requires three distinct banks");
  Location loc = getLoc();
  int64_t funct = target::getFunct7("INTNORM");
  if ((funct >> 4) != 4)
    return emitOpError(
        "integer normalization requires a two-read, one-write opcode");
  emitInstruction(builder, loc, funct,
                  constantI64(builder, loc, bankFields({a, b}, c, false, rows)),
                  constantI64(builder, loc,
                              field(k, 0, 15) | field(in, 16, 29) |
                                  field(out, 30, 43) | field(coeff, 44, 57)));
  return success();
}
