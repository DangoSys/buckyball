//===- vectorOps.cpp - VecBall operations ---------------------------------===//
//
// The dialect of VecBall, its virtual operations building their physical ones
// and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "vectorOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::vector_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "vectorDialect.cpp.inc"

#define GET_OP_CLASSES
#include "vectorOps.cpp.inc"

void VecBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "vectorOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect(
         [](DialectRegistry &registry) { registry.insert<VecBallDialect>(); }),
     true);
} // namespace

SmallVector<Value, 2> BankMatmulAtI8Op::getSourceBanks() {
  return {getAt(), getB()};
}
Value BankMatmulAtI8Op::getDestinationBank() { return getC(); }
bool BankMatmulAtI8Op::readsDestination() { return true; }
Operation *BankMatmulAtI8Op::buildPhysical(OpBuilder &builder,
                                           ArrayRef<int64_t> sources,
                                           int64_t destination) {
  return MatmulAtI8Op::create(builder, getLoc(), sources[0], sources[1],
                              destination, getRows());
}

SmallVector<int64_t, 2> MatmulAtI8Op::getSourceBanks() {
  return {int64_t(getAt()), int64_t(getB())};
}
int64_t MatmulAtI8Op::getDestinationBank() { return getC(); }
LogicalResult MatmulAtI8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), at = getAt(), b = getB(), c = getC();
  const auto &target = buckyball_target::getBuckyballTarget();
  if (target.bankWidthBits != 128 || rows <= 0 || rows > target.bankDepth)
    return emitOpError("matmul_at_i8 rows exceed the bank");
  if (at == c || b == c)
    return emitOpError("matmul_at_i8 requires C apart from A^T and B");
  emitInstruction(builder, loc,
                  buckyball_target::getBuckyballFunct7("VECMAT16"),
                  constantI64(builder, loc, bankFields({at, b}, c, true, rows)),
                  constantI64(builder, loc, 0));
  return success();
}
