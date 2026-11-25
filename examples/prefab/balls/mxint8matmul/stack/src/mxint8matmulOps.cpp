//===- mxint8matmulOps.cpp - Mxint8matmulBall operations ------------------===//
//
// The dialect of Mxint8matmulBall, its virtual operation building its
// physical one and the instruction of the physical operation.
//
//===----------------------------------------------------------------------===//

#include "mxint8matmulOps.h"

#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;
using namespace ::buddy::mxint8matmul_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "mxint8matmulDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mxint8matmulOps.cpp.inc"

void Mxint8matmulBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxint8matmulOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Mxint8matmulBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankMatmulMxint8Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankMatmulMxint8Op::getDestinationBank() { return getC(); }
bool BankMatmulMxint8Op::readsDestination() { return getAccumulate(); }
Operation *BankMatmulMxint8Op::buildPhysical(OpBuilder &builder,
                                             ArrayRef<int64_t> sources,
                                             int64_t destination) {
  return MatmulMxint8Op::create(builder, getLoc(), sources[0], sources[1],
                                destination, getM(), getN(), getK(),
                                getAccumulate());
}

SmallVector<int64_t, 2> MatmulMxint8Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t MatmulMxint8Op::getDestinationBank() { return getC(); }
LogicalResult MatmulMxint8Op::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t a = getA(), b = getB(), c = getC(), n = getN(), k = getK();
  bool accumulate = getAccumulate();
  const auto &target = target::getTarget();
  int64_t packed = k / 16 + (k + 511) / 512;
  if (target.bankWidthBits != 128 || n <= 0 || n >= 1024 || n % 4 || k <= 0 ||
      k % 32 || n * packed > target.bankDepth)
    return emitOpError("matmul_mxint8 exceeds the bank");
  if (auto m = getConstantIntValue(getM());
      m && (*m <= 0 || *m >= 1024 || *m * packed > target.bankDepth ||
            *m * n / 4 > target.bankDepth))
    return emitOpError("matmul_mxint8 exceeds the bank");
  if (a == b || a == c || b == c)
    return emitOpError("matmul_mxint8 requires distinct banks");
  // A, B and C start at bank row 0 of their banks.
  emitInstruction(
      builder, loc, target::getFunct7("MXINT8MM"),
      constantI64(builder, loc, bankFields({a, b}, c, accumulate, k)),
      arith::AddIOp::create(
          builder, loc,
          arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                     getM()),
          constantI64(builder, loc,
                      field(n, 10, 19) | field(accumulate, 62, 62))));
  return success();
}
