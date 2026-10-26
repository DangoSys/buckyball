#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
template <typename Op> LogicalResult verifyMatmul(Op op) {
  if (op.getRows() <= 0 || op.getRows() > 4095)
    return op.emitOpError("rows must be in [1, 4095]");
  if (op.getK() <= 0 || op.getK() % 8 || op.getK() >= (int64_t(1) << 34))
    return op.emitOpError(
        "K must be a positive multiple of eight within 34 bits");
  for (auto base : {op.getOp1Base(), op.getOp2Base(), op.getWrBase()})
    if (base < 0 || base > 1023)
      return op.emitOpError("bank row bases must be in [0, 1023]");
  const auto &target = buckyball_target::getBuckyballTarget();
  if (target.bankWidthBits != 128 ||
      op.getOp1Base() + op.getRows() * (op.getK() / 8) > target.bankDepth ||
      op.getOp2Base() + op.getK() > target.bankDepth ||
      op.getWrBase() + op.getRows() > target.bankDepth)
    return op.emitOpError("matrix window exceeds target 128-bit bank capacity");
  auto banks = op.getOperands();
  for (unsigned i = 0; i < banks.size(); ++i) {
    auto lhs = banks[i].template getDefiningOp<arith::ConstantIntOp>();
    if (lhs && (lhs.value() < 0 || lhs.value() > 1023))
      return op.emitOpError("bank ID must be in [0, 1023]");
    for (unsigned j = 0; j < i; ++j) {
      auto rhs = banks[j].template getDefiningOp<arith::ConstantIntOp>();
      if (banks[i] == banks[j] || (lhs && rhs && lhs.value() == rhs.value()))
        return op.emitOpError("A, B and C banks must differ");
    }
  }
  return success();
}
} // namespace

LogicalResult RedmuleF16Op::verify() { return verifyMatmul(*this); }
LogicalResult BankRedmuleF16Op::verify() { return verifyMatmul(*this); }
