#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace buddy::buckyball;

namespace {
template <typename Op> LogicalResult verifyAddition(Op op) {
  if (op.getRows() <= 0 || op.getRows() > ((int64_t(1) << 34) - 1))
    return op.emitOpError("rows must be in [1, 17179869183]");
  if (op.getGroup() < 0 || op.getGroup() > 31)
    return op.emitOpError("group must be in [0, 31]");
  auto banks = op.getOperands();
  for (unsigned i = 0; i < banks.size(); ++i) {
    auto lhs = banks[i].template getDefiningOp<arith::ConstantIntOp>();
    if (lhs && (lhs.value() < 0 || lhs.value() > 1023))
      return op.emitOpError("bank ID must be in [0, 1023]");
    for (unsigned j = 0; j < i; ++j) {
      auto rhs = banks[j].template getDefiningOp<arith::ConstantIntOp>();
      if (banks[i] == banks[j] || (lhs && rhs && lhs.value() == rhs.value()))
        return op.emitOpError(
            "source, accumulator and output banks must differ");
    }
  }
  return success();
}
} // namespace

LogicalResult F32AddOp::verify() { return verifyAddition(*this); }
LogicalResult BankF32AddOp::verify() { return verifyAddition(*this); }
