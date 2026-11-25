#include "mxint5gemvOps.h"
#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
using namespace mlir;
using namespace ::buddy::mxint5gemv_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;
#include "mxint5gemvDialect.cpp.inc"
#define GET_OP_CLASSES
#include "mxint5gemvOps.cpp.inc"
void Mxint5gemvBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxint5gemvOps.cpp.inc"
      >();
}
namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<Mxint5gemvBallDialect>();
       arith::registerValueBoundsOpInterfaceExternalModels(registry);
       scf::registerValueBoundsOpInterfaceExternalModels(registry);
     }),
     true);
} // namespace
SmallVector<Value, 2> BankGemvMxint5Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankGemvMxint5Op::getDestinationBank() { return getC(); }
bool BankGemvMxint5Op::readsDestination() {
  return getAccumulate() || getUpdate();
}
Operation *BankGemvMxint5Op::buildPhysical(OpBuilder &builder,
                                           ArrayRef<int64_t> sources,
                                           int64_t destination) {
  return GemvMxint5Op::create(builder, getLoc(), sources[0], sources[1],
                              destination, getK(), getACodes(), getAExponents(),
                              getBBase(), getCBase(), getAccumulate(),
                              getFinalize(), getUpdate());
}
SmallVector<int64_t, 2> GemvMxint5Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t GemvMxint5Op::getDestinationBank() { return getC(); }
LogicalResult GemvMxint5Op::lowerToInstructions(OpBuilder &builder) {
  int64_t a = getA(), b = getB(), c = getC(), k = getK();
  int64_t ac = getACodes(), ae = getAExponents(), bb = getBBase();
  ValueBoundsOptions bounds{.closedUB = true, .allowIntegerType = true};
  Value base = getCBase();
  if (auto cast = base.getDefiningOp<arith::IndexCastOp>())
    base = cast.getIn();
  auto low = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::LB, base, nullptr, bounds);
  auto high = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, base, nullptr, bounds);
  if (failed(low) || failed(high) || *low < 0 ||
      *high > target::getTarget().bankDepth - 17)
    return emitOpError("C base must be proven within the 17-row state window");
  const auto &target = target::getTarget();
  int64_t packed = (5 * (k / 32) + 3) / 4 + 2 * ((k / 32 + 15) / 16);
  if (target.bankWidthBits != 128 || target.bankDepth != 1024 || k <= 0 ||
      k > 32768 || k % 32 || ac < 0 || ae < 0 || bb < 0 ||
      ac > target.bankDepth || ae > target.bankDepth || bb > target.bankDepth ||
      ac + k / 16 > target.bankDepth ||
      ae + (k / 32 + 15) / 16 > target.bankDepth ||
      bb + 8 * packed > target.bankDepth)
    return emitOpError(
        "GEMV window exceeds the bank or K is not aligned to 32");
  if (a == b || a == c || b == c)
    return emitOpError("GEMV requires distinct A, B and C banks");
  Value offset = arith::ShLIOp::create(builder, getLoc(), getCBase(),
                                       constantI64(builder, getLoc(), 42));
  Value rs2 = arith::AddIOp::create(
      builder, getLoc(), offset,
      constantI64(builder, getLoc(),
                  field(ac, 0, 13) | field(ae, 14, 27) | field(bb, 28, 41) |
                      field(getAccumulate(), 56, 56) |
                      field(getFinalize(), 57, 57)));
  emitInstruction(builder, getLoc(), target::getFunct7("MXINT5GEMV"),
                  constantI64(builder, getLoc(),
                              bankFields({a, b}, c, getAccumulate(), k) |
                                  (uint64_t(getUpdate()) << 63)),
                  rs2);
  return success();
}
