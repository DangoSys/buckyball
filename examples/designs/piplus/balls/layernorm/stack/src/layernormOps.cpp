//===- layernormOps.cpp - LayerNormBall operations
//----------------------------===//
//
// The dialect of LayerNormBall, its virtual operation building its physical one
// and the instruction of the physical operation.
//
//===----------------------------------------------------------------------===//

#include "layernormOps.h"

#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"

using namespace mlir;
using namespace ::buddy::layernorm_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "layernormDialect.cpp.inc"

#define GET_OP_CLASSES
#include "layernormOps.cpp.inc"

void LayerNormBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "layernormOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<LayerNormBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankLayerNormOp::getSourceBanks() {
  return {getInput(), getParams()};
}
Value BankLayerNormOp::getDestinationBank() { return getOutput(); }
bool BankLayerNormOp::readsDestination() { return false; }
Operation *BankLayerNormOp::buildPhysical(OpBuilder &builder,
                                          ArrayRef<int64_t> sources,
                                          int64_t destination) {
  return LayerNormOp::create(builder, getLoc(), sources[0], sources[1],
                             destination, getRows(), getW(), getRms());
}

SmallVector<int64_t, 2> LayerNormOp::getSourceBanks() {
  return {int64_t(getInput()), int64_t(getParams())};
}
int64_t LayerNormOp::getDestinationBank() { return getOutput(); }
LogicalResult LayerNormOp::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  int64_t rows = getRows(), w = getW(), input = getInput(),
          params = getParams(), output = getOutput();
  const auto &target = target::getTarget();
  int64_t lines = (w + 3) / 4;
  if (target.bankWidthBits != 128 || rows <= 0 || w < 16 || w > 2048 ||
      rows * lines > target.bankDepth ||
      (getRms() ? lines + 1 : 2 * lines + 1) > target.bankDepth)
    return emitOpError("layernorm exceeds the bank");
  if (input == output || params == output || input == params)
    return emitOpError("layernorm requires distinct banks");
  // Input, params and output start at bank row 0: the bank-row fields
  // in_base (16-29), out_base (30-43) and param_base (44-57) are 0.
  emitInstruction(
      builder, loc, target::getFunct7("LAYERNORM"),
      constantI64(builder, loc,
                  bankFields({input, params}, output, false, rows)),
      constantI64(builder, loc, field(w, 0, 15) | field(getRms(), 58, 58)));
  return success();
}
