//===- gemminiOps.cpp - GemminiBall operations ----------------------------===//
//
// The dialect of GemminiBall, its virtual operations building their physical
// ones and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "gemminiOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::gemmini_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "gemminiDialect.cpp.inc"

#define GET_OP_CLASSES
#include "gemminiOps.cpp.inc"

void GemminiBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gemminiOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect([](DialectRegistry &registry) {
       registry.insert<GemminiBallDialect>();
     }),
     true);
} // namespace

SmallVector<Value, 2> BankMatmulI8TileOp::getSourceBanks() {
  return {getA(), getB()};
}
Value BankMatmulI8TileOp::getDestinationBank() { return getC(); }
bool BankMatmulI8TileOp::readsDestination() { return getReadDestination(); }
Operation *BankMatmulI8TileOp::buildPhysical(OpBuilder &builder,
                                             ArrayRef<int64_t> sources,
                                             int64_t destination) {
  return MatmulI8TileOp::create(
      builder, getLoc(), builder.getI64IntegerAttr(sources[0]),
      builder.getI64IntegerAttr(sources[1]),
      builder.getI64IntegerAttr(destination), getKAttr(),
      getReadDestinationAttr(), getShiftAttr());
}

SmallVector<int64_t, 2> MatmulI8TileOp::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t MatmulI8TileOp::getDestinationBank() { return getC(); }
LogicalResult MatmulI8TileOp::lowerToInstructions(OpBuilder &builder) {
  Location loc = getLoc();
  bool readDestination = getReadDestination();
  if (readDestination != getShift().has_value())
    return emitOpError(
        "matmul_i8_tile takes a shift exactly with read_destination");
  int64_t k = getK(), a = getA(), b = getB(), c = getC();
  int64_t shift = readDestination ? int64_t(*getShift()) : 0;
  int64_t meshRows = buckyball_target::getBuckyballBallParam(
      "GEMMINI_COMPUTE_PRELOADED", "meshRows");
  int64_t meshColumns = buckyball_target::getBuckyballBallParam(
      "GEMMINI_COMPUTE_PRELOADED", "meshColumns");
  const auto &target = buckyball_target::getBuckyballTarget();
  if (target.bankWidthBits != 128 || meshRows != 16 || meshColumns != 16 ||
      target.bankDepth % 16 || target.bankDepth > 1024 || a == b || k <= 0 ||
      k % 16 || k > 64 * target.bankDepth || a == c || b == c)
    return emitOpError("matmul_i8_tile requires distinct A, B and C banks, k "
                       "a multiple of 16 within 64 groups and a 16x16 mesh");
  int64_t steps = k / 16, groupBlocks = target.bankDepth / 16;
  uint64_t config = ~0ULL;
  for (int64_t step = 0; step < steps; ++step) {
    uint64_t group = step / groupBlocks, base = step % groupBlocks * 16;
    bool accumulate = readDestination || step > 0;
    // CONFIG: rs2[4] picks the weight-stationary array, rs2[40:9] the shift.
    uint64_t word =
        field(accumulate, 4, 4) | field(step == steps - 1 ? shift : 0, 9, 40);
    if (word != config)
      emitInstruction(
          builder, loc, buckyball_target::getBuckyballFunct7("GEMMINI_CONFIG"),
          constantI64(builder, loc, 0), constantI64(builder, loc, word));
    config = word;
    // rs2: the subcommand in [3:0], the first and second source row bases in
    // [15:6] and [25:16], their groups in [41:36] and [47:42]; C is at row 0.
    uint64_t bases = field(base, 6, 15) | field(base, 16, 25) |
                     field(group, 36, 41) | field(group, 42, 47);
    if (accumulate) {
      emitInstruction(
          builder, loc,
          buckyball_target::getBuckyballFunct7("GEMMINI_COMPUTE_ACCUMULATED"),
          constantI64(builder, loc, bankFields({a, b}, c, true, 16)),
          constantI64(builder, loc, 3 | bases));
      continue;
    }
    // The output-stationary preload reads sixteen rows of B.
    emitInstruction(builder, loc,
                    buckyball_target::getBuckyballFunct7("GEMMINI_PRELOAD"),
                    constantI64(builder, loc, bankFields({b}, c, false, 16)),
                    constantI64(builder, loc,
                                1 | field(base, 6, 15) | field(group, 36, 41)));
    emitInstruction(
        builder, loc,
        buckyball_target::getBuckyballFunct7("GEMMINI_COMPUTE_PRELOADED"),
        constantI64(builder, loc, bankFields({a, b}, c, false, 16)),
        constantI64(builder, loc, 2 | bases));
  }
  return success();
}
