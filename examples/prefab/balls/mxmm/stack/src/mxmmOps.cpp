//===- mxmmOps.cpp - MxmmBall operations ----------------------------------===//
//
// The dialect of MxmmBall, its virtual operations building their physical ones
// and the instructions of the physical operations.
//
//===----------------------------------------------------------------------===//

#include "mxmmOps.h"

#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::mxmm_ball;
using mlir::buddy::bankFields;
using mlir::buddy::constantI64;
using mlir::buddy::emitInstruction;
using mlir::buddy::field;

#include "mxmmDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mxmmOps.cpp.inc"

void MxmmBallDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mxmmOps.cpp.inc"
      >();
}

namespace {
[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallDialect(
         [](DialectRegistry &registry) { registry.insert<MxmmBallDialect>(); }),
     true);

// Emits one chain step of `mnemonic`. An MXFP8 operand takes one byte per
// element and one scale per 32; an FP32 operand four bytes per element. A
// spans fullK columns, of which the step reads k from startK; only the window
// format encodes fullK and startK.
LogicalResult emitMatmul(Operation *op, OpBuilder &builder, StringRef mnemonic,
                         bool mx, bool window, int64_t a, int64_t b, int64_t c,
                         int64_t rows, int64_t cols, int64_t k, int64_t fullK,
                         int64_t startK, int64_t base, bool first, bool last) {
  const auto &target = buckyball_target::getBuckyballTarget();
  int64_t bytes = target.bankDepth * 16;
  int64_t aElements = rows * fullK, bElements = cols * k;
  if (target.bankWidthBits != 128 ||
      !(rows == 1 || (rows > 0 && rows % 16 == 0)) || rows > 4095 ||
      cols <= 0 || cols % 16 || cols > 4095 || k <= 0 || k % (mx ? 32 : 4) ||
      k >= (1LL << 33) || base < 0 || base > 63)
    return op->emitOpError("shape exceeds the ISA geometry");
  if (window && (fullK % 32 || startK % 32 || startK + k > fullK ||
                 fullK > 65535 || startK > 65535))
    return op->emitOpError("A window exceeds full_k");
  if ((mx ? aElements + aElements / 32 : aElements * 4) > bytes ||
      (mx ? bElements + bElements / 32 : bElements * 4) > bytes ||
      base * 16 + rows * cols * 4 > bytes)
    return op->emitOpError("operands exceed the bank");
  if (a == b || a == c || b == c)
    return op->emitOpError("requires distinct A, B and C banks");
  uint64_t rs2 = field(rows, 0, 11) | field(cols, 12, 23) |
                 field(first, 24, 24) | field(last, 25, 25) |
                 field(base, 26, 31);
  if (window)
    rs2 |= field(fullK, 32, 47) | field(startK, 48, 63);
  emitInstruction(
      builder, op->getLoc(), buckyball_target::getBuckyballFunct7(mnemonic),
      constantI64(builder, op->getLoc(), bankFields({a, b}, c, false, k)),
      constantI64(builder, op->getLoc(), rs2));
  return success();
}
} // namespace

SmallVector<Value, 2> BankMatmulMxfp8Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankMatmulMxfp8Op::getDestinationBank() { return getC(); }
bool BankMatmulMxfp8Op::readsDestination() { return false; }
Operation *BankMatmulMxfp8Op::buildPhysical(OpBuilder &builder,
                                            ArrayRef<int64_t> sources,
                                            int64_t destination) {
  return MatmulMxfp8Op::create(builder, getLoc(), sources[0], sources[1],
                               destination, getRows(), getCols(), getK(),
                               getBase(), getFirst(), getLast());
}

SmallVector<int64_t, 2> MatmulMxfp8Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t MatmulMxfp8Op::getDestinationBank() { return getC(); }
LogicalResult MatmulMxfp8Op::lowerToInstructions(OpBuilder &builder) {
  return emitMatmul(*this, builder, "MXMM_MXFP8", true, false, getA(), getB(),
                    getC(), getRows(), getCols(), getK(), getK(), 0, getBase(),
                    getFirst(), getLast());
}

SmallVector<Value, 2> BankMatmulMxfp8WindowOp::getSourceBanks() {
  return {getA(), getB()};
}
Value BankMatmulMxfp8WindowOp::getDestinationBank() { return getC(); }
bool BankMatmulMxfp8WindowOp::readsDestination() { return false; }
Operation *BankMatmulMxfp8WindowOp::buildPhysical(OpBuilder &builder,
                                                  ArrayRef<int64_t> sources,
                                                  int64_t destination) {
  return MatmulMxfp8WindowOp::create(builder, getLoc(), sources[0], sources[1],
                                     destination, getRows(), getCols(), getK(),
                                     getFullK(), getStartK(), getBase(),
                                     getFirst(), getLast());
}

SmallVector<int64_t, 2> MatmulMxfp8WindowOp::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t MatmulMxfp8WindowOp::getDestinationBank() { return getC(); }
LogicalResult MatmulMxfp8WindowOp::lowerToInstructions(OpBuilder &builder) {
  return emitMatmul(*this, builder, "MXMM_MXFP8_WINDOW", true, true, getA(),
                    getB(), getC(), getRows(), getCols(), getK(), getFullK(),
                    getStartK(), getBase(), getFirst(), getLast());
}

SmallVector<Value, 2> BankMatmulFma32Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankMatmulFma32Op::getDestinationBank() { return getC(); }
bool BankMatmulFma32Op::readsDestination() { return false; }
Operation *BankMatmulFma32Op::buildPhysical(OpBuilder &builder,
                                            ArrayRef<int64_t> sources,
                                            int64_t destination) {
  return MatmulFma32Op::create(builder, getLoc(), sources[0], sources[1],
                               destination, getRows(), getCols(), getK(),
                               getBase(), getFirst(), getLast());
}

SmallVector<int64_t, 2> MatmulFma32Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t MatmulFma32Op::getDestinationBank() { return getC(); }
LogicalResult MatmulFma32Op::lowerToInstructions(OpBuilder &builder) {
  return emitMatmul(*this, builder, "MXMM_FMA32", false, false, getA(), getB(),
                    getC(), getRows(), getCols(), getK(), getK(), 0, getBase(),
                    getFirst(), getLast());
}

SmallVector<Value, 2> BankMatmulF32Op::getSourceBanks() {
  return {getA(), getB()};
}
Value BankMatmulF32Op::getDestinationBank() { return getC(); }
bool BankMatmulF32Op::readsDestination() { return false; }
Operation *BankMatmulF32Op::buildPhysical(OpBuilder &builder,
                                          ArrayRef<int64_t> sources,
                                          int64_t destination) {
  return MatmulF32Op::create(builder, getLoc(), sources[0], sources[1],
                             destination, getRows(), getCols(), getK(),
                             getBase(), getFirst(), getLast());
}

SmallVector<int64_t, 2> MatmulF32Op::getSourceBanks() {
  return {int64_t(getA()), int64_t(getB())};
}
int64_t MatmulF32Op::getDestinationBank() { return getC(); }
LogicalResult MatmulF32Op::lowerToInstructions(OpBuilder &builder) {
  return emitMatmul(*this, builder, "MXMM_F32", false, false, getA(), getB(),
                    getC(), getRows(), getCols(), getK(), getK(), 0, getBase(),
                    getFirst(), getLast());
}
