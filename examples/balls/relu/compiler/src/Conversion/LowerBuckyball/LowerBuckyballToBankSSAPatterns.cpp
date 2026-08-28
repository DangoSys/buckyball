#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"

#include <algorithm>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kValuesPerLine = 4;

class ReluMatrixToBankSSAPattern : public OpRewritePattern<ReluMatrixOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluMatrixOp op,
                                PatternRewriter &b) const override {
    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        inputType != outputType || !inputType.getElementType().isInteger(32))
      return op.emitError("requires matching static memref<MxNxi32>");

    int64_t rows = inputType.getShape()[0];
    int64_t columns = inputType.getShape()[1];
    if (rows <= 0 || columns <= 0 || rows % 16 || columns % 16)
      return op.emitError("requires positive 16-aligned dimensions");

    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth <= 0 ||
        target.bankDepth % 16)
      return op.emitError(
          "ReluBall requires a 128-bit bank with 16-aligned depth");

    Location loc = op.getLoc();
    int64_t lines = target.bankDepth;
    int64_t values = lines * kValuesPerLine;
    int64_t total = rows * columns;
    Value bank = allocBank(b, loc, 1, 1);
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value four = b.create<arith::ConstantIndexOp>(loc, kValuesPerLine);
    Value columnsValue = b.create<arith::ConstantIndexOp>(loc, columns);
    auto packType = MemRefType::get({lines, kValuesPerLine}, b.getI32Type());

    for (int64_t base = 0; base < total; base += values) {
      int64_t count = std::min(values, total - base);
      Value inputPack = b.create<memref::AllocOp>(loc, packType);
      Value outputPack = b.create<memref::AllocOp>(loc, packType);
      Value zeroValue = b.create<arith::ConstantIntOp>(loc, 0, 32);
      b.create<linalg::FillOp>(loc, zeroValue, inputPack);

      Value countValue = b.create<arith::ConstantIndexOp>(loc, count);
      Value baseValue = b.create<arith::ConstantIndexOp>(loc, base);
      auto copyIn = b.create<scf::ForOp>(loc, zero, countValue, one);
      b.setInsertionPointToStart(copyIn.getBody());
      Value index = copyIn.getInductionVar();
      Value source = b.create<arith::AddIOp>(loc, baseValue, index);
      Value row = b.create<arith::DivUIOp>(loc, source, columnsValue);
      Value column = b.create<arith::RemUIOp>(loc, source, columnsValue);
      Value line = b.create<arith::DivUIOp>(loc, index, four);
      Value lane = b.create<arith::RemUIOp>(loc, index, four);
      Value value =
          b.create<memref::LoadOp>(loc, op.getInput(), ValueRange{row, column});
      b.create<memref::StoreOp>(loc, value, inputPack, ValueRange{line, lane});
      b.setInsertionPointAfter(copyIn);

      Value loaded = mvinBank(b, loc, inputPack, bank, lines);
      Value result = b.create<BankReluOp>(
          loc, bank.getType(), loaded, createI64Const(b, loc, 0),
          createI64Const(b, loc, lines), createI64Const(b, loc, lines));
      bank = mvoutBank(b, loc, outputPack, result, lines);
      b.create<FenceOp>(loc);

      auto copyOut = b.create<scf::ForOp>(loc, zero, countValue, one);
      b.setInsertionPointToStart(copyOut.getBody());
      index = copyOut.getInductionVar();
      source = b.create<arith::AddIOp>(loc, baseValue, index);
      row = b.create<arith::DivUIOp>(loc, source, columnsValue);
      column = b.create<arith::RemUIOp>(loc, source, columnsValue);
      line = b.create<arith::DivUIOp>(loc, index, four);
      lane = b.create<arith::RemUIOp>(loc, index, four);
      value = b.create<memref::LoadOp>(loc, outputPack, ValueRange{line, lane});
      b.create<memref::StoreOp>(loc, value, op.getOutput(),
                                ValueRange{row, column});
      b.setInsertionPointAfter(copyOut);
      b.create<memref::DeallocOp>(loc, inputPack);
      b.create<memref::DeallocOp>(loc, outputPack);
    }

    releaseBank(b, loc, bank);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populateReluBallLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ReluMatrixToBankSSAPattern>(patterns.getContext());
}
