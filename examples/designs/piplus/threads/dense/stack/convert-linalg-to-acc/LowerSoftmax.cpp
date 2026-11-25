//===- LowerSoftmax.cpp - softmax on SoftmaxBall --------------------------===//
//
// A bbacc.custom "softmax" of [rows, w] FP32 (convert-linalg-to-hart) becomes
// one affine loop over blocks of rows: each block moves into a bank, the Ball
// writes its softmax to another bank, and the block moves out. A block is the
// most rows a bank holds (w/4 bank rows each) that divide the rows. With
// mxint8 (convert-linalg-to-hart's OutlineSoftmax.cpp) mxint8quant packs the
// block's softmax in a third bank, and the packed rows move out.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mxint8quantOps.h"
#include "softmaxOps.h"

using namespace mlir;
using namespace ::buddy::acc;

MemRefType bufferType(Value tensor);

LogicalResult lowerSoftmax(ModuleOp module) {
  SmallVector<CustomOp> calls;
  module.walk([&](CustomOp op) {
    if (op.getKind() == "softmax")
      calls.push_back(op);
  });
  for (CustomOp op : calls) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    auto type = cast<RankedTensorType>(op.getInputs()[0].getType());
    bool packed = op->hasAttr("mxint8");
    if (type.getRank() != 2)
      return op.emitError("softmax takes [rows, w]");
    int64_t rows = type.getDimSize(0), w = type.getDimSize(1);
    int64_t packedRows = w / 16 + (w + 511) / 512;
    // The strided DMA of a row's tail addresses the first 1024 bank rows.
    int64_t depth = std::min<int64_t>(target::getTarget().bankDepth, 1024);
    // A row starts on a bank row and takes ceil(w/4) of them.
    int64_t lines = (w + 3) / 4;
    if (w < 16 || w > 2048 || lines > depth)
      return op.emitError("softmax row exceeds SoftmaxBall or its bank");
    int64_t block = std::min(rows, depth / lines);
    while (rows % block)
      --block;
    auto destination = dyn_cast<bufferization::MaterializeInDestinationOp>(
        *op.getResult(0).getUsers().begin());
    if (!destination || !op.getResult(0).hasOneUse())
      return op.emitError("softmax must write its hart destination");
    Value input = bufferization::ToBufferOp::create(
        b, loc, bufferType(op.getInputs()[0]), op.getInputs()[0]);
    input.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
    Value output = bufferization::ToBufferOp::create(
        b, loc, bufferType(destination.getDest()), destination.getDest());

    // A stream loop counts from zero in unit steps: one step per block.
    auto loop =
        scf::ForOp::create(b, loc, index(0), index(rows / block), index(1));
    loop->setAttr("hart.stream", b.getUnitAttr());
    b.setInsertionPointToStart(loop.getBody());
    Value first =
        arith::MulIOp::create(b, loc, loop.getInductionVar(), index(block));
    auto rowsOf = [&](Value buffer) -> Value {
      return memref::SubViewOp::create(
          b, loc, buffer, ArrayRef<OpFoldResult>{first, b.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(block), b.getIndexAttr(w)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    };
    // Rows of a multiple of four fill whole bank rows: the block moves as
    // one. Otherwise each row moves on its own, its whole bank rows and then
    // the tail, so that no row overwrites the next in memory.
    auto move = [&](Value buffer, Value bank, bool out) -> Value {
      if (w % 4 == 0)
        return out ? Value(BankMvoutOp::create(b, loc, b.getI64Type(),
                                               rowsOf(buffer), bank,
                                               block * lines, 1))
                   : Value(BankMvinOp::create(b, loc, b.getI64Type(),
                                              rowsOf(buffer), bank,
                                              block * lines, 1));
      for (int64_t row = 0; row < block; ++row)
        for (auto [column, count] :
             {std::pair<int64_t, int64_t>{0, w / 4 * 4}, {w / 4 * 4, w % 4}}) {
          Value part = memref::SubViewOp::create(
              b, loc, buffer,
              ArrayRef<OpFoldResult>{
                  arith::AddIOp::create(b, loc, first, index(row)).getResult(),
                  b.getIndexAttr(column)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(count)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
          bool tail = column != 0;
          Value at = index(row * lines + column / 4);
          Value depthOf = index(tail ? 1 : w / 4);
          Value bytes = index(tail ? count * 4 : 16);
          bank = out ? Value(BankMvoutStridedOp::create(
                           b, loc, b.getI64Type(), part, bank, at, depthOf,
                           index(16), 0, 1, bytes, index(0)))
                     : Value(BankMvinStridedOp::create(
                           b, loc, b.getI64Type(), part, bank, at, depthOf,
                           index(16), 0, 1, bytes, index(0)));
        }
      return bank;
    };
    Value in = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    in = move(input, in, false);
    Value out = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    out = ::buddy::softmax_ball::BankSoftmaxOp::create(
        b, loc, b.getI64Type(), in, out, block, w,
        cast<IntegerAttr>(op->getAttr("scale_log2")).getInt());
    if (packed) {
      Value codes = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      codes = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
          b, loc, b.getI64Type(), out, codes, block, w, false, 0, 0, w / 4,
          false);
      BankMvoutOp::create(
          b, loc, b.getI64Type(),
          memref::SubViewOp::create(
              b, loc, output, ArrayRef<OpFoldResult>{first, b.getIndexAttr(0)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(block),
                                     b.getIndexAttr(resultType.getDimSize(1))},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)}),
          codes, block * packedRows, 1);
      BankReleaseOp::create(b, loc, codes);
    } else {
      out = move(output, out, true);
    }
    BankReleaseOp::create(b, loc, in);
    BankReleaseOp::create(b, loc, out);
    b.setInsertionPointAfter(loop);

    Value written = bufferization::ToTensorOp::create(b, loc, resultType,
                                                      output, true, true);
    destination.getResult().replaceAllUsesWith(written);
    destination.erase();
    op.erase();
  }
  return success();
}
