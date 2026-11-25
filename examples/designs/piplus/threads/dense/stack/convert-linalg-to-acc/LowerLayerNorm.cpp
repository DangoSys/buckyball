//===- LowerLayerNorm.cpp - layernorm on LayerNormBall
//----------------------===//
//
// A bbacc.custom "layernorm" of x [rows, w], gamma [w] and beta [w] FP32
// (convert-linalg-to-hart), or with `rms` of x and gamma (RMSNorm), becomes
// one affine loop over blocks of rows, as a softmax does: each block moves
// into a bank, the Ball writes its normalization to another bank, and the
// block moves out. The parameter bank holds gamma, beta and eps for the whole
// loop: the CPU lays them out once in a buffer of 2 * ceil(w/4) + 1 bank rows
// (gamma, beta, eps in lane 0; with rms gamma and eps), one mvin.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Target/TargetRegistry.h"
#include "layernormOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace ::buddy::acc;

MemRefType bufferType(Value tensor);

LogicalResult lowerLayerNorm(ModuleOp module) {
  SmallVector<CustomOp> calls;
  module.walk([&](CustomOp op) {
    if (op.getKind() == "layernorm")
      calls.push_back(op);
  });
  for (CustomOp op : calls) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    auto type = cast<RankedTensorType>(op.getResult(0).getType());
    if (type.getRank() != 2)
      return op.emitError("layernorm takes [rows, w]");
    int64_t rows = type.getDimSize(0), w = type.getDimSize(1);
    bool rms = op->hasAttr("rms");
    // The strided DMA of a row's tail (a row that is not whole bank rows)
    // addresses the first 1024 bank rows.
    int64_t depth = target::getTarget().bankDepth;
    if (w % 4)
      depth = std::min<int64_t>(depth, 1024);
    // A row starts on a bank row and takes ceil(w/4) of them; the
    // parameters are gamma, beta (not of an RMSNorm) and eps.
    int64_t lines = (w + 3) / 4;
    if (w < 16 || w > 2048 || (rms ? lines : 2 * lines) + 1 > depth)
      return op.emitError("layernorm row exceeds LayerNormBall or its bank");
    int64_t block = std::min(rows, depth / lines);
    while (rows % block)
      --block;
    auto destination = dyn_cast<bufferization::MaterializeInDestinationOp>(
        *op.getResult(0).getUsers().begin());
    if (!destination || !op.getResult(0).hasOneUse())
      return op.emitError("layernorm must write its hart destination");
    Value input = bufferization::ToBufferOp::create(
        b, loc, bufferType(op.getInputs()[0]), op.getInputs()[0]);
    input.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
    Value output = bufferization::ToBufferOp::create(
        b, loc, bufferType(destination.getDest()), destination.getDest());

    // gamma, beta and eps in the layout of the parameter bank.
    auto vector = MemRefType::get({w}, b.getF32Type());
    int64_t tableLines = rms ? lines + 1 : 2 * lines + 1;
    Value table = memref::AllocOp::create(
        b, loc, MemRefType::get({tableLines * 4}, b.getF32Type()));
    linalg::FillOp::create(
        b, loc,
        ValueRange{arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0f))},
        ValueRange{table});
    for (auto [operand, at] :
         {std::pair<int64_t, int64_t>{1, 0}, {2, lines * 4}}) {
      if (rms && operand == 2)
        continue;
      Value source = bufferization::ToBufferOp::create(b, loc, vector,
                                                       op.getInputs()[operand]);
      source.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      Value part = memref::SubViewOp::create(
          b, loc, table, ArrayRef<OpFoldResult>{b.getIndexAttr(at)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(w)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      memref::CopyOp::create(b, loc, source, part);
    }
    memref::StoreOp::create(
        b, loc,
        arith::ConstantOp::create(b, loc, cast<FloatAttr>(op->getAttr("eps"))),
        table, ValueRange{index((tableLines - 1) * 4)});
    Value parameters = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    parameters = BankMvinOp::create(b, loc, b.getI64Type(), table, parameters,
                                    tableLines, 1);

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
    out = ::buddy::layernorm_ball::BankLayerNormOp::create(
        b, loc, b.getI64Type(), in, parameters, out, block, w, rms);
    out = move(output, out, true);
    BankReleaseOp::create(b, loc, in);
    BankReleaseOp::create(b, loc, out);
    b.setInsertionPointAfter(loop);
    BankReleaseOp::create(b, loc, parameters);

    Value written =
        bufferization::ToTensorOp::create(b, loc, type, output, true, true);
    destination.getResult().replaceAllUsesWith(written);
    destination.erase();
    op.erase();
  }
  return success();
}
